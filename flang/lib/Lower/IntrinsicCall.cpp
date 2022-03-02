//===-- IntrinsicCall.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper routines for constructing the FIR dialect of MLIR. As FIR is a
// dialect of MLIR, it makes extensive use of MLIR interfaces and MLIR's coding
// style (https://mlir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/IntrinsicCall.h"
#include "flang/Common/static-multimap-view.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "flang-lower-intrinsic"

#define PGMATH_DECLARE
#include "flang/Evaluate/pgmath.h.inc"

/// Enums used to templatize and share lowering of MIN and MAX.
enum class Extremum { Min, Max };

// There are different ways to deal with NaNs in MIN and MAX.
// Known existing behaviors are listed below and can be selected for
// f18 MIN/MAX implementation.
enum class ExtremumBehavior {
  // Note: the Signaling/quiet aspect of NaNs in the behaviors below are
  // not described because there is no way to control/observe such aspect in
  // MLIR/LLVM yet. The IEEE behaviors come with requirements regarding this
  // aspect that are therefore currently not enforced. In the descriptions
  // below, NaNs can be signaling or quite. Returned NaNs may be signaling
  // if one of the input NaN was signaling but it cannot be guaranteed either.
  // Existing compilers using an IEEE behavior (gfortran) also do not fulfill
  // signaling/quiet requirements.
  IeeeMinMaximumNumber,
  // IEEE minimumNumber/maximumNumber behavior (754-2019, section 9.6):
  // If one of the argument is and number and the other is NaN, return the
  // number. If both arguements are NaN, return NaN.
  // Compilers: gfortran.
  IeeeMinMaximum,
  // IEEE minimum/maximum behavior (754-2019, section 9.6):
  // If one of the argument is NaN, return NaN.
  MinMaxss,
  // x86 minss/maxss behavior:
  // If the second argument is a number and the other is NaN, return the number.
  // In all other cases where at least one operand is NaN, return NaN.
  // Compilers: xlf (only for MAX), ifort, pgfortran -nollvm, and nagfor.
  PgfortranLlvm,
  // "Opposite of" x86 minss/maxss behavior:
  // If the first argument is a number and the other is NaN, return the
  // number.
  // In all other cases where at least one operand is NaN, return NaN.
  // Compilers: xlf (only for MIN), and pgfortran (with llvm).
  IeeeMinMaxNum
  // IEEE minNum/maxNum behavior (754-2008, section 5.3.1):
  // TODO: Not implemented.
  // It is the only behavior where the signaling/quiet aspect of a NaN argument
  // impacts if the result should be NaN or the argument that is a number.
  // LLVM/MLIR do not provide ways to observe this aspect, so it is not
  // possible to implement it without some target dependent runtime.
};

/// This file implements lowering of Fortran intrinsic procedures.
/// Intrinsics are lowered to a mix of FIR and MLIR operations as
/// well as call to runtime functions or LLVM intrinsics.

/// Lowering of intrinsic procedure calls is based on a map that associates
/// Fortran intrinsic generic names to FIR generator functions.
/// All generator functions are member functions of the IntrinsicLibrary class
/// and have the same interface.
/// If no generator is given for an intrinsic name, a math runtime library
/// is searched for an implementation and, if a runtime function is found,
/// a call is generated for it. LLVM intrinsics are handled as a math
/// runtime library here.

fir::ExtendedValue Fortran::lower::getAbsentIntrinsicArgument() {
  return fir::UnboxedValue{};
}

// TODO error handling -> return a code or directly emit messages ?
struct IntrinsicLibrary {

  // Constructors.
  explicit IntrinsicLibrary(fir::FirOpBuilder &builder, mlir::Location loc)
      : builder{builder}, loc{loc} {}
  IntrinsicLibrary() = delete;
  IntrinsicLibrary(const IntrinsicLibrary &) = delete;

  /// Generate FIR for call to Fortran intrinsic \p name with arguments \p arg
  /// and expected result type \p resultType.
  fir::ExtendedValue genIntrinsicCall(llvm::StringRef name,
                                      llvm::Optional<mlir::Type> resultType,
                                      llvm::ArrayRef<fir::ExtendedValue> arg);

  /// Search a runtime function that is associated to the generic intrinsic name
  /// and whose signature matches the intrinsic arguments and result types.
  /// If no such runtime function is found but a runtime function associated
  /// with the Fortran generic exists and has the same number of arguments,
  /// conversions will be inserted before and/or after the call. This is to
  /// mainly to allow 16 bits float support even-though little or no math
  /// runtime is currently available for it.
  mlir::Value genRuntimeCall(llvm::StringRef name, mlir::Type,
                             llvm::ArrayRef<mlir::Value>);

  using RuntimeCallGenerator = std::function<mlir::Value(
      fir::FirOpBuilder &, mlir::Location, llvm::ArrayRef<mlir::Value>)>;
  RuntimeCallGenerator
  getRuntimeCallGenerator(llvm::StringRef name,
                          mlir::FunctionType soughtFuncType);

  /// Lowering for the ABS intrinsic. The ABS intrinsic expects one argument in
  /// the llvm::ArrayRef. The ABS intrinsic is lowered into MLIR/FIR operation
  /// if the argument is an integer, into llvm intrinsics if the argument is
  /// real and to the `hypot` math routine if the argument is of complex type.
  mlir::Value genAbs(mlir::Type, llvm::ArrayRef<mlir::Value>);
  template <Extremum, ExtremumBehavior>
  mlir::Value genExtremum(mlir::Type, llvm::ArrayRef<mlir::Value>);
  /// Lowering for the IAND intrinsic. The IAND intrinsic expects two arguments
  /// in the llvm::ArrayRef.
  mlir::Value genIand(mlir::Type, llvm::ArrayRef<mlir::Value>);
  /// Define the different FIR generators that can be mapped to intrinsic to
  /// generate the related code. The intrinsic is lowered into an MLIR
  /// arith::AndIOp.
  using ElementalGenerator = decltype(&IntrinsicLibrary::genAbs);
  using Generator = std::variant<ElementalGenerator>;

  /// Generate calls to ElementalGenerator, handling the elemental aspects
  template <typename GeneratorType>
  fir::ExtendedValue
  genElementalCall(GeneratorType, llvm::StringRef name, mlir::Type resultType,
                   llvm::ArrayRef<fir::ExtendedValue> args, bool outline);

  /// Helper to invoke code generator for the intrinsics given arguments.
  mlir::Value invokeGenerator(ElementalGenerator generator,
                              mlir::Type resultType,
                              llvm::ArrayRef<mlir::Value> args);
  mlir::Value invokeGenerator(RuntimeCallGenerator generator,
                              mlir::Type resultType,
                              llvm::ArrayRef<mlir::Value> args);
  fir::FirOpBuilder &builder;
  mlir::Location loc;
};

struct IntrinsicDummyArgument {
  const char *name = nullptr;
  Fortran::lower::LowerIntrinsicArgAs lowerAs =
      Fortran::lower::LowerIntrinsicArgAs::Value;
  bool handleDynamicOptional = false;
};

struct Fortran::lower::IntrinsicArgumentLoweringRules {
  /// There is no more than 7 non repeated arguments in Fortran intrinsics.
  IntrinsicDummyArgument args[7];
  constexpr bool hasDefaultRules() const { return args[0].name == nullptr; }
};

/// Structure describing what needs to be done to lower intrinsic "name".
struct IntrinsicHandler {
  const char *name;
  IntrinsicLibrary::Generator generator;
  Fortran::lower::IntrinsicArgumentLoweringRules argLoweringRules = {};
};

using I = IntrinsicLibrary;

/// Table that drives the fir generation depending on the intrinsic.
/// one to one mapping with Fortran arguments. If no mapping is
/// defined here for a generic intrinsic, genRuntimeCall will be called
/// to look for a match in the runtime a emit a call. Note that the argument
/// lowering rules for an intrinsic need to be provided only if at least one
/// argument must not be lowered by value. In which case, the lowering rules
/// should be provided for all the intrinsic arguments for completeness.
static constexpr IntrinsicHandler handlers[]{
    {"abs", &I::genAbs},
    {"iand", &I::genIand},
};

static const IntrinsicHandler *findIntrinsicHandler(llvm::StringRef name) {
  auto compare = [](const IntrinsicHandler &handler, llvm::StringRef name) {
    return name.compare(handler.name) > 0;
  };
  auto result =
      std::lower_bound(std::begin(handlers), std::end(handlers), name, compare);
  return result != std::end(handlers) && result->name == name ? result
                                                              : nullptr;
}

//===----------------------------------------------------------------------===//
// Math runtime description and matching utility
//===----------------------------------------------------------------------===//

/// Command line option to modify math runtime version used to implement
/// intrinsics.
enum MathRuntimeVersion { fastVersion, llvmOnly };
llvm::cl::opt<MathRuntimeVersion> mathRuntimeVersion(
    "math-runtime", llvm::cl::desc("Select math runtime version:"),
    llvm::cl::values(
        clEnumValN(fastVersion, "fast", "use pgmath fast runtime"),
        clEnumValN(llvmOnly, "llvm",
                   "only use LLVM intrinsics (may be incomplete)")),
    llvm::cl::init(fastVersion));

struct RuntimeFunction {
  // llvm::StringRef comparison operator are not constexpr, so use string_view.
  using Key = std::string_view;
  // Needed for implicit compare with keys.
  constexpr operator Key() const { return key; }
  Key key; // intrinsic name
  llvm::StringRef symbol;
  fir::runtime::FuncTypeBuilderFunc typeGenerator;
};

#define RUNTIME_STATIC_DESCRIPTION(name, func)                                 \
  {#name, #func, fir::runtime::RuntimeTableKey<decltype(func)>::getTypeModel()},
static constexpr RuntimeFunction pgmathFast[] = {
#define PGMATH_FAST
#define PGMATH_USE_ALL_TYPES(name, func) RUNTIME_STATIC_DESCRIPTION(name, func)
#include "flang/Evaluate/pgmath.h.inc"
};

static mlir::FunctionType genF32F32FuncType(mlir::MLIRContext *context) {
  mlir::Type t = mlir::FloatType::getF32(context);
  return mlir::FunctionType::get(context, {t}, {t});
}

static mlir::FunctionType genF64F64FuncType(mlir::MLIRContext *context) {
  mlir::Type t = mlir::FloatType::getF64(context);
  return mlir::FunctionType::get(context, {t}, {t});
}

static mlir::FunctionType genF32F32F32FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF32(context);
  return mlir::FunctionType::get(context, {t, t}, {t});
}

static mlir::FunctionType genF64F64F64FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF64(context);
  return mlir::FunctionType::get(context, {t, t}, {t});
}

// TODO : Fill-up this table with more intrinsic.
// Note: These are also defined as operations in LLVM dialect. See if this
// can be use and has advantages.
static constexpr RuntimeFunction llvmIntrinsics[] = {
    {"abs", "llvm.fabs.f32", genF32F32FuncType},
    {"abs", "llvm.fabs.f64", genF64F64FuncType},
    {"pow", "llvm.pow.f32", genF32F32F32FuncType},
    {"pow", "llvm.pow.f64", genF64F64F64FuncType},
};

// This helper class computes a "distance" between two function types.
// The distance measures how many narrowing conversions of actual arguments
// and result of "from" must be made in order to use "to" instead of "from".
// For instance, the distance between ACOS(REAL(10)) and ACOS(REAL(8)) is
// greater than the one between ACOS(REAL(10)) and ACOS(REAL(16)). This means
// if no implementation of ACOS(REAL(10)) is available, it is better to use
// ACOS(REAL(16)) with casts rather than ACOS(REAL(8)).
// Note that this is not a symmetric distance and the order of "from" and "to"
// arguments matters, d(foo, bar) may not be the same as d(bar, foo) because it
// may be safe to replace foo by bar, but not the opposite.
class FunctionDistance {
public:
  FunctionDistance() : infinite{true} {}

  FunctionDistance(mlir::FunctionType from, mlir::FunctionType to) {
    unsigned nInputs = from.getNumInputs();
    unsigned nResults = from.getNumResults();
    if (nResults != to.getNumResults() || nInputs != to.getNumInputs()) {
      infinite = true;
    } else {
      for (decltype(nInputs) i = 0; i < nInputs && !infinite; ++i)
        addArgumentDistance(from.getInput(i), to.getInput(i));
      for (decltype(nResults) i = 0; i < nResults && !infinite; ++i)
        addResultDistance(to.getResult(i), from.getResult(i));
    }
  }

  /// Beware both d1.isSmallerThan(d2) *and* d2.isSmallerThan(d1) may be
  /// false if both d1 and d2 are infinite. This implies that
  ///  d1.isSmallerThan(d2) is not equivalent to !d2.isSmallerThan(d1)
  bool isSmallerThan(const FunctionDistance &d) const {
    return !infinite &&
           (d.infinite || std::lexicographical_compare(
                              conversions.begin(), conversions.end(),
                              d.conversions.begin(), d.conversions.end()));
  }

  bool isLosingPrecision() const {
    return conversions[narrowingArg] != 0 || conversions[extendingResult] != 0;
  }

  bool isInfinite() const { return infinite; }

private:
  enum class Conversion { Forbidden, None, Narrow, Extend };

  void addArgumentDistance(mlir::Type from, mlir::Type to) {
    switch (conversionBetweenTypes(from, to)) {
    case Conversion::Forbidden:
      infinite = true;
      break;
    case Conversion::None:
      break;
    case Conversion::Narrow:
      conversions[narrowingArg]++;
      break;
    case Conversion::Extend:
      conversions[nonNarrowingArg]++;
      break;
    }
  }

  void addResultDistance(mlir::Type from, mlir::Type to) {
    switch (conversionBetweenTypes(from, to)) {
    case Conversion::Forbidden:
      infinite = true;
      break;
    case Conversion::None:
      break;
    case Conversion::Narrow:
      conversions[nonExtendingResult]++;
      break;
    case Conversion::Extend:
      conversions[extendingResult]++;
      break;
    }
  }

  // Floating point can be mlir::FloatType or fir::real
  static unsigned getFloatingPointWidth(mlir::Type t) {
    if (auto f{t.dyn_cast<mlir::FloatType>()})
      return f.getWidth();
    // FIXME: Get width another way for fir.real/complex
    // - use fir/KindMapping.h and llvm::Type
    // - or use evaluate/type.h
    if (auto r{t.dyn_cast<fir::RealType>()})
      return r.getFKind() * 4;
    if (auto cplx{t.dyn_cast<fir::ComplexType>()})
      return cplx.getFKind() * 4;
    llvm_unreachable("not a floating-point type");
  }

  static Conversion conversionBetweenTypes(mlir::Type from, mlir::Type to) {
    if (from == to)
      return Conversion::None;

    if (auto fromIntTy{from.dyn_cast<mlir::IntegerType>()}) {
      if (auto toIntTy{to.dyn_cast<mlir::IntegerType>()}) {
        return fromIntTy.getWidth() > toIntTy.getWidth() ? Conversion::Narrow
                                                         : Conversion::Extend;
      }
    }

    if (fir::isa_real(from) && fir::isa_real(to)) {
      return getFloatingPointWidth(from) > getFloatingPointWidth(to)
                 ? Conversion::Narrow
                 : Conversion::Extend;
    }

    if (auto fromCplxTy{from.dyn_cast<fir::ComplexType>()}) {
      if (auto toCplxTy{to.dyn_cast<fir::ComplexType>()}) {
        return getFloatingPointWidth(fromCplxTy) >
                       getFloatingPointWidth(toCplxTy)
                   ? Conversion::Narrow
                   : Conversion::Extend;
      }
    }
    // Notes:
    // - No conversion between character types, specialization of runtime
    // functions should be made instead.
    // - It is not clear there is a use case for automatic conversions
    // around Logical and it may damage hidden information in the physical
    // storage so do not do it.
    return Conversion::Forbidden;
  }

  // Below are indexes to access data in conversions.
  // The order in data does matter for lexicographical_compare
  enum {
    narrowingArg = 0,   // usually bad
    extendingResult,    // usually bad
    nonExtendingResult, // usually ok
    nonNarrowingArg,    // usually ok
    dataSize
  };

  std::array<int, dataSize> conversions = {};
  bool infinite = false; // When forbidden conversion or wrong argument number
};

/// Build mlir::FuncOp from runtime symbol description and add
/// fir.runtime attribute.
static mlir::FuncOp getFuncOp(mlir::Location loc, fir::FirOpBuilder &builder,
                              const RuntimeFunction &runtime) {
  mlir::FuncOp function = builder.addNamedFunction(
      loc, runtime.symbol, runtime.typeGenerator(builder.getContext()));
  function->setAttr("fir.runtime", builder.getUnitAttr());
  return function;
}

/// Select runtime function that has the smallest distance to the intrinsic
/// function type and that will not imply narrowing arguments or extending the
/// result.
/// If nothing is found, the mlir::FuncOp will contain a nullptr.
mlir::FuncOp searchFunctionInLibrary(
    mlir::Location loc, fir::FirOpBuilder &builder,
    const Fortran::common::StaticMultimapView<RuntimeFunction> &lib,
    llvm::StringRef name, mlir::FunctionType funcType,
    const RuntimeFunction **bestNearMatch,
    FunctionDistance &bestMatchDistance) {
  std::pair<const RuntimeFunction *, const RuntimeFunction *> range =
      lib.equal_range(name);
  for (auto iter = range.first; iter != range.second && iter; ++iter) {
    const RuntimeFunction &impl = *iter;
    mlir::FunctionType implType = impl.typeGenerator(builder.getContext());
    if (funcType == implType)
      return getFuncOp(loc, builder, impl); // exact match

    FunctionDistance distance(funcType, implType);
    if (distance.isSmallerThan(bestMatchDistance)) {
      *bestNearMatch = &impl;
      bestMatchDistance = std::move(distance);
    }
  }
  return {};
}

/// Search runtime for the best runtime function given an intrinsic name
/// and interface. The interface may not be a perfect match in which case
/// the caller is responsible to insert argument and return value conversions.
/// If nothing is found, the mlir::FuncOp will contain a nullptr.
static mlir::FuncOp getRuntimeFunction(mlir::Location loc,
                                       fir::FirOpBuilder &builder,
                                       llvm::StringRef name,
                                       mlir::FunctionType funcType) {
  const RuntimeFunction *bestNearMatch = nullptr;
  FunctionDistance bestMatchDistance{};
  mlir::FuncOp match;
  using RtMap = Fortran::common::StaticMultimapView<RuntimeFunction>;
  static constexpr RtMap pgmathF(pgmathFast);
  static_assert(pgmathF.Verify() && "map must be sorted");
  if (mathRuntimeVersion == fastVersion) {
    match = searchFunctionInLibrary(loc, builder, pgmathF, name, funcType,
                                    &bestNearMatch, bestMatchDistance);
  } else {
    assert(mathRuntimeVersion == llvmOnly && "unknown math runtime");
  }
  if (match)
    return match;

  // Go through llvm intrinsics if not exact match in libpgmath or if
  // mathRuntimeVersion == llvmOnly
  static constexpr RtMap llvmIntr(llvmIntrinsics);
  static_assert(llvmIntr.Verify() && "map must be sorted");
  if (mlir::FuncOp exactMatch =
          searchFunctionInLibrary(loc, builder, llvmIntr, name, funcType,
                                  &bestNearMatch, bestMatchDistance))
    return exactMatch;

  if (bestNearMatch != nullptr) {
    if (bestMatchDistance.isLosingPrecision()) {
      // Using this runtime version requires narrowing the arguments
      // or extending the result. It is not numerically safe. There
      // is currently no quad math library that was described in
      // lowering and could be used here. Emit an error and continue
      // generating the code with the narrowing cast so that the user
      // can get a complete list of the problematic intrinsic calls.
      std::string message("TODO: no math runtime available for '");
      llvm::raw_string_ostream sstream(message);
      if (name == "pow") {
        assert(funcType.getNumInputs() == 2 &&
               "power operator has two arguments");
        sstream << funcType.getInput(0) << " ** " << funcType.getInput(1);
      } else {
        sstream << name << "(";
        if (funcType.getNumInputs() > 0)
          sstream << funcType.getInput(0);
        for (mlir::Type argType : funcType.getInputs().drop_front())
          sstream << ", " << argType;
        sstream << ")";
      }
      sstream << "'";
      mlir::emitError(loc, message);
    }
    return getFuncOp(loc, builder, *bestNearMatch);
  }
  return {};
}

/// Helpers to get function type from arguments and result type.
static mlir::FunctionType getFunctionType(llvm::Optional<mlir::Type> resultType,
                                          llvm::ArrayRef<mlir::Value> arguments,
                                          fir::FirOpBuilder &builder) {
  llvm::SmallVector<mlir::Type> argTypes;
  for (mlir::Value arg : arguments)
    argTypes.push_back(arg.getType());
  llvm::SmallVector<mlir::Type> resTypes;
  if (resultType)
    resTypes.push_back(*resultType);
  return mlir::FunctionType::get(builder.getModule().getContext(), argTypes,
                                 resTypes);
}
//===----------------------------------------------------------------------===//
// IntrinsicLibrary
//===----------------------------------------------------------------------===//

template <typename GeneratorType>
fir::ExtendedValue IntrinsicLibrary::genElementalCall(
    GeneratorType generator, llvm::StringRef name, mlir::Type resultType,
    llvm::ArrayRef<fir::ExtendedValue> args, bool outline) {
  llvm::SmallVector<mlir::Value> scalarArgs;
  for (const fir::ExtendedValue &arg : args)
    if (arg.getUnboxed() || arg.getCharBox())
      scalarArgs.emplace_back(fir::getBase(arg));
    else
      fir::emitFatalError(loc, "nonscalar intrinsic argument");
  return invokeGenerator(generator, resultType, scalarArgs);
}

static fir::ExtendedValue
invokeHandler(IntrinsicLibrary::ElementalGenerator generator,
              const IntrinsicHandler &handler,
              llvm::Optional<mlir::Type> resultType,
              llvm::ArrayRef<fir::ExtendedValue> args, bool outline,
              IntrinsicLibrary &lib) {
  assert(resultType && "expect elemental intrinsic to be functions");
  return lib.genElementalCall(generator, handler.name, *resultType, args,
                              outline);
}

fir::ExtendedValue
IntrinsicLibrary::genIntrinsicCall(llvm::StringRef name,
                                   llvm::Optional<mlir::Type> resultType,
                                   llvm::ArrayRef<fir::ExtendedValue> args) {
  if (const IntrinsicHandler *handler = findIntrinsicHandler(name)) {
    bool outline = false;
    return std::visit(
        [&](auto &generator) -> fir::ExtendedValue {
          return invokeHandler(generator, *handler, resultType, args, outline,
                               *this);
        },
        handler->generator);
  }

  TODO(loc, "genIntrinsicCall runtime");
  return {};
}

mlir::Value
IntrinsicLibrary::invokeGenerator(ElementalGenerator generator,
                                  mlir::Type resultType,
                                  llvm::ArrayRef<mlir::Value> args) {
  return std::invoke(generator, *this, resultType, args);
}

mlir::Value
IntrinsicLibrary::invokeGenerator(RuntimeCallGenerator generator,
                                  mlir::Type resultType,
                                  llvm::ArrayRef<mlir::Value> args) {
  return generator(builder, loc, args);
}
IntrinsicLibrary::RuntimeCallGenerator
IntrinsicLibrary::getRuntimeCallGenerator(llvm::StringRef name,
                                          mlir::FunctionType soughtFuncType) {
  mlir::FuncOp funcOp = getRuntimeFunction(loc, builder, name, soughtFuncType);
  if (!funcOp) {
    mlir::emitError(loc,
                    "TODO: missing intrinsic lowering: " + llvm::Twine(name));
    llvm::errs() << "requested type was: " << soughtFuncType << "\n";
    exit(1);
  }

  mlir::FunctionType actualFuncType = funcOp.getType();
  assert(actualFuncType.getNumResults() == soughtFuncType.getNumResults() &&
         actualFuncType.getNumInputs() == soughtFuncType.getNumInputs() &&
         actualFuncType.getNumResults() == 1 && "Bad intrinsic match");

  return [funcOp, actualFuncType,
          soughtFuncType](fir::FirOpBuilder &builder, mlir::Location loc,
                          llvm::ArrayRef<mlir::Value> args) {
    llvm::SmallVector<mlir::Value> convertedArguments;
    for (auto [fst, snd] : llvm::zip(actualFuncType.getInputs(), args))
      convertedArguments.push_back(builder.createConvert(loc, fst, snd));
    auto call = builder.create<fir::CallOp>(loc, funcOp, convertedArguments);
    mlir::Type soughtType = soughtFuncType.getResult(0);
    return builder.createConvert(loc, soughtType, call.getResult(0));
  };
}
//===----------------------------------------------------------------------===//
// Code generators for the intrinsic
//===----------------------------------------------------------------------===//

mlir::Value IntrinsicLibrary::genRuntimeCall(llvm::StringRef name,
                                             mlir::Type resultType,
                                             llvm::ArrayRef<mlir::Value> args) {
  mlir::FunctionType soughtFuncType =
      getFunctionType(resultType, args, builder);
  return getRuntimeCallGenerator(name, soughtFuncType)(builder, loc, args);
}

// ABS
mlir::Value IntrinsicLibrary::genAbs(mlir::Type resultType,
                                     llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);
  mlir::Value arg = args[0];
  mlir::Type type = arg.getType();
  if (fir::isa_real(type)) {
    // Runtime call to fp abs. An alternative would be to use mlir
    // math::AbsFOp but it does not support all fir floating point types.
    return genRuntimeCall("abs", resultType, args);
  }
  if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    // At the time of this implementation there is no abs op in mlir.
    // So, implement abs here without branching.
    mlir::Value shift =
        builder.createIntegerConstant(loc, intType, intType.getWidth() - 1);
    auto mask = builder.create<mlir::arith::ShRSIOp>(loc, arg, shift);
    auto xored = builder.create<mlir::arith::XOrIOp>(loc, arg, mask);
    return builder.create<mlir::arith::SubIOp>(loc, xored, mask);
  }
  if (fir::isa_complex(type)) {
    // Use HYPOT to fulfill the no underflow/overflow requirement.
    auto parts = fir::factory::Complex{builder, loc}.extractParts(arg);
    llvm::SmallVector<mlir::Value> args = {parts.first, parts.second};
    return genRuntimeCall("hypot", resultType, args);
  }
  llvm_unreachable("unexpected type in ABS argument");
}

// IAND
mlir::Value IntrinsicLibrary::genIand(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  return builder.create<mlir::arith::AndIOp>(loc, args[0], args[1]);
}

// Compare two FIR values and return boolean result as i1.
template <Extremum extremum, ExtremumBehavior behavior>
static mlir::Value createExtremumCompare(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         mlir::Value left, mlir::Value right) {
  static constexpr mlir::arith::CmpIPredicate integerPredicate =
      extremum == Extremum::Max ? mlir::arith::CmpIPredicate::sgt
                                : mlir::arith::CmpIPredicate::slt;
  static constexpr mlir::arith::CmpFPredicate orderedCmp =
      extremum == Extremum::Max ? mlir::arith::CmpFPredicate::OGT
                                : mlir::arith::CmpFPredicate::OLT;
  mlir::Type type = left.getType();
  mlir::Value result;
  if (fir::isa_real(type)) {
    // Note: the signaling/quit aspect of the result required by IEEE
    // cannot currently be obtained with LLVM without ad-hoc runtime.
    if constexpr (behavior == ExtremumBehavior::IeeeMinMaximumNumber) {
      // Return the number if one of the inputs is NaN and the other is
      // a number.
      auto leftIsResult =
          builder.create<mlir::arith::CmpFOp>(loc, orderedCmp, left, right);
      auto rightIsNan = builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::UNE, right, right);
      result =
          builder.create<mlir::arith::OrIOp>(loc, leftIsResult, rightIsNan);
    } else if constexpr (behavior == ExtremumBehavior::IeeeMinMaximum) {
      // Always return NaNs if one the input is NaNs
      auto leftIsResult =
          builder.create<mlir::arith::CmpFOp>(loc, orderedCmp, left, right);
      auto leftIsNan = builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::UNE, left, left);
      result = builder.create<mlir::arith::OrIOp>(loc, leftIsResult, leftIsNan);
    } else if constexpr (behavior == ExtremumBehavior::MinMaxss) {
      // If the left is a NaN, return the right whatever it is.
      result =
          builder.create<mlir::arith::CmpFOp>(loc, orderedCmp, left, right);
    } else if constexpr (behavior == ExtremumBehavior::PgfortranLlvm) {
      // If one of the operand is a NaN, return left whatever it is.
      static constexpr auto unorderedCmp =
          extremum == Extremum::Max ? mlir::arith::CmpFPredicate::UGT
                                    : mlir::arith::CmpFPredicate::ULT;
      result =
          builder.create<mlir::arith::CmpFOp>(loc, unorderedCmp, left, right);
    } else {
      // TODO: ieeeMinNum/ieeeMaxNum
      static_assert(behavior == ExtremumBehavior::IeeeMinMaxNum,
                    "ieeeMinNum/ieeeMaxNum behavior not implemented");
    }
  } else if (fir::isa_integer(type)) {
    result =
        builder.create<mlir::arith::CmpIOp>(loc, integerPredicate, left, right);
  } else if (fir::isa_char(type)) {
    // TODO: ! character min and max is tricky because the result
    // length is the length of the longest argument!
    // So we may need a temp.
    TODO(loc, "CHARACTER min and max");
  }
  assert(result && "result must be defined");
  return result;
}

// MIN and MAX
template <Extremum extremum, ExtremumBehavior behavior>
mlir::Value IntrinsicLibrary::genExtremum(mlir::Type,
                                          llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() >= 1);
  mlir::Value result = args[0];
  for (auto arg : args.drop_front()) {
    mlir::Value mask =
        createExtremumCompare<extremum, behavior>(loc, builder, result, arg);
    result = builder.create<mlir::arith::SelectOp>(loc, mask, result, arg);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Argument lowering rules interface
//===----------------------------------------------------------------------===//

const Fortran::lower::IntrinsicArgumentLoweringRules *
Fortran::lower::getIntrinsicArgumentLowering(llvm::StringRef intrinsicName) {
  if (const IntrinsicHandler *handler = findIntrinsicHandler(intrinsicName))
    if (!handler->argLoweringRules.hasDefaultRules())
      return &handler->argLoweringRules;
  return nullptr;
}

/// Return how argument \p argName should be lowered given the rules for the
/// intrinsic function.
Fortran::lower::ArgLoweringRule Fortran::lower::lowerIntrinsicArgumentAs(
    mlir::Location loc, const IntrinsicArgumentLoweringRules &rules,
    llvm::StringRef argName) {
  for (const IntrinsicDummyArgument &arg : rules.args) {
    if (arg.name && arg.name == argName)
      return {arg.lowerAs, arg.handleDynamicOptional};
  }
  fir::emitFatalError(
      loc, "internal: unknown intrinsic argument name in lowering '" + argName +
               "'");
}

//===----------------------------------------------------------------------===//
// Public intrinsic call helpers
//===----------------------------------------------------------------------===//

fir::ExtendedValue
Fortran::lower::genIntrinsicCall(fir::FirOpBuilder &builder, mlir::Location loc,
                                 llvm::StringRef name,
                                 llvm::Optional<mlir::Type> resultType,
                                 llvm::ArrayRef<fir::ExtendedValue> args) {
  return IntrinsicLibrary{builder, loc}.genIntrinsicCall(name, resultType,
                                                         args);
}

mlir::Value Fortran::lower::genMax(fir::FirOpBuilder &builder,
                                   mlir::Location loc,
                                   llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() > 0 && "max requires at least one argument");
  return IntrinsicLibrary{builder, loc}
      .genExtremum<Extremum::Max, ExtremumBehavior::MinMaxss>(args[0].getType(),
                                                              args);
}

mlir::Value Fortran::lower::genPow(fir::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Type type,
                                   mlir::Value x, mlir::Value y) {
  return IntrinsicLibrary{builder, loc}.genRuntimeCall("pow", type, {x, y});
}
