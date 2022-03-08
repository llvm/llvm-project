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
#include "flang/Lower/Mangler.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Reduction.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

/// Test if an ExtendedValue is absent.
static bool isAbsent(const fir::ExtendedValue &exv) {
  return !fir::getBase(exv);
}

/// Process calls to Maxval, Minval, Product, Sum intrinsic functions that
/// take a DIM argument.
template <typename FD>
static fir::ExtendedValue
genFuncDim(FD funcDim, mlir::Type resultType, fir::FirOpBuilder &builder,
           mlir::Location loc, Fortran::lower::StatementContext *stmtCtx,
           llvm::StringRef errMsg, mlir::Value array, fir::ExtendedValue dimArg,
           mlir::Value mask, int rank) {

  // Create mutable fir.box to be passed to the runtime for the result.
  mlir::Type resultArrayType = builder.getVarLenSeqTy(resultType, rank - 1);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  mlir::Value dim =
      isAbsent(dimArg)
          ? builder.createIntegerConstant(loc, builder.getIndexType(), 0)
          : fir::getBase(dimArg);
  funcDim(builder, loc, resultIrBox, array, dim, mask);

  fir::ExtendedValue res =
      fir::factory::genMutableBoxRead(builder, loc, resultMutableBox);
  return res.match(
      [&](const fir::ArrayBoxValue &box) -> fir::ExtendedValue {
        // Add cleanup code
        assert(stmtCtx);
        fir::FirOpBuilder *bldr = &builder;
        mlir::Value temp = box.getAddr();
        stmtCtx->attachCleanup(
            [=]() { bldr->create<fir::FreeMemOp>(loc, temp); });
        return box;
      },
      [&](const fir::CharArrayBoxValue &box) -> fir::ExtendedValue {
        // Add cleanup code
        assert(stmtCtx);
        fir::FirOpBuilder *bldr = &builder;
        mlir::Value temp = box.getAddr();
        stmtCtx->attachCleanup(
            [=]() { bldr->create<fir::FreeMemOp>(loc, temp); });
        return box;
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc, errMsg);
      });
}

/// Process calls to Product, Sum intrinsic functions
template <typename FN, typename FD>
static fir::ExtendedValue
genProdOrSum(FN func, FD funcDim, mlir::Type resultType,
             fir::FirOpBuilder &builder, mlir::Location loc,
             Fortran::lower::StatementContext *stmtCtx, llvm::StringRef errMsg,
             llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 3);

  // Handle required array argument
  fir::BoxValue arryTmp = builder.createBox(loc, args[0]);
  mlir::Value array = fir::getBase(arryTmp);
  int rank = arryTmp.rank();
  assert(rank >= 1);

  // Handle optional mask argument
  auto mask = isAbsent(args[2])
                  ? builder.create<fir::AbsentOp>(
                        loc, fir::BoxType::get(builder.getI1Type()))
                  : builder.createBox(loc, args[2]);

  bool absentDim = isAbsent(args[1]);

  // We call the type specific versions because the result is scalar
  // in the case below.
  if (absentDim || rank == 1) {
    mlir::Type ty = array.getType();
    mlir::Type arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    if (fir::isa_complex(eleTy)) {
      mlir::Value result = builder.createTemporary(loc, eleTy);
      func(builder, loc, array, mask, result);
      return builder.create<fir::LoadOp>(loc, result);
    }
    auto resultBox = builder.create<fir::AbsentOp>(
        loc, fir::BoxType::get(builder.getI1Type()));
    return func(builder, loc, array, mask, resultBox);
  }
  // Handle Product/Sum cases that have an array result.
  return genFuncDim(funcDim, resultType, builder, loc, stmtCtx, errMsg, array,
                    args[1], mask, rank);
}

// TODO error handling -> return a code or directly emit messages ?
struct IntrinsicLibrary {

  // Constructors.
  explicit IntrinsicLibrary(fir::FirOpBuilder &builder, mlir::Location loc,
                            Fortran::lower::StatementContext *stmtCtx = nullptr)
      : builder{builder}, loc{loc}, stmtCtx{stmtCtx} {}
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
  fir::ExtendedValue genSum(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  /// Define the different FIR generators that can be mapped to intrinsic to
  /// generate the related code. The intrinsic is lowered into an MLIR
  /// arith::AndIOp.
  using ElementalGenerator = decltype(&IntrinsicLibrary::genAbs);
  using ExtendedGenerator = decltype(&IntrinsicLibrary::genSum);
  using Generator = std::variant<ElementalGenerator, ExtendedGenerator>;

  template <typename GeneratorType>
  fir::ExtendedValue
  outlineInExtendedWrapper(GeneratorType, llvm::StringRef name,
                           llvm::Optional<mlir::Type> resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args);

  template <typename GeneratorType>
  mlir::FuncOp getWrapper(GeneratorType, llvm::StringRef name,
                          mlir::FunctionType, bool loadRefArguments = false);

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
  mlir::Value invokeGenerator(ExtendedGenerator generator,
                              mlir::Type resultType,
                              llvm::ArrayRef<mlir::Value> args);

  fir::FirOpBuilder &builder;
  mlir::Location loc;
  Fortran::lower::StatementContext *stmtCtx;
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
  // The following may be omitted in the table below.
  Fortran::lower::IntrinsicArgumentLoweringRules argLoweringRules = {};
  bool isElemental = true;
};

constexpr auto asValue = Fortran::lower::LowerIntrinsicArgAs::Value;
constexpr auto asBox = Fortran::lower::LowerIntrinsicArgAs::Box;
using I = IntrinsicLibrary;

/// Flag to indicate that an intrinsic argument has to be handled as
/// being dynamically optional (e.g. special handling when actual
/// argument is an optional variable in the current scope).
static constexpr bool handleDynamicOptional = true;

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
    {"sum",
     &I::genSum,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
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

/// fir::ExtendedValue to mlir::Value translation layer

fir::ExtendedValue toExtendedValue(mlir::Value val, fir::FirOpBuilder &builder,
                                   mlir::Location loc) {
  assert(val && "optional unhandled here");
  mlir::Type type = val.getType();
  mlir::Value base = val;
  mlir::IndexType indexType = builder.getIndexType();
  llvm::SmallVector<mlir::Value> extents;

  fir::factory::CharacterExprHelper charHelper{builder, loc};
  // FIXME: we may want to allow non character scalar here.
  if (charHelper.isCharacterScalar(type))
    return charHelper.toExtendedValue(val);

  if (auto refType = type.dyn_cast<fir::ReferenceType>())
    type = refType.getEleTy();

  if (auto arrayType = type.dyn_cast<fir::SequenceType>()) {
    type = arrayType.getEleTy();
    for (fir::SequenceType::Extent extent : arrayType.getShape()) {
      if (extent == fir::SequenceType::getUnknownExtent())
        break;
      extents.emplace_back(
          builder.createIntegerConstant(loc, indexType, extent));
    }
    // Last extent might be missing in case of assumed-size. If more extents
    // could not be deduced from type, that's an error (a fir.box should
    // have been used in the interface).
    if (extents.size() + 1 < arrayType.getShape().size())
      mlir::emitError(loc, "cannot retrieve array extents from type");
  } else if (type.isa<fir::BoxType>() || type.isa<fir::RecordType>()) {
    fir::emitFatalError(loc, "not yet implemented: descriptor or derived type");
  }

  if (!extents.empty())
    return fir::ArrayBoxValue{base, extents};
  return base;
}

mlir::Value toValue(const fir::ExtendedValue &val, fir::FirOpBuilder &builder,
                    mlir::Location loc) {
  if (const fir::CharBoxValue *charBox = val.getCharBox()) {
    mlir::Value buffer = charBox->getBuffer();
    if (buffer.getType().isa<fir::BoxCharType>())
      return buffer;
    return fir::factory::CharacterExprHelper{builder, loc}.createEmboxChar(
        buffer, charBox->getLen());
  }

  // FIXME: need to access other ExtendedValue variants and handle them
  // properly.
  return fir::getBase(val);
}

//===----------------------------------------------------------------------===//
// IntrinsicLibrary
//===----------------------------------------------------------------------===//

/// Emit a TODO error message for as yet unimplemented intrinsics.
static void crashOnMissingIntrinsic(mlir::Location loc, llvm::StringRef name) {
  TODO(loc, "missing intrinsic lowering: " + llvm::Twine(name));
}

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

template <>
fir::ExtendedValue
IntrinsicLibrary::genElementalCall<IntrinsicLibrary::ExtendedGenerator>(
    ExtendedGenerator generator, llvm::StringRef name, mlir::Type resultType,
    llvm::ArrayRef<fir::ExtendedValue> args, bool outline) {
  for (const fir::ExtendedValue &arg : args)
    if (!arg.getUnboxed() && !arg.getCharBox())
      fir::emitFatalError(loc, "nonscalar intrinsic argument");
  if (outline)
    return outlineInExtendedWrapper(generator, name, resultType, args);
  return std::invoke(generator, *this, resultType, args);
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

static fir::ExtendedValue
invokeHandler(IntrinsicLibrary::ExtendedGenerator generator,
              const IntrinsicHandler &handler,
              llvm::Optional<mlir::Type> resultType,
              llvm::ArrayRef<fir::ExtendedValue> args, bool outline,
              IntrinsicLibrary &lib) {
  assert(resultType && "expect intrinsic function");
  if (handler.isElemental)
    return lib.genElementalCall(generator, handler.name, *resultType, args,
                                outline);
  if (outline)
    return lib.outlineInExtendedWrapper(generator, handler.name, *resultType,
                                        args);
  return std::invoke(generator, lib, *resultType, args);
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

  if (!resultType)
    // Subroutine should have a handler, they are likely missing for now.
    crashOnMissingIntrinsic(loc, name);

  // Try the runtime if no special handler was defined for the
  // intrinsic being called. Maths runtime only has numerical elemental.
  // No optional arguments are expected at this point, the code will
  // crash if it gets absent optional.

  // FIXME: using toValue to get the type won't work with array arguments.
  llvm::SmallVector<mlir::Value> mlirArgs;
  for (const fir::ExtendedValue &extendedVal : args) {
    mlir::Value val = toValue(extendedVal, builder, loc);
    if (!val)
      // If an absent optional gets there, most likely its handler has just
      // not yet been defined.
      crashOnMissingIntrinsic(loc, name);
    mlirArgs.emplace_back(val);
  }
  mlir::FunctionType soughtFuncType =
      getFunctionType(*resultType, mlirArgs, builder);

  IntrinsicLibrary::RuntimeCallGenerator runtimeCallGenerator =
      getRuntimeCallGenerator(name, soughtFuncType);
  return genElementalCall(runtimeCallGenerator, name, *resultType, args,
                          /* outline */ true);
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

mlir::Value
IntrinsicLibrary::invokeGenerator(ExtendedGenerator generator,
                                  mlir::Type resultType,
                                  llvm::ArrayRef<mlir::Value> args) {
  llvm::SmallVector<fir::ExtendedValue> extendedArgs;
  for (mlir::Value arg : args)
    extendedArgs.emplace_back(toExtendedValue(arg, builder, loc));
  auto extendedResult = std::invoke(generator, *this, resultType, extendedArgs);
  return toValue(extendedResult, builder, loc);
}

template <typename GeneratorType>
mlir::FuncOp IntrinsicLibrary::getWrapper(GeneratorType generator,
                                          llvm::StringRef name,
                                          mlir::FunctionType funcType,
                                          bool loadRefArguments) {
  std::string wrapperName = fir::mangleIntrinsicProcedure(name, funcType);
  mlir::FuncOp function = builder.getNamedFunction(wrapperName);
  if (!function) {
    // First time this wrapper is needed, build it.
    function = builder.createFunction(loc, wrapperName, funcType);
    function->setAttr("fir.intrinsic", builder.getUnitAttr());
    auto internalLinkage = mlir::LLVM::linkage::Linkage::Internal;
    auto linkage =
        mlir::LLVM::LinkageAttr::get(builder.getContext(), internalLinkage);
    function->setAttr("llvm.linkage", linkage);
    function.addEntryBlock();

    // Create local context to emit code into the newly created function
    // This new function is not linked to a source file location, only
    // its calls will be.
    auto localBuilder =
        std::make_unique<fir::FirOpBuilder>(function, builder.getKindMap());
    localBuilder->setInsertionPointToStart(&function.front());
    // Location of code inside wrapper of the wrapper is independent from
    // the location of the intrinsic call.
    mlir::Location localLoc = localBuilder->getUnknownLoc();
    llvm::SmallVector<mlir::Value> localArguments;
    for (mlir::BlockArgument bArg : function.front().getArguments()) {
      auto refType = bArg.getType().dyn_cast<fir::ReferenceType>();
      if (loadRefArguments && refType) {
        auto loaded = localBuilder->create<fir::LoadOp>(localLoc, bArg);
        localArguments.push_back(loaded);
      } else {
        localArguments.push_back(bArg);
      }
    }

    IntrinsicLibrary localLib{*localBuilder, localLoc};

    assert(funcType.getNumResults() == 1 &&
           "expect one result for intrinsic function wrapper type");
    mlir::Type resultType = funcType.getResult(0);
    auto result =
        localLib.invokeGenerator(generator, resultType, localArguments);
    localBuilder->create<mlir::func::ReturnOp>(localLoc, result);
  } else {
    // Wrapper was already built, ensure it has the sought type
    assert(function.getType() == funcType &&
           "conflict between intrinsic wrapper types");
  }
  return function;
}

/// Helpers to detect absent optional (not yet supported in outlining).
bool static hasAbsentOptional(llvm::ArrayRef<fir::ExtendedValue> args) {
  for (const fir::ExtendedValue &arg : args)
    if (!fir::getBase(arg))
      return true;
  return false;
}

template <typename GeneratorType>
fir::ExtendedValue IntrinsicLibrary::outlineInExtendedWrapper(
    GeneratorType generator, llvm::StringRef name,
    llvm::Optional<mlir::Type> resultType,
    llvm::ArrayRef<fir::ExtendedValue> args) {
  if (hasAbsentOptional(args))
    TODO(loc, "cannot outline call to intrinsic " + llvm::Twine(name) +
                  " with absent optional argument");
  llvm::SmallVector<mlir::Value> mlirArgs;
  for (const auto &extendedVal : args)
    mlirArgs.emplace_back(toValue(extendedVal, builder, loc));
  mlir::FunctionType funcType = getFunctionType(resultType, mlirArgs, builder);
  mlir::FuncOp wrapper = getWrapper(generator, name, funcType);
  auto call = builder.create<fir::CallOp>(loc, wrapper, mlirArgs);
  if (resultType)
    return toExtendedValue(call.getResult(0), builder, loc);
  // Subroutine calls
  return mlir::Value{};
}

IntrinsicLibrary::RuntimeCallGenerator
IntrinsicLibrary::getRuntimeCallGenerator(llvm::StringRef name,
                                          mlir::FunctionType soughtFuncType) {
  mlir::FuncOp funcOp = getRuntimeFunction(loc, builder, name, soughtFuncType);
  if (!funcOp) {
    std::string buffer("not yet implemented: missing intrinsic lowering: ");
    llvm::raw_string_ostream sstream(buffer);
    sstream << name << "\nrequested type was: " << soughtFuncType << '\n';
    fir::emitFatalError(loc, buffer);
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

// SUM
fir::ExtendedValue
IntrinsicLibrary::genSum(mlir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {
  return genProdOrSum(fir::runtime::genSum, fir::runtime::genSumDim, resultType,
                      builder, loc, stmtCtx, "unexpected result for Sum", args);
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
                                 llvm::ArrayRef<fir::ExtendedValue> args,
                                 Fortran::lower::StatementContext &stmtCtx) {
  return IntrinsicLibrary{builder, loc, &stmtCtx}.genIntrinsicCall(
      name, resultType, args);
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
