//===-- Intrinsics.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder routines for constructing the FIR dialect of MLIR. As FIR is a
// dialect of MLIR, it makes extensive use of MLIR interfaces and MLIR's coding
// style (https://mlir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Intrinsics.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/Runtime.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <unordered_map> // FIXME: must be removed
#include <utility>

namespace Fortran::lower {

/// MathRuntimeLibrary maps Fortran generic intrinsic names to runtime function
/// signatures. There is no guarantee that that runtime functions are available
/// for all intrinsic functions and possible types.
/// To be easy and fast to use, this class holds a map and uses
/// mlir::FunctionType to represent the runtime function type. This imply that
/// MathRuntimeLibrary cannot be constexpr built and requires an
/// mlir::MLIRContext to be built. Its constructor uses a constexpr table
/// description of the runtime. The runtime functions are not declared into the
/// mlir::module until there is a query that needs them. This is to avoid
/// polluting the FIR/LLVM IR dumps with unused functions.
class MathRuntimeLibrary {
public:
  /// The map key are Fortran generic intrinsic names.
  using Key = llvm::StringRef;
  struct Hash { // Need custom hash for this kind of key
    size_t operator()(const Key &k) const { return llvm::hash_value(k); }
  };
  /// Runtime function description that is sufficient to build an
  /// mlir::FuncOp and to compare function types.
  struct RuntimeFunction {
    RuntimeFunction(llvm::StringRef n, mlir::FunctionType t)
        : symbol{n}, type{t} {}
    llvm::StringRef symbol;
    mlir::FunctionType type;
  };
  using Map = std::unordered_multimap<Key, RuntimeFunction, Hash>;

  MathRuntimeLibrary(IntrinsicLibrary::Version, mlir::MLIRContext &);

  /// Probe the intrinsic library for a certain intrinsic and get/build the
  /// related mlir::FuncOp if a runtime description is found.
  /// Also add a unit attribute "fir.runtime" to the function so that later
  /// it is possible to quickly know what function are intrinsics vs users.
  llvm::Optional<mlir::FuncOp> getFunction(Fortran::lower::FirOpBuilder &,
                                           llvm::StringRef,
                                           mlir::FunctionType) const;

private:
  mlir::FuncOp getFuncOp(Fortran::lower::FirOpBuilder &builder,
                         const RuntimeFunction &runtime) const;
  Map library;
};

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

/// The implementation of IntrinsicLibrary is based on a map that associates
/// Fortran intrinsics generic names to the related FIR generator functions.
/// All generator functions are member functions of the Implementation class
/// and they all take the same context argument that contains the name and
/// arguments of the Fortran intrinsics call to lower among other things.
/// A same FIR generator function may be able to generate the FIR for several
/// intrinsics. For instance genRuntimeCall tries to find a runtime
/// functions that matches the Fortran intrinsic call and generate the
/// operations to call this functions if it was found.
/// IntrinsicLibrary holds a constant MathRuntimeLibrary that it uses to
/// find and place call to math runtime functions. This library is built
/// when the Implementation is built. Because of this, Implementation is
/// not cheap to build and it should be kept as long as possible.

// TODO it is unclear how optional argument are handled
// TODO error handling -> return a code or directly emit messages ?
class IntrinsicLibrary::Implementation {
public:
  Implementation(Version v, mlir::MLIRContext &c) : runtime{v, c} {}
  inline mlir::Value genval(mlir::Location loc,
                            Fortran::lower::FirOpBuilder &builder,
                            llvm::StringRef name, mlir::Type resultType,
                            llvm::ArrayRef<mlir::Value> args);

private:
  // Info needed by Generators is passed in Context struct to keep Generator
  // signatures modification easy.
  struct Context {
    mlir::Location loc;
    Fortran::lower::FirOpBuilder *builder = nullptr;
    llvm::StringRef name;
    llvm::ArrayRef<mlir::Value> arguments;
    mlir::FunctionType funcType;
    mlir::ModuleOp getModuleOp() { return builder->getModule(); }
    mlir::MLIRContext *getMLIRContext() { return getModuleOp().getContext(); }
    mlir::Type getResultType() {
      assert(funcType.getNumResults() == 1);
      return funcType.getResult(0);
    }
  };

  /// Define the different FIR generators that can be mapped to intrinsic to
  /// generate the related code.
  using Generator = mlir::Value (*)(Context &, MathRuntimeLibrary &);

  /// Search a runtime function that is associated to the generic intrinsic name
  /// and whose signature matches the intrinsic arguments and result types.
  /// If no such runtime function is found but a runtime function associated
  /// with the Fortran generic exists and has the same number of arguments,
  /// conversions will be inserted before and/or after the call. This is to
  /// mainly to allow 16 bits float support even-though little or no math
  /// runtime is currently available for it.
  static mlir::Value genRuntimeCall(Context &, MathRuntimeLibrary &);

  /// All generators can be combined with genWrapperCall that will build a
  /// function named "fir."+ <generic name> + "." + <result type code> and
  /// generate the intrinsic implementation inside instead of at the intrinsic
  /// call sites. This can be used to keep the FIR more readable.
  template <Generator g>
  static mlir::Value genWrapperCall(Context &c, MathRuntimeLibrary &r) {
    return outlineInWrapper(g, c, r);
  }

  /// The defaultGenerator is always attempted if no mapping was found for the
  /// generic name provided.
  static mlir::Value defaultGenerator(Context &c, MathRuntimeLibrary &r) {
    return genWrapperCall<&I::genRuntimeCall>(c, r);
  }

  static mlir::Value genConjg(Context &, MathRuntimeLibrary &);
  template <Extremum, ExtremumBehavior>
  static mlir::Value genExtremum(Context &, MathRuntimeLibrary &);
  static mlir::Value genMerge(Context &, MathRuntimeLibrary &);

  struct IntrinsicHanlder {
    const char *name;
    Generator generator{&I::defaultGenerator};
  };
  using I = Implementation;
  /// Table that drives the fir generation depending on the intrinsic.
  /// one to one mapping with Fortran arguments. If no mapping is
  /// defined here for a generic intrinsic, the defaultGenerator will
  /// be attempted.
  static constexpr IntrinsicHanlder handlers[]{
      {"conjg", &I::genConjg},
      {"max", &I::genExtremum<Extremum::Max, ExtremumBehavior::MinMaxss>},
      {"min", &I::genExtremum<Extremum::Min, ExtremumBehavior::MinMaxss>},
      {"merge", &I::genMerge},
  };

  // helpers
  static mlir::Value outlineInWrapper(Generator, Context &c,
                                      MathRuntimeLibrary &r);

  MathRuntimeLibrary runtime;
};

// helpers
static std::string getIntrinsicWrapperName(const llvm::StringRef &intrinsic,
                                           mlir::FunctionType funTy);
static mlir::FunctionType getFunctionType(mlir::Type resultType,
                                          llvm::ArrayRef<mlir::Value> arguments,
                                          Fortran::lower::FirOpBuilder &);

/// Define a simple static runtime description that will be transformed into
/// RuntimeFunction when building the IntrinsicLibrary.
class MathsRuntimeStaticDescription : public RuntimeStaticDescription {
public:
  constexpr MathsRuntimeStaticDescription(const char *n, const char *s,
                                          MaybeTypeCode r, TypeCodeVector a)
      : RuntimeStaticDescription{s, r, a}, name{n} {}
  llvm::StringRef getName() const { return name; }

private:
  // Generic math function name
  const char *name = nullptr;
};

/// Description of the runtime functions available on the target.
using RType = typename RuntimeStaticDescription::TypeCode;
using Args = typename RuntimeStaticDescription::TypeCodeVector;
static constexpr MathsRuntimeStaticDescription llvmRuntime[] = {
    {"abs", "llvm.fabs.f32", RType::f32, Args::create<RType::f32>()},
    {"abs", "llvm.fabs.f64", RType::f64, Args::create<RType::f64>()},
    {"acos", "acosf", RType::f32, Args::create<RType::f32>()},
    {"acos", "acos", RType::f64, Args::create<RType::f64>()},
    {"atan", "atan2f", RType::f32, Args::create<RType::f32, RType::f32>()},
    {"atan", "atan2", RType::f64, Args::create<RType::f64, RType::f64>()},
    {"sqrt", "llvm.sqrt.f32", RType::f32, Args::create<RType::f32>()},
    {"sqrt", "llvm.sqrt.f64", RType::f64, Args::create<RType::f64>()},
    {"cos", "llvm.cos.f32", RType::f32, Args::create<RType::f32>()},
    {"cos", "llvm.cos.f64", RType::f64, Args::create<RType::f64>()},
    {"sin", "llvm.sin.f32", RType::f32, Args::create<RType::f32>()},
    {"sin", "llvm.sin.f64", RType::f64, Args::create<RType::f64>()},
};

static constexpr MathsRuntimeStaticDescription pgmathPreciseRuntime[] = {
    {"acos", "__pc_acos_1", RType::c32, Args::create<RType::c32>()},
    {"acos", "__pz_acos_1", RType::c64, Args::create<RType::c64>()},
    {"pow", "__pc_pow_1", RType::c32, Args::create<RType::c32, RType::c32>()},
    {"pow", "__pc_powi_1", RType::c32, Args::create<RType::c32, RType::i32>()},
    {"pow", "__pc_powk_1", RType::c32, Args::create<RType::c32, RType::i64>()},
    {"pow", "__pd_pow_1", RType::f64, Args::create<RType::f64, RType::f64>()},
    {"pow", "__pd_powi_1", RType::f64, Args::create<RType::f64, RType::i32>()},
    {"pow", "__pd_powk_1", RType::f64, Args::create<RType::f64, RType::i64>()},
    {"pow", "__ps_pow_1", RType::f32, Args::create<RType::f32, RType::f32>()},
    {"pow", "__ps_powi_1", RType::f32, Args::create<RType::f32, RType::i32>()},
    {"pow", "__ps_powk_1", RType::f32, Args::create<RType::f32, RType::i64>()},
    {"pow", "__pz_pow_1", RType::c64, Args::create<RType::c64, RType::c64>()},
    {"pow", "__pz_powi_1", RType::c64, Args::create<RType::c64, RType::i32>()},
    {"pow", "__pz_powk_1", RType::c64, Args::create<RType::c64, RType::i64>()},
    {"pow", "__mth_i_ipowi", RType::i32,
     Args::create<RType::i32, RType::i32>()},
    {"pow", "__mth_i_kpowi", RType::i64,
     Args::create<RType::i64, RType::i32>()},
    {"pow", "__mth_i_kpowk", RType::i64,
     Args::create<RType::i64, RType::i64>()},
};

// TODO : Tables above should be generated in a clever ways and probably shared
// with lib/evaluate intrinsic folding.

// Implementations

// IntrinsicLibrary implementation

IntrinsicLibrary::IntrinsicLibrary(IntrinsicLibrary::Version v,
                                   mlir::MLIRContext &context)
    : impl{new Implementation(v, context)} {}
IntrinsicLibrary::~IntrinsicLibrary() = default;

mlir::Value IntrinsicLibrary::genval(mlir::Location loc,
                                     Fortran::lower::FirOpBuilder &builder,
                                     llvm::StringRef name,
                                     mlir::Type resultType,
                                     llvm::ArrayRef<mlir::Value> args) const {
  assert(impl);
  return impl->genval(loc, builder, name, resultType, args);
}

// MathRuntimeLibrary implementation

// Create the runtime description for the targeted library version.
// So far ignore the version an only load the dummy llvm lib and pgmath precise
MathRuntimeLibrary::MathRuntimeLibrary(IntrinsicLibrary::Version,
                                       mlir::MLIRContext &mlirContext) {
  for (const MathsRuntimeStaticDescription &func : llvmRuntime) {
    RuntimeFunction impl{func.getSymbol(),
                         func.getMLIRFunctionType(&mlirContext)};
    library.insert({Key{func.getName()}, impl});
  }
  for (const MathsRuntimeStaticDescription &func : pgmathPreciseRuntime) {
    RuntimeFunction impl{func.getSymbol(),
                         func.getMLIRFunctionType(&mlirContext)};
    library.insert({Key{func.getName()}, impl});
  }
}

mlir::FuncOp
MathRuntimeLibrary::getFuncOp(Fortran::lower::FirOpBuilder &builder,
                              const RuntimeFunction &runtime) const {
  auto function = builder.addNamedFunction(runtime.symbol, runtime.type);
  function.setAttr("fir.runtime", builder.getUnitAttr());
  return function;
}

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
    auto nInputs = from.getNumInputs();
    auto nResults = from.getNumResults();
    if (nResults != to.getNumResults() || nInputs != to.getNumInputs()) {
      infinite = true;
    } else {
      for (decltype(nInputs) i{0}; i < nInputs; ++i)
        addArgumentDistance(from.getInput(i), to.getInput(i));
      for (decltype(nResults) i{0}; i < nResults; ++i)
        addResultDistance(to.getResult(i), from.getResult(i));
    }
  }
  bool isSmallerThan(const FunctionDistance &d) const {
    return d.infinite ||
           (!infinite && std::lexicographical_compare(
                             conversions.begin(), conversions.end(),
                             d.conversions.begin(), d.conversions.end()));
  }
  bool isLoosingPrecision() const {
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
    if (auto cplx{t.dyn_cast<fir::CplxType>()})
      return cplx.getFKind() * 4;
    assert(false && "not a floating-point type");
    return 0;
  }
  static bool isFloatingPointType(mlir::Type t) {
    return t.isa<mlir::FloatType>() || t.isa<fir::RealType>();
  }
  static Conversion conversionBetweenTypes(mlir::Type from, mlir::Type to) {
    if (from == to) {
      return Conversion::None;
    }
    if (auto fromIntTy{from.dyn_cast<mlir::IntegerType>()}) {
      if (auto toIntTy{to.dyn_cast<mlir::IntegerType>()}) {
        return fromIntTy.getWidth() > toIntTy.getWidth() ? Conversion::Narrow
                                                         : Conversion::Extend;
      }
    }
    if (isFloatingPointType(from) && isFloatingPointType(to)) {
      return getFloatingPointWidth(from) > getFloatingPointWidth(to)
                 ? Conversion::Narrow
                 : Conversion::Extend;
    }
    if (auto fromCplxTy{from.dyn_cast<fir::CplxType>()}) {
      if (auto toCplxTy{to.dyn_cast<fir::CplxType>()}) {
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
  std::array<int, dataSize> conversions{/* zero init*/};
  bool infinite{false}; // When forbidden conversion or wrong argument number
};

// Select runtime function that has the smallest distance to the intrinsic
// function type and that will not imply narrowing arguments or extending the
// result.
llvm::Optional<mlir::FuncOp>
MathRuntimeLibrary::getFunction(Fortran::lower::FirOpBuilder &builder,
                                llvm::StringRef name,
                                mlir::FunctionType funcType) const {
  auto range = library.equal_range(name);
  const RuntimeFunction *bestNearMatch = nullptr;
  FunctionDistance bestMatchDistance{};
  for (auto iter{range.first}; iter != range.second; ++iter) {
    const RuntimeFunction &impl = iter->second;
    if (funcType == impl.type) {
      return getFuncOp(builder, impl); // exact match
    } else {
      FunctionDistance distance(funcType, impl.type);
      if (distance.isSmallerThan(bestMatchDistance)) {
        bestNearMatch = &impl;
        bestMatchDistance = std::move(distance);
      }
    }
  }
  if (bestNearMatch != nullptr) {
    assert(!bestMatchDistance.isLoosingPrecision() &&
           "runtime selection looses precision");
    return getFuncOp(builder, *bestNearMatch);
  }
  return {};
}

// IntrinsicLibrary::Implementation implementation

mlir::Value IntrinsicLibrary::Implementation::genval(
    mlir::Location loc, Fortran::lower::FirOpBuilder &builder,
    llvm::StringRef name, mlir::Type resultType,
    llvm::ArrayRef<mlir::Value> args) {
  Context context{loc, &builder, name, args,
                  getFunctionType(resultType, args, builder)};
  for (auto &handler : handlers) {
    if (name == handler.name) {
      assert(handler.generator != nullptr);
      return handler.generator(context, runtime);
    }
  }
  // Try the default generator if no special handler was defined for the
  // intrinsic being called.
  return defaultGenerator(context, runtime);
}

static mlir::FunctionType
getFunctionType(mlir::Type resultType, llvm::ArrayRef<mlir::Value> arguments,
                Fortran::lower::FirOpBuilder &builder) {
  llvm::SmallVector<mlir::Type, 2> argumentTypes;
  for (auto &arg : arguments) {
    assert(arg != nullptr); // TODO think about optionals
    argumentTypes.push_back(arg.getType());
  }
  return mlir::FunctionType::get(argumentTypes, resultType,
                                 builder.getModule().getContext());
}

// TODO find nicer type to string infra or move this in a mangling utility
// mlir as Type::dump(ostream) methods but it may adds !
static std::string typeToString(mlir::Type t) {
  if (auto i{t.dyn_cast<mlir::IntegerType>()}) {
    return "i" + std::to_string(i.getWidth());
  }
  if (auto cplx{t.dyn_cast<fir::CplxType>()}) {
    return "z" + std::to_string(cplx.getFKind());
  }
  if (auto real{t.dyn_cast<fir::RealType>()}) {
    return "r" + std::to_string(real.getFKind());
  }
  if (auto f{t.dyn_cast<mlir::FloatType>()}) {
    return "f" + std::to_string(f.getWidth());
  }
  if (auto logical{t.dyn_cast<fir::LogicalType>()}) {
    return "l" + std::to_string(logical.getFKind());
  }
  if (auto character{t.dyn_cast<fir::CharacterType>()}) {
    return "c" + std::to_string(character.getFKind());
  }
  assert(false && "no mangling for type");
  return ""s;
}

static std::string getIntrinsicWrapperName(const llvm::StringRef &intrinsic,
                                           mlir::FunctionType funTy) {
  std::string name{"fir." + intrinsic.str() + "."};
  assert(funTy.getNumResults() == 1 && "only function mangling supported");
  name += typeToString(funTy.getResult(0));
  auto e = funTy.getNumInputs();
  for (decltype(e) i = 0; i < e; ++i) {
    name += "." + typeToString(funTy.getInput(i));
  }
  return name;
}

mlir::Value IntrinsicLibrary::Implementation::outlineInWrapper(
    Generator generator, Context &context, MathRuntimeLibrary &runtime) {
  auto *builder = context.builder;
  auto module = builder->getModule();
  auto *mlirContext = module.getContext();
  std::string wrapperName =
      getIntrinsicWrapperName(context.name, context.funcType);
  auto function = builder->getNamedFunction(wrapperName);
  if (!function) {
    // First time this wrapper is needed, build it.
    function = builder->createFunction(wrapperName, context.funcType);
    function.setAttr("fir.intrinsic", builder->getUnitAttr());
    function.addEntryBlock();

    // Create local context to emit code into the newly created function
    // This new function is not linked to a source file location, only
    // its calls will be.
    Context localContext = context;
    auto localBuilder =
        std::make_unique<Fortran::lower::FirOpBuilder>(function);
    localBuilder->setInsertionPointToStart(&function.front());
    localContext.builder = &(*localBuilder);
    llvm::SmallVector<mlir::Value, 2> localArguments;
    for (mlir::BlockArgument bArg : function.front().getArguments())
      localArguments.push_back(bArg);
    localContext.arguments = localArguments;
    localContext.loc = mlir::UnknownLoc::get(mlirContext);

    mlir::Value result = generator(localContext, runtime);
    localBuilder->create<mlir::ReturnOp>(localContext.loc, result);
  } else {
    // Wrapper was already built, ensure it has the sought type
    assert(function.getType() == context.funcType);
  }
  auto call =
      builder->create<mlir::CallOp>(context.loc, function, context.arguments);
  return call.getResult(0);
}

mlir::Value
IntrinsicLibrary::Implementation::genRuntimeCall(Context &context,
                                                 MathRuntimeLibrary &runtime) {
  // Look up runtime
  mlir::FunctionType soughtFuncType = context.funcType;
  if (auto funcOp =
          runtime.getFunction(*context.builder, context.name, soughtFuncType)) {
    mlir::FunctionType actualFuncType = funcOp->getType();
    if (actualFuncType.getNumResults() != soughtFuncType.getNumResults() ||
        actualFuncType.getNumInputs() != soughtFuncType.getNumInputs() ||
        actualFuncType.getNumInputs() != context.arguments.size() ||
        actualFuncType.getNumResults() != 1) {
      assert(false); // TODO better error handling
      return nullptr;
    }
    llvm::SmallVector<mlir::Value, 2> convertedArguments;
    int i = 0;
    for (mlir::Value arg : context.arguments) {
      auto actualType = actualFuncType.getInput(i);
      if (soughtFuncType.getInput(i) != actualType) {
        auto castedArg = context.builder->create<fir::ConvertOp>(
            context.loc, actualType, arg);
        convertedArguments.push_back(castedArg.getResult());
      } else {
        convertedArguments.push_back(arg);
      }
      ++i;
    }
    auto call = context.builder->create<mlir::CallOp>(context.loc, *funcOp,
                                                      convertedArguments);
    mlir::Type soughtType = soughtFuncType.getResult(0);
    mlir::Value res = call.getResult(0);
    if (actualFuncType.getResult(0) != soughtType) {
      auto castedRes =
          context.builder->create<fir::ConvertOp>(context.loc, soughtType, res);
      return castedRes.getResult();
    } else {
      return res;
    }
  } else {
    // could not find runtime function
    assert(false && "no runtime found for this intrinsics");
    // TODO: better error handling ?
    //  - Try to have compile time check of runtime compltness ?
  }
  return {}; // gets rid of warnings
}

// CONJG
mlir::Value IntrinsicLibrary::Implementation::genConjg(Context &genCtxt,
                                                       MathRuntimeLibrary &) {
  assert(genCtxt.arguments.size() == 1);
  mlir::Type resType = genCtxt.getResultType();
  if (resType != genCtxt.arguments[0].getType())
    llvm_unreachable("argument type mismatch");
  Fortran::lower::FirOpBuilder &builder = *genCtxt.builder;
  builder.setLocation(genCtxt.loc);

  mlir::Value cplx = genCtxt.arguments[0];
  auto imag = builder.extractComplexPart(cplx, /*isImagPart=*/true);
  auto negImag = builder.create<fir::NegfOp>(genCtxt.loc, imag);
  return builder.insertComplexPart(cplx, negImag, /*isImagPart=*/true);
}

// MERGE
mlir::Value IntrinsicLibrary::Implementation::genMerge(Context &genCtxt,
                                                       MathRuntimeLibrary &) {
  assert(genCtxt.arguments.size() == 3);
  [[maybe_unused]] auto resType = genCtxt.getResultType();
  Fortran::lower::FirOpBuilder &builder = *genCtxt.builder;

  auto &trueVal = genCtxt.arguments[0];
  auto &falseVal = genCtxt.arguments[1];
  auto &mask = genCtxt.arguments[2];
  auto i1Type = mlir::IntegerType::get(1, builder.getContext());
  auto msk = builder.create<fir::ConvertOp>(genCtxt.loc, i1Type, mask);
  return builder.create<mlir::SelectOp>(genCtxt.loc, msk, trueVal, falseVal);
}

// Compare two FIR values and return boolean result as i1.
template <Extremum extremum, ExtremumBehavior behavior>
static mlir::Value createExtremumCompare(mlir::Location loc,
                                         Fortran::lower::FirOpBuilder &builder,
                                         mlir::Value left, mlir::Value right) {
  static constexpr auto integerPredicate = extremum == Extremum::Max
                                               ? mlir::CmpIPredicate::sgt
                                               : mlir::CmpIPredicate::slt;
  static constexpr auto orderedCmp = extremum == Extremum::Max
                                         ? mlir::CmpFPredicate::OGT
                                         : mlir::CmpFPredicate::OLT;
  auto type = left.getType();
  mlir::Value result;
  if (type.isa<mlir::FloatType>() || type.isa<fir::RealType>()) {
    // Note: the signaling/quit aspect of the result required by IEEE
    // cannot currently be obtained with LLVM without ad-hoc runtime.
    if constexpr (behavior == ExtremumBehavior::IeeeMinMaximumNumber) {
      // Return the number if one of the inputs is NaN and the other is
      // a number.
      auto leftIsResult =
          builder.create<fir::CmpfOp>(loc, orderedCmp, left, right);
      auto rightIsNan = builder.create<fir::CmpfOp>(
          loc, mlir::CmpFPredicate::UNE, right, right);
      result = builder.create<mlir::OrOp>(loc, leftIsResult, rightIsNan);
    } else if constexpr (behavior == ExtremumBehavior::IeeeMinMaximum) {
      // Always return NaNs if one the input is NaNs
      auto leftIsResult =
          builder.create<fir::CmpfOp>(loc, orderedCmp, left, right);
      auto leftIsNan = builder.create<fir::CmpfOp>(
          loc, mlir::CmpFPredicate::UNE, left, left);
      result = builder.create<mlir::OrOp>(loc, leftIsResult, leftIsNan);
    } else if constexpr (behavior == ExtremumBehavior::MinMaxss) {
      // If the left is a NaN, return the right whatever it is.
      result = builder.create<fir::CmpfOp>(loc, orderedCmp, left, right);
    } else if constexpr (behavior == ExtremumBehavior::PgfortranLlvm) {
      // If one of the operand is a NaN, return left whatever it is.
      static constexpr auto unorderedCmp = extremum == Extremum::Max
                                               ? mlir::CmpFPredicate::UGT
                                               : mlir::CmpFPredicate::ULT;
      result = builder.create<fir::CmpfOp>(loc, unorderedCmp, left, right);
    } else {
      // TODO: ieeMinNum/ieeeMaxNum
      static_assert(behavior == ExtremumBehavior::IeeeMinMaxNum,
                    "ieeeMinNum/ieeMaxNum behavior not implemented");
    }
  } else if (type.isa<mlir::IntegerType>()) {
    result = builder.create<mlir::CmpIOp>(loc, integerPredicate, left, right);
  } else if (type.isa<fir::CharacterType>()) {
    // TODO: ! character min and max is tricky because the result
    // length is the length of the longest argument!
    // So we may need a temp.
  }
  assert(result);
  return result;
}

// MIN and MAX
template <Extremum extremum, ExtremumBehavior behavior>
mlir::Value
IntrinsicLibrary::Implementation::genExtremum(Context &genCtxt,
                                              MathRuntimeLibrary &) {
  auto &builder = *genCtxt.builder;
  auto loc = genCtxt.loc;
  assert(genCtxt.arguments.size() >= 2);
  mlir::Value result = genCtxt.arguments[0];
  for (auto arg : genCtxt.arguments.drop_front()) {
    auto mask =
        createExtremumCompare<extremum, behavior>(loc, builder, result, arg);
    result = builder.create<mlir::SelectOp>(loc, mask, result, arg);
  }
  return result;
}

} // namespace Fortran::lower
