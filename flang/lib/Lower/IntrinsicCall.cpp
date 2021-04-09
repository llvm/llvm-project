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
#include "RTBuilder.h"
#include "StatementContext.h"
#include "SymbolMap.h"
#include "flang/Common/static-multimap-view.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/CharacterRuntime.h"
#include "flang/Lower/ComplexExpr.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/ReductionRuntime.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <string_view>
#include <utility>

#define DEBUG_TYPE "flang-lower-intrinsic"

#define PGMATH_DECLARE
#include "../runtime/pgmath.h.inc"

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

fir::ExtendedValue Fortran::lower::getAbsentIntrinsicArgument() {
  return fir::UnboxedValue{};
}

/// Helper to test if an intrinsic argument is absent.
static bool isAbsent(const fir::ExtendedValue &exv) {
  return !fir::getBase(exv);
}

// TODO error handling -> return a code or directly emit messages ?
struct IntrinsicLibrary {

  // Constructors.
  explicit IntrinsicLibrary(Fortran::lower::FirOpBuilder &builder,
                            mlir::Location loc,
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

  using RuntimeCallGenerator =
      std::function<mlir::Value(Fortran::lower::FirOpBuilder &, mlir::Location,
                                llvm::ArrayRef<mlir::Value>)>;
  RuntimeCallGenerator
  getRuntimeCallGenerator(llvm::StringRef name,
                          mlir::FunctionType soughtFuncType);

  mlir::Value genAbs(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genAimag(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genAint(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genAll(mlir::Type,
                            llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genAllocated(mlir::Type,
                                  llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genAnint(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genAny(mlir::Type,
                            llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genAssociated(mlir::Type,
                                   llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genCeiling(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genChar(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genConjg(mlir::Type, llvm::ArrayRef<mlir::Value>);
  void genDateAndTime(llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genDim(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genDprod(mlir::Type, llvm::ArrayRef<mlir::Value>);
  template <Extremum, ExtremumBehavior>
  mlir::Value genExtremum(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genFloor(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIAnd(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genIchar(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genIEOr(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genIndex(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genIOr(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genLen(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genLenTrim(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genMerge(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genMod(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genModulo(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genNint(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genPresent(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genScan(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genSign(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genTrim(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genVerify(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  /// Implement all conversion functions like DBLE, the first argument is
  /// the value to convert. There may be an additional KIND arguments that
  /// is ignored because this is already reflected in the result type.
  mlir::Value genConversion(mlir::Type, llvm::ArrayRef<mlir::Value>);

  /// Define the different FIR generators that can be mapped to intrinsic to
  /// generate the related code.
  using ElementalGenerator = decltype(&IntrinsicLibrary::genAbs);
  using ExtendedGenerator = decltype(&IntrinsicLibrary::genLenTrim);
  using SubroutineGenerator = decltype(&IntrinsicLibrary::genDateAndTime);
  using Generator =
      std::variant<ElementalGenerator, ExtendedGenerator, SubroutineGenerator>;

  /// All generators can be outlined. This will build a function named
  /// "fir."+ <generic name> + "." + <result type code> and generate the
  /// intrinsic implementation inside instead of at the intrinsic call sites.
  /// This can be used to keep the FIR more readable. Only one function will
  /// be generated for all the similar calls in a program.
  /// If the Generator is nullptr, the wrapper uses genRuntimeCall.
  template <typename GeneratorType>
  mlir::Value outlineInWrapper(GeneratorType, llvm::StringRef name,
                               mlir::Type resultType,
                               llvm::ArrayRef<mlir::Value> args);
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
  mlir::Value invokeGenerator(SubroutineGenerator generator,
                              llvm::ArrayRef<mlir::Value> args);

  /// Get pointer to unrestricted intrinsic. Generate the related unrestricted
  /// intrinsic if it is not defined yet.
  mlir::SymbolRefAttr
  getUnrestrictedIntrinsicSymbolRefAttr(llvm::StringRef name,
                                        mlir::FunctionType signature);

  /// Add clean-up for \p temp to the current statement context;
  void addCleanUpForTemp(mlir::Location loc, mlir::Value temp);

  Fortran::lower::FirOpBuilder &builder;
  mlir::Location loc;
  Fortran::lower::StatementContext *stmtCtx;
};

struct IntrinsicDummyArgument {
  const char *name = nullptr;
  Fortran::lower::LowerIntrinsicArgAs lowerAs =
      Fortran::lower::LowerIntrinsicArgAs::Value;
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
  bool isElemental = true;
  /// Code heavy intrinsic can be outlined to make FIR
  /// more readable.
  bool outline = false;
};

constexpr auto asInquired = Fortran::lower::LowerIntrinsicArgAs::Inquired;
constexpr auto asAddr = Fortran::lower::LowerIntrinsicArgAs::Addr;
constexpr auto asValue = Fortran::lower::LowerIntrinsicArgAs::Value;
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
    {"achar", &I::genChar},
    {"aimag", &I::genAimag},
    {"aint", &I::genAint},
    {"all", 
     &I::genAll,
     {{{"mask", asAddr},
       {"dim", asValue}}},
     /*isElemental=*/false},
    {"allocated",
     &I::genAllocated,
     {{{"array", asInquired}, {"scalar", asInquired}}},
     /*isElemental=*/false},
    {"anint", &I::genAnint},
    {"any", 
     &I::genAny,
     {{{"mask", asAddr},
       {"dim", asValue}}},
     /*isElemental=*/false},
    {"associated",
     &I::genAssociated,
     {{{"pointer", asInquired}, {"target", asInquired}}},
     /*isElemental=*/false},
    {"ceiling", &I::genCeiling},
    {"char", &I::genChar},
    {"conjg", &I::genConjg},
    {"date_and_time",
     &I::genDateAndTime,
     {{{"date", asAddr},
       {"time", asAddr},
       {"zone", asAddr},
       {"values", asAddr}}},
     /*isElemental=*/false},
    {"dim", &I::genDim},
    {"dble", &I::genConversion},
    {"dprod", &I::genDprod},
    {"floor", &I::genFloor},
    {"iachar", &I::genIchar},
    {"iand", &I::genIAnd},
    {"ichar", &I::genIchar},
    {"ieor", &I::genIEOr},
    {"index", &I::genIndex},
    {"ior", &I::genIOr},
    {"len", &I::genLen},
    {"len_trim", &I::genLenTrim},
    {"max", &I::genExtremum<Extremum::Max, ExtremumBehavior::MinMaxss>},
    {"min", &I::genExtremum<Extremum::Min, ExtremumBehavior::MinMaxss>},
    {"merge", &I::genMerge},
    {"mod", &I::genMod},
    {"modulo", &I::genModulo},
    {"nint", &I::genNint},
    {"present", &I::genPresent, {{{"a", asInquired}}}, /*isElemental=*/false},
    {"scan",
     &I::genScan,
     {{{"string", asAddr},
       {"set", asAddr},
       {"back", asValue},
       {"kind", asValue}}},
     /*isElemental=*/true},
    {"sign", &I::genSign},
    {"trim", &I::genTrim, {{{"string", asAddr}}}, /*isElemental=*/false},
    {"verify",
     &I::genVerify,
     {{{"string", asAddr},
       {"set", asAddr},
       {"back", asValue},
       {"kind", asValue}}},
     /*isElemental=*/true},
};

/// To make fir output more readable for debug, one can outline all intrinsic
/// implementation in wrappers (overrides the IntrinsicHandler::outline flag).
static llvm::cl::opt<bool> outlineAllIntrinsics(
    "outline-intrinsics",
    llvm::cl::desc(
        "Lower all intrinsic procedure implementation in their own functions"),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// Math runtime description and matching utility
//===----------------------------------------------------------------------===//

/// Command line option to modify math runtime version used to implement
/// intrinsics.
enum MathRuntimeVersion {
  fastVersion,
  relaxedVersion,
  preciseVersion,
  llvmOnly
};
llvm::cl::opt<MathRuntimeVersion> mathRuntimeVersion(
    "math-runtime", llvm::cl::desc("Select math runtime version:"),
    llvm::cl::values(
        clEnumValN(fastVersion, "fast", "use pgmath fast runtime"),
        clEnumValN(relaxedVersion, "relaxed", "use pgmath relaxed runtime"),
        clEnumValN(preciseVersion, "precise", "use pgmath precise runtime"),
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
  Fortran::lower::FuncTypeBuilderFunc typeGenerator;
};

#define RUNTIME_STATIC_DESCRIPTION(name, func)                                 \
  {#name, #func,                                                               \
   Fortran::lower::RuntimeTableKey<decltype(func)>::getTypeModel()},
static constexpr RuntimeFunction pgmathFast[] = {
#define PGMATH_FAST
#define PGMATH_USE_ALL_TYPES(name, func) RUNTIME_STATIC_DESCRIPTION(name, func)
#include "../runtime/pgmath.h.inc"
};
static constexpr RuntimeFunction pgmathRelaxed[] = {
#define PGMATH_RELAXED
#define PGMATH_USE_ALL_TYPES(name, func) RUNTIME_STATIC_DESCRIPTION(name, func)
#include "../runtime/pgmath.h.inc"
};
static constexpr RuntimeFunction pgmathPrecise[] = {
#define PGMATH_PRECISE
#define PGMATH_USE_ALL_TYPES(name, func) RUNTIME_STATIC_DESCRIPTION(name, func)
#include "../runtime/pgmath.h.inc"
};

static mlir::FunctionType genF32F32FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF32(context);
  return mlir::FunctionType::get(context, {t}, {t});
}

static mlir::FunctionType genF64F64FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF64(context);
  return mlir::FunctionType::get(context, {t}, {t});
}

template <int Bits>
static mlir::FunctionType genIntF64FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF64(context);
  auto r = mlir::IntegerType::get(context, Bits);
  return mlir::FunctionType::get(context, {t}, {r});
}

template <int Bits>
static mlir::FunctionType genIntF32FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF32(context);
  auto r = mlir::IntegerType::get(context, Bits);
  return mlir::FunctionType::get(context, {t}, {r});
}

// TODO : Fill-up this table with more intrinsic.
// Note: These are also defined as operations in LLVM dialect. See if this
// can be use and has advantages.
static constexpr RuntimeFunction llvmIntrinsics[] = {
    {"abs", "llvm.fabs.f32", genF32F32FuncType},
    {"abs", "llvm.fabs.f64", genF64F64FuncType},
    {"aint", "llvm.trunc.f32", genF32F32FuncType},
    {"aint", "llvm.trunc.f64", genF64F64FuncType},
    {"anint", "llvm.round.f32", genF32F32FuncType},
    {"anint", "llvm.round.f64", genF64F64FuncType},
    // ceil is used for CEILING but is different, it returns a real.
    {"ceil", "llvm.ceil.f32", genF32F32FuncType},
    {"ceil", "llvm.ceil.f64", genF64F64FuncType},
    {"cos", "llvm.cos.f32", genF32F32FuncType},
    {"cos", "llvm.cos.f64", genF64F64FuncType},
    // llvm.floor is used for FLOOR, but returns real.
    {"floor", "llvm.floor.f32", genF32F32FuncType},
    {"floor", "llvm.floor.f64", genF64F64FuncType},
    {"log", "llvm.log.f32", genF32F32FuncType},
    {"log", "llvm.log.f64", genF64F64FuncType},
    {"log10", "llvm.log10.f32", genF32F32FuncType},
    {"log10", "llvm.log10.f64", genF64F64FuncType},
    {"nint", "llvm.lround.i64.f64", genIntF64FuncType<64>},
    {"nint", "llvm.lround.i64.f32", genIntF32FuncType<64>},
    {"nint", "llvm.lround.i32.f64", genIntF64FuncType<32>},
    {"nint", "llvm.lround.i32.f32", genIntF32FuncType<32>},
    {"sin", "llvm.sin.f32", genF32F32FuncType},
    {"sin", "llvm.sin.f64", genF64F64FuncType},
    {"sqrt", "llvm.sqrt.f32", genF32F32FuncType},
    {"sqrt", "llvm.sqrt.f64", genF64F64FuncType},
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
    auto nInputs = from.getNumInputs();
    auto nResults = from.getNumResults();
    if (nResults != to.getNumResults() || nInputs != to.getNumInputs()) {
      infinite = true;
    } else {
      for (decltype(nInputs) i{0}; i < nInputs && !infinite; ++i)
        addArgumentDistance(from.getInput(i), to.getInput(i));
      for (decltype(nResults) i{0}; i < nResults && !infinite; ++i)
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
    if (from == to) {
      return Conversion::None;
    }
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

  std::array<int, dataSize> conversions{/* zero init*/};
  bool infinite{false}; // When forbidden conversion or wrong argument number
};

/// Build mlir::FuncOp from runtime symbol description and add
/// fir.runtime attribute.
static mlir::FuncOp getFuncOp(mlir::Location loc,
                              Fortran::lower::FirOpBuilder &builder,
                              const RuntimeFunction &runtime) {
  auto function = builder.addNamedFunction(
      loc, runtime.symbol, runtime.typeGenerator(builder.getContext()));
  function->setAttr("fir.runtime", builder.getUnitAttr());
  return function;
}

/// Select runtime function that has the smallest distance to the intrinsic
/// function type and that will not imply narrowing arguments or extending the
/// result.
/// If nothing is found, the mlir::FuncOp will contain a nullptr.
mlir::FuncOp searchFunctionInLibrary(
    mlir::Location loc, Fortran::lower::FirOpBuilder &builder,
    const Fortran::common::StaticMultimapView<RuntimeFunction> &lib,
    llvm::StringRef name, mlir::FunctionType funcType,
    const RuntimeFunction **bestNearMatch,
    FunctionDistance &bestMatchDistance) {
  auto range = lib.equal_range(name);
  for (auto iter{range.first}; iter != range.second && iter; ++iter) {
    const auto &impl = *iter;
    auto implType = impl.typeGenerator(builder.getContext());
    if (funcType == implType) {
      return getFuncOp(loc, builder, impl); // exact match
    } else {
      FunctionDistance distance(funcType, implType);
      if (distance.isSmallerThan(bestMatchDistance)) {
        *bestNearMatch = &impl;
        bestMatchDistance = std::move(distance);
      }
    }
  }
  return {};
}

/// Search runtime for the best runtime function given an intrinsic name
/// and interface. The interface may not be a perfect match in which case
/// the caller is responsible to insert argument and return value conversions.
/// If nothing is found, the mlir::FuncOp will contain a nullptr.
static mlir::FuncOp getRuntimeFunction(mlir::Location loc,
                                       Fortran::lower::FirOpBuilder &builder,
                                       llvm::StringRef name,
                                       mlir::FunctionType funcType) {
  const RuntimeFunction *bestNearMatch = nullptr;
  FunctionDistance bestMatchDistance{};
  mlir::FuncOp match;
  using RtMap = Fortran::common::StaticMultimapView<RuntimeFunction>;
  static constexpr RtMap pgmathF(pgmathFast);
  static_assert(pgmathF.Verify() && "map must be sorted");
  static constexpr RtMap pgmathR(pgmathRelaxed);
  static_assert(pgmathR.Verify() && "map must be sorted");
  static constexpr RtMap pgmathP(pgmathPrecise);
  static_assert(pgmathP.Verify() && "map must be sorted");
  if (mathRuntimeVersion == fastVersion) {
    match = searchFunctionInLibrary(loc, builder, pgmathF, name, funcType,
                                    &bestNearMatch, bestMatchDistance);
  } else if (mathRuntimeVersion == relaxedVersion) {
    match = searchFunctionInLibrary(loc, builder, pgmathR, name, funcType,
                                    &bestNearMatch, bestMatchDistance);
  } else if (mathRuntimeVersion == preciseVersion) {
    match = searchFunctionInLibrary(loc, builder, pgmathP, name, funcType,
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
  if (auto exactMatch =
          searchFunctionInLibrary(loc, builder, llvmIntr, name, funcType,
                                  &bestNearMatch, bestMatchDistance))
    return exactMatch;

  if (bestNearMatch != nullptr) {
    assert(!bestMatchDistance.isLosingPrecision() &&
           "runtime selection loses precision");
    return getFuncOp(loc, builder, *bestNearMatch);
  }
  return {};
}

/// Helpers to get function type from arguments and result type.
static mlir::FunctionType
getFunctionType(llvm::Optional<mlir::Type> resultType,
                llvm::ArrayRef<mlir::Value> arguments,
                Fortran::lower::FirOpBuilder &builder) {
  llvm::SmallVector<mlir::Type> argTypes;
  for (auto &arg : arguments)
    argTypes.push_back(arg.getType());
  llvm::SmallVector<mlir::Type> resTypes;
  if (resultType)
    resTypes.push_back(*resultType);
  return mlir::FunctionType::get(builder.getModule().getContext(), argTypes,
                                 resTypes);
}

/// fir::ExtendedValue to mlir::Value translation layer

fir::ExtendedValue toExtendedValue(mlir::Value val,
                                   Fortran::lower::FirOpBuilder &builder,
                                   mlir::Location loc) {
  assert(val && "optional unhandled here");
  auto type = val.getType();
  auto base = val;
  auto indexType = builder.getIndexType();
  llvm::SmallVector<mlir::Value> extents;

  Fortran::lower::CharacterExprHelper charHelper{builder, loc};
  // FIXME: we may want to allow non character scalar here.
  if (charHelper.isCharacterScalar(type))
    return charHelper.toExtendedValue(val);

  if (auto refType = type.dyn_cast<fir::ReferenceType>())
    type = refType.getEleTy();

  if (auto arrayType = type.dyn_cast<fir::SequenceType>()) {
    type = arrayType.getEleTy();
    for (auto extent : arrayType.getShape()) {
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
    mlir::emitError(loc, "descriptor or derived type not yet handled");
  }

  if (!extents.empty())
    return fir::ArrayBoxValue{base, extents};
  return base;
}

mlir::Value toValue(const fir::ExtendedValue &val,
                    Fortran::lower::FirOpBuilder &builder, mlir::Location loc) {
  if (auto charBox = val.getCharBox()) {
    auto buffer = charBox->getBuffer();
    if (buffer.getType().isa<fir::BoxCharType>())
      return buffer;
    return Fortran::lower::CharacterExprHelper{builder, loc}.createEmboxChar(
        buffer, charBox->getLen());
  }

  // FIXME: need to access other ExtendedValue variants and handle them
  // properly.
  return fir::getBase(val);
}

//===----------------------------------------------------------------------===//
// IntrinsicLibrary
//===----------------------------------------------------------------------===//

template <typename GeneratorType>
fir::ExtendedValue IntrinsicLibrary::genElementalCall(
    GeneratorType generator, llvm::StringRef name, mlir::Type resultType,
    llvm::ArrayRef<fir::ExtendedValue> args, bool outline) {
  llvm::SmallVector<mlir::Value> scalarArgs;
  for (const auto &arg : args) {
    if (arg.getUnboxed() || arg.getCharBox()) {
      scalarArgs.emplace_back(fir::getBase(arg));
    } else {
      // TODO: get the result shape and create the loop...
      mlir::emitError(loc, "array or descriptor not yet handled in elemental "
                           "intrinsic lowering");
      exit(1);
    }
  }
  if (outline)
    return outlineInWrapper(generator, name, resultType, scalarArgs);
  return invokeGenerator(generator, resultType, scalarArgs);
}

/// Some ExtendedGenerator operating on characters are also elemental
/// (e.g LEN_TRIM).
template <>
fir::ExtendedValue
IntrinsicLibrary::genElementalCall<IntrinsicLibrary::ExtendedGenerator>(
    ExtendedGenerator generator, llvm::StringRef name, mlir::Type resultType,
    llvm::ArrayRef<fir::ExtendedValue> args, bool outline) {
  for (const auto &arg : args)
    if (!arg.getUnboxed() && !arg.getCharBox()) {
      // TODO: get the result shape and create the loop...
      mlir::emitError(loc, "array or descriptor not yet handled in elemental "
                           "intrinsic lowering");
      exit(1);
    }
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
static fir::ExtendedValue
invokeHandler(IntrinsicLibrary::SubroutineGenerator generator,
              const IntrinsicHandler &handler,
              llvm::Optional<mlir::Type> resultType,
              llvm::ArrayRef<fir::ExtendedValue> args, bool outline,
              IntrinsicLibrary &lib) {
  if (outline)
    return lib.outlineInExtendedWrapper(generator, handler.name, resultType,
                                        args);
  std::invoke(generator, lib, args);
  return mlir::Value{};
}

/// Many intrinsics are not yet lowered, provide a clear error message to user
/// instead of hitting harder to understand asserts.
static void crashOnMissingIntrinsic(mlir::Location loc, llvm::StringRef name) {
  mlir::emitError(loc,
                  "TODO: missing intrinsic lowering: " + llvm::Twine(name));
  exit(1);
}

fir::ExtendedValue
IntrinsicLibrary::genIntrinsicCall(llvm::StringRef name,
                                   llvm::Optional<mlir::Type> resultType,
                                   llvm::ArrayRef<fir::ExtendedValue> args) {
  for (const auto &handler : handlers)
    if (name == handler.name) {
      bool outline = handler.outline || outlineAllIntrinsics;
      return std::visit(
          [&](auto &generator) -> fir::ExtendedValue {
            return invokeHandler(generator, handler, resultType, args, outline,
                                 *this);
          },
          handler.generator);
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
  for (const auto &extendedVal : args) {
    auto val = toValue(extendedVal, builder, loc);
    if (!val)
      // If an absent optional gets there, most likely its handler has just
      // not yet been defined.
      crashOnMissingIntrinsic(loc, name);
    mlirArgs.emplace_back(val);
  }
  mlir::FunctionType soughtFuncType =
      getFunctionType(*resultType, mlirArgs, builder);

  auto runtimeCallGenerator = getRuntimeCallGenerator(name, soughtFuncType);
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
  for (auto arg : args)
    extendedArgs.emplace_back(toExtendedValue(arg, builder, loc));
  auto extendedResult = std::invoke(generator, *this, resultType, extendedArgs);
  return toValue(extendedResult, builder, loc);
}

mlir::Value
IntrinsicLibrary::invokeGenerator(SubroutineGenerator generator,
                                  llvm::ArrayRef<mlir::Value> args) {
  llvm::SmallVector<fir::ExtendedValue> extendedArgs;
  for (auto arg : args)
    extendedArgs.emplace_back(toExtendedValue(arg, builder, loc));
  std::invoke(generator, *this, extendedArgs);
  return mlir::Value{};
}

template <typename GeneratorType>
mlir::FuncOp IntrinsicLibrary::getWrapper(GeneratorType generator,
                                          llvm::StringRef name,
                                          mlir::FunctionType funcType,
                                          bool loadRefArguments) {
  std::string wrapperName = fir::mangleIntrinsicProcedure(name, funcType);
  auto function = builder.getNamedFunction(wrapperName);
  if (!function) {
    // First time this wrapper is needed, build it.
    function = builder.createFunction(loc, wrapperName, funcType);
    function->setAttr("fir.intrinsic", builder.getUnitAttr());
    function->setAttr("linkName", builder.createInternalLinkage());
    function.addEntryBlock();

    // Create local context to emit code into the newly created function
    // This new function is not linked to a source file location, only
    // its calls will be.
    auto localBuilder = std::make_unique<Fortran::lower::FirOpBuilder>(
        function, builder.getKindMap());
    localBuilder->setInsertionPointToStart(&function.front());
    // Location of code inside wrapper of the wrapper is independent from
    // the location of the intrinsic call.
    auto localLoc = localBuilder->getUnknownLoc();
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

    if constexpr (std::is_same_v<GeneratorType, SubroutineGenerator>) {
      localLib.invokeGenerator(generator, localArguments);
      localBuilder->create<mlir::ReturnOp>(localLoc);
    } else {
      assert(funcType.getNumResults() == 1 &&
             "expect one result for intrinsic function wrapper type");
      auto resultType = funcType.getResult(0);
      auto result =
          localLib.invokeGenerator(generator, resultType, localArguments);
      localBuilder->create<mlir::ReturnOp>(localLoc, result);
    }
  } else {
    // Wrapper was already built, ensure it has the sought type
    assert(function.getType() == funcType &&
           "conflict between intrinsic wrapper types");
  }
  return function;
}

/// Helpers to detect absent optional (not yet supported in outlining).
bool static hasAbsentOptional(llvm::ArrayRef<mlir::Value> args) {
  for (const auto &arg : args)
    if (!arg)
      return true;
  return false;
}
bool static hasAbsentOptional(llvm::ArrayRef<fir::ExtendedValue> args) {
  for (const auto &arg : args)
    if (!fir::getBase(arg))
      return true;
  return false;
}

template <typename GeneratorType>
mlir::Value
IntrinsicLibrary::outlineInWrapper(GeneratorType generator,
                                   llvm::StringRef name, mlir::Type resultType,
                                   llvm::ArrayRef<mlir::Value> args) {
  if (hasAbsentOptional(args)) {
    // TODO: absent optional in outlining is an issue: we cannot just ignore
    // them. Needs a better interface here. The issue is that we cannot easily
    // tell that a value is optional or not here if it is presents. And if it is
    // absent, we cannot tell what it type should be.
    mlir::emitError(loc, "todo: cannot outline call to intrinsic " +
                             llvm::Twine(name) +
                             " with absent optional argument");
    exit(1);
  }

  auto funcType = getFunctionType(resultType, args, builder);
  auto wrapper = getWrapper(generator, name, funcType);
  return builder.create<fir::CallOp>(loc, wrapper, args).getResult(0);
}

template <typename GeneratorType>
fir::ExtendedValue IntrinsicLibrary::outlineInExtendedWrapper(
    GeneratorType generator, llvm::StringRef name,
    llvm::Optional<mlir::Type> resultType,
    llvm::ArrayRef<fir::ExtendedValue> args) {
  if (hasAbsentOptional(args)) {
    // TODO
    mlir::emitError(loc, "todo: cannot outline call to intrinsic " +
                             llvm::Twine(name) +
                             " with absent optional argument");
    exit(1);
  }
  llvm::SmallVector<mlir::Value> mlirArgs;
  for (const auto &extendedVal : args)
    mlirArgs.emplace_back(toValue(extendedVal, builder, loc));
  auto funcType = getFunctionType(resultType, mlirArgs, builder);
  auto wrapper = getWrapper(generator, name, funcType);
  auto call = builder.create<fir::CallOp>(loc, wrapper, mlirArgs);
  if (resultType)
    return toExtendedValue(call.getResult(0), builder, loc);
  // Subroutine calls
  return mlir::Value{};
}

IntrinsicLibrary::RuntimeCallGenerator
IntrinsicLibrary::getRuntimeCallGenerator(llvm::StringRef name,
                                          mlir::FunctionType soughtFuncType) {
  auto funcOp = getRuntimeFunction(loc, builder, name, soughtFuncType);
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

  return [funcOp, actualFuncType, soughtFuncType](
             Fortran::lower::FirOpBuilder &builder, mlir::Location loc,
             llvm::ArrayRef<mlir::Value> args) {
    llvm::SmallVector<mlir::Value> convertedArguments;
    for (auto [fst, snd] : llvm::zip(actualFuncType.getInputs(), args))
      convertedArguments.push_back(builder.createConvert(loc, fst, snd));
    auto call = builder.create<fir::CallOp>(loc, funcOp, convertedArguments);
    mlir::Type soughtType = soughtFuncType.getResult(0);
    return builder.createConvert(loc, soughtType, call.getResult(0));
  };
}

mlir::SymbolRefAttr IntrinsicLibrary::getUnrestrictedIntrinsicSymbolRefAttr(
    llvm::StringRef name, mlir::FunctionType signature) {
  // Unrestricted intrinsics signature follows implicit rules: argument
  // are passed by references. But the runtime versions expect values.
  // So instead of duplicating the runtime, just have the wrappers loading
  // this before calling the code generators.
  bool loadRefArguments = true;
  mlir::FuncOp funcOp;
  for (const auto &handler : handlers)
    if (name == handler.name)
      funcOp = std::visit(
          [&](auto generator) {
            return getWrapper(generator, name, signature, loadRefArguments);
          },
          handler.generator);

  if (!funcOp) {
    llvm::SmallVector<mlir::Type> argTypes;
    for (auto type : signature.getInputs()) {
      if (auto refType = type.dyn_cast<fir::ReferenceType>())
        argTypes.push_back(refType.getEleTy());
      else
        argTypes.push_back(type);
    }
    auto soughtFuncType =
        builder.getFunctionType(signature.getResults(), argTypes);
    auto rtCallGenerator = getRuntimeCallGenerator(name, soughtFuncType);
    funcOp = getWrapper(rtCallGenerator, name, signature, loadRefArguments);
  }

  return builder.getSymbolRefAttr(funcOp.getName());
}

void IntrinsicLibrary::addCleanUpForTemp(mlir::Location loc, mlir::Value temp) {
  assert(stmtCtx);
  auto *bldr = &builder;
  stmtCtx->attachCleanup([=]() { bldr->create<fir::FreeMemOp>(loc, temp); });
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

mlir::Value IntrinsicLibrary::genConversion(mlir::Type resultType,
                                            llvm::ArrayRef<mlir::Value> args) {
  // There can be an optional kind in second argument.
  assert(args.size() >= 1);
  return builder.convertWithSemantics(loc, resultType, args[0]);
}

// ABS
mlir::Value IntrinsicLibrary::genAbs(mlir::Type resultType,
                                     llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);
  auto arg = args[0];
  auto type = arg.getType();
  if (fir::isa_real(type)) {
    // Runtime call to fp abs. An alternative would be to use mlir AbsFOp
    // but it does not support all fir floating point types.
    return genRuntimeCall("abs", resultType, args);
  }
  if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    // At the time of this implementation there is no abs op in mlir.
    // So, implement abs here without branching.
    auto shift =
        builder.createIntegerConstant(loc, intType, intType.getWidth() - 1);
    auto mask = builder.create<mlir::SignedShiftRightOp>(loc, arg, shift);
    auto xored = builder.create<mlir::XOrOp>(loc, arg, mask);
    return builder.create<mlir::SubIOp>(loc, xored, mask);
  }
  if (fir::isa_complex(type)) {
    // Use HYPOT to fulfill the no underflow/overflow requirement.
    auto parts =
        Fortran::lower::ComplexExprHelper{builder, loc}.extractParts(arg);
    llvm::SmallVector<mlir::Value> args = {parts.first, parts.second};
    return genRuntimeCall("hypot", resultType, args);
  }
  llvm_unreachable("unexpected type in ABS argument");
}

// AIMAG
mlir::Value IntrinsicLibrary::genAimag(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);
  return Fortran::lower::ComplexExprHelper{builder, loc}.extractComplexPart(
      args[0], true /* isImagPart */);
}

// ALL
fir::ExtendedValue
IntrinsicLibrary::genAll(mlir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 2);
  // Handle required mask argument
  auto mask = builder.createBox(loc, args[0]);

  fir::BoxValue maskArry = builder.createBox(loc, args[0]);
  int rank = maskArry.rank();
  assert(rank >= 1);

  // Handle optional dim argument
  bool absentDim = isAbsent(args[1]);
  auto dim = absentDim
             ? builder.createIntegerConstant(loc, builder.getIndexType(), 1)
             : fir::getBase(args[1]);

   if (rank == 1 || absentDim)
     return builder.createConvert(
        loc, resultType,
        Fortran::lower::genAll(builder, loc, mask, dim));

  // else use the result descriptor AllDim() intrinsic

  // Create mutable fir.box to be passed to the runtime for the result.

  auto resultArrayType = builder.getVarLenSeqTy(resultType, rank - 1);
  auto resultMutableBox =
       Fortran::lower::createTempMutableBox(builder, loc, resultArrayType);
  auto resultIrBox =
       Fortran::lower::getMutableIRBox(builder, loc, resultMutableBox);

  // Call runtime. The runtime is allocating the result.
  Fortran::lower::genAllDescriptor(builder, loc, resultIrBox, mask, dim);
  return Fortran::lower::genMutableBoxRead(builder, loc, resultMutableBox)
      .match(
          [&](const fir::ArrayBoxValue &box) -> fir::ExtendedValue {
            addCleanUpForTemp(loc, box.getAddr());
            auto ldVal = builder.create<fir::LoadOp>(loc, box.getAddr());
            return builder.createConvert(loc, resultArrayType, ldVal);
          },
          [&](const auto &) -> fir::ExtendedValue {
            fir::emitFatalError(loc, "Invalid result for ALL");
          });
}


// ALLOCATED
fir::ExtendedValue
IntrinsicLibrary::genAllocated(mlir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  return args[0].match(
      [&](const fir::MutableBoxValue &x) -> fir::ExtendedValue {
        return Fortran::lower::genIsAllocatedOrAssociatedTest(builder, loc, x);
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc,
                            "allocated arg not lowered to MutableBoxValue");
      });
}

// ANINT
mlir::Value IntrinsicLibrary::genAnint(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() >= 1);
  // Skip optional kind argument to search the runtime; it is already reflected
  // in result type.
  return genRuntimeCall("anint", resultType, {args[0]});
}

// ANY
fir::ExtendedValue
IntrinsicLibrary::genAny(mlir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 2);
  // Handle required mask argument
  auto mask = builder.createBox(loc, args[0]);

  fir::BoxValue maskArry = builder.createBox(loc, args[0]);
  int rank = maskArry.rank();
  assert(rank >= 1);

  // Handle optional dim argument
  bool absentDim = isAbsent(args[1]);
  auto dim = absentDim
             ? builder.createIntegerConstant(loc, builder.getIndexType(), 1)
             : fir::getBase(args[1]);

   if (rank == 1 || absentDim)
     return builder.createConvert(
        loc, resultType,
        Fortran::lower::genAny(builder, loc, mask, dim));

  // else use the result descriptor AnyDim() intrinsic

  // Create mutable fir.box to be passed to the runtime for the result.

  auto resultArrayType = builder.getVarLenSeqTy(resultType, rank - 1);
  auto resultMutableBox =
       Fortran::lower::createTempMutableBox(builder, loc, resultArrayType);
  auto resultIrBox =
       Fortran::lower::getMutableIRBox(builder, loc, resultMutableBox);

  // Call runtime. The runtime is allocating the result.
  Fortran::lower::genAnyDescriptor(builder, loc, resultIrBox, mask, dim);
  return Fortran::lower::genMutableBoxRead(builder, loc, resultMutableBox)
      .match(
          [&](const fir::ArrayBoxValue &box) -> fir::ExtendedValue {
            addCleanUpForTemp(loc, box.getAddr());
            auto ldVal = builder.create<fir::LoadOp>(loc, box.getAddr());
            return builder.createConvert(loc, resultArrayType, ldVal);
          },
          [&](const auto &) -> fir::ExtendedValue {
            fir::emitFatalError(loc, "Invalid result for ANY");
          });
}

// ASSOCIATED
fir::ExtendedValue
IntrinsicLibrary::genAssociated(mlir::Type resultType,
                                llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  // TODO: Runtime would be better to deal with all the cases with TARGET
  // arguments.
  if (!isAbsent(args[1]))
    TODO(loc, "ASSOCIATED intrinsic with TARGET argument");
  return args[0].match(
      [&](const fir::MutableBoxValue &x) -> fir::ExtendedValue {
        return Fortran::lower::genIsAllocatedOrAssociatedTest(builder, loc, x);
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc,
                            "associated arg not lowered to MutableBoxValue");
      });
}

// AINT
mlir::Value IntrinsicLibrary::genAint(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() >= 1);
  // Skip optional kind argument to search the runtime; it is already reflected
  // in result type.
  return genRuntimeCall("aint", resultType, {args[0]});
}

// CEILING
mlir::Value IntrinsicLibrary::genCeiling(mlir::Type resultType,
                                         llvm::ArrayRef<mlir::Value> args) {
  // Optional KIND argument.
  assert(args.size() >= 1);
  auto arg = args[0];
  // Use ceil that is not an actual Fortran intrinsic but that is
  // an llvm intrinsic that does the same, but return a floating
  // point.
  auto ceil = genRuntimeCall("ceil", arg.getType(), {arg});
  return builder.createConvert(loc, resultType, ceil);
}

fir::ExtendedValue
IntrinsicLibrary::genChar(mlir::Type type,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  // Optional KIND argument.
  assert(args.size() >= 1);
  auto *arg = args[0].getUnboxed();
  // expect argument to be a scalar integer
  if (!arg)
    mlir::emitError(loc, "CHAR intrinsic argument not unboxed");
  Fortran::lower::CharacterExprHelper helper{builder, loc};
  auto kind = helper.getCharacterType(type).getFKind();
  auto cast = helper.createSingletonFromCode(*arg, kind);
  auto len =
      builder.createIntegerConstant(loc, builder.getCharacterLengthType(), 1);
  return fir::CharBoxValue{cast, len};
}

// CONJG
mlir::Value IntrinsicLibrary::genConjg(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);
  if (resultType != args[0].getType())
    llvm_unreachable("argument type mismatch");

  mlir::Value cplx = args[0];
  auto imag =
      Fortran::lower::ComplexExprHelper{builder, loc}.extractComplexPart(
          cplx, /*isImagPart=*/true);
  auto negImag = builder.create<fir::NegfOp>(loc, imag);
  return Fortran::lower::ComplexExprHelper{builder, loc}.insertComplexPart(
      cplx, negImag, /*isImagPart=*/true);
}

// DATE_AND_TIME
void IntrinsicLibrary::genDateAndTime(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4 && "date_and_time has 4 args");
  llvm::SmallVector<llvm::Optional<fir::CharBoxValue>> charArgs(3);
  for (auto i = 0; i < 3; ++i)
    if (auto *charBox = args[i].getCharBox())
      charArgs[i] = *charBox;
  // TODO: build descriptor for VALUES (also update runtime)
  if (fir::getBase(args[3]))
    mlir::emitError(loc, "TODO: lowering of DATE_AND_TIME VALUES argument not "
                         "yet implemented\n");

  Fortran::lower::genDateAndTime(builder, loc, charArgs[0], charArgs[1],
                                 charArgs[2]);
}

// DIM
mlir::Value IntrinsicLibrary::genDim(mlir::Type resultType,
                                     llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  if (resultType.isa<mlir::IntegerType>()) {
    auto zero = builder.createIntegerConstant(loc, resultType, 0);
    auto diff = builder.create<mlir::SubIOp>(loc, args[0], args[1]);
    auto cmp =
        builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt, diff, zero);
    return builder.create<mlir::SelectOp>(loc, cmp, diff, zero);
  }
  assert(fir::isa_real(resultType) && "Only expects real and integer in DIM");
  auto zero = builder.createRealZeroConstant(loc, resultType);
  auto diff = builder.create<mlir::SubFOp>(loc, args[0], args[1]);
  auto cmp =
      builder.create<fir::CmpfOp>(loc, mlir::CmpFPredicate::OGT, diff, zero);
  return builder.create<mlir::SelectOp>(loc, cmp, diff, zero);
}

// DPROD
mlir::Value IntrinsicLibrary::genDprod(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  assert(fir::isa_real(resultType) &&
         "Result must be double precision in DPROD");
  auto a = builder.createConvert(loc, resultType, args[0]);
  auto b = builder.createConvert(loc, resultType, args[1]);
  return builder.create<mlir::MulFOp>(loc, a, b);
}

// FLOOR
mlir::Value IntrinsicLibrary::genFloor(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  // Optional KIND argument.
  assert(args.size() >= 1);
  auto arg = args[0];
  // Use LLVM floor that returns real.
  auto floor = genRuntimeCall("floor", arg.getType(), {arg});
  return builder.createConvert(loc, resultType, floor);
}

// IAND
mlir::Value IntrinsicLibrary::genIAnd(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  return builder.create<mlir::AndOp>(loc, args[0], args[1]);
}

// ICHAR
fir::ExtendedValue
IntrinsicLibrary::genIchar(mlir::Type resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  // There can be an optional kind in second argument.
  assert(args.size() == 2);
  auto charBox = args[0].getCharBox();
  if (!charBox)
    llvm::report_fatal_error("expected character scalar");

  Fortran::lower::CharacterExprHelper helper{builder, loc};
  auto buffer = charBox->getBuffer();
  auto bufferTy = buffer.getType();
  mlir::Value charVal;
  if (auto charTy = bufferTy.dyn_cast<fir::CharacterType>()) {
    assert(charTy.singleton());
    charVal = buffer;
  } else {
    // Character is in memory, cast to fir.ref<char> and load.
    auto ty = fir::dyn_cast_ptrEleTy(bufferTy);
    if (!ty)
      llvm::report_fatal_error("expected memory type");
    // The length of in the character type may be unknown. Casting
    // to a singleton ref is required before loading.
    auto eleType = helper.getCharacterType(ty);
    auto charType =
        fir::CharacterType::get(builder.getContext(), eleType.getFKind(), 1);
    auto toTy = builder.getRefType(charType);
    auto cast = builder.createConvert(loc, toTy, buffer);
    charVal = builder.create<fir::LoadOp>(loc, cast);
  }
  LLVM_DEBUG(llvm::dbgs() << "ichar(" << charVal << ")\n");
  auto code = helper.extractCodeFromSingleton(charVal);
  return builder.createConvert(loc, resultType, code);
}

// IEOR
mlir::Value IntrinsicLibrary::genIEOr(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  return builder.create<mlir::XOrOp>(loc, args[0], args[1]);
}

// INDEX
fir::ExtendedValue
IntrinsicLibrary::genIndex(mlir::Type resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  for (int i = 0; i < 4; ++i)
    if (fir::getBase(args[i]))
      LLVM_DEBUG(llvm::dbgs() << "index(" << i << "): " << args[i] << '\n');

  auto stringBase = fir::getBase(args[0]);
  auto kind =
      Fortran::lower::CharacterExprHelper{builder, loc}.getCharacterKind(
          stringBase.getType());
  auto stringLen = fir::getLen(args[0]);
  auto substringBase = fir::getBase(args[1]);
  auto substringLen = fir::getLen(args[1]);
  mlir::Value back =
      fir::isUnboxedValue(args[2])
          ? fir::getBase(args[2])
          : builder.createIntegerConstant(loc, builder.getI1Type(), 0);
  if (!fir::isUnboxedValue(args[3]))
    return builder.createConvert(
        loc, resultType,
        Fortran::lower::genIndex(builder, loc, kind, stringBase, stringLen,
                                 substringBase, substringLen, back));

  // Call the descriptor-based Index implementation
  auto string = builder.createBox(loc, args[0]);
  auto substring = builder.createBox(loc, args[1]);
  auto makeRefThenEmbox = [&](mlir::Value b) {
    auto logTy = fir::LogicalType::get(
        builder.getContext(), builder.getKindMap().defaultLogicalKind());
    auto temp = builder.createTemporary(loc, logTy);
    auto castb = builder.createConvert(loc, logTy, b);
    builder.create<fir::StoreOp>(loc, castb, temp);
    return builder.createBox(loc, temp);
  };
  auto backOpt = fir::isUnboxedValue(args[2])
                     ? makeRefThenEmbox(*args[2].getUnboxed())
                     : builder.create<fir::AbsentOp>(
                           loc, fir::BoxType::get(builder.getI1Type()));
  auto kindVal = fir::isUnboxedValue(args[3])
                     ? fir::getBase(args[3])
                     : builder.createIntegerConstant(
                           loc, builder.getIndexType(),
                           builder.getKindMap().defaultIntegerKind());
  // Create mutable fir.box to be passed to the runtime for the result.
  auto mutBox = Fortran::lower::createTempMutableBox(builder, loc, resultType);
  auto resBox = Fortran::lower::getMutableIRBox(builder, loc, mutBox);
  // Call runtime. The runtime is allocating the result.
  Fortran::lower::genIndexDescriptor(builder, loc, resBox, string, substring,
                                     backOpt, kindVal);
  // Read back the result from the mutable box.
  return Fortran::lower::genMutableBoxRead(builder, loc, mutBox)
      .match(
          [&](const fir::AbstractBox &box) -> fir::ExtendedValue {
            addCleanUpForTemp(loc, box.getAddr());
            auto ldVal = builder.create<fir::LoadOp>(loc, box.getAddr());
            return builder.createConvert(loc, resultType, ldVal);
          },
          [&](const auto &) -> fir::ExtendedValue {
            fir::emitFatalError(loc, "result of INDEX is not a scalar integer");
          });
}

// IOR
mlir::Value IntrinsicLibrary::genIOr(mlir::Type resultType,
                                     llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  return builder.create<mlir::OrOp>(loc, args[0], args[1]);
}

// LEN
// Note that this is only used for unrestricted intrinsic.
// Usage of LEN are otherwise rewritten as descriptor inquiries by the
// front-end.
fir::ExtendedValue
IntrinsicLibrary::genLen(mlir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {
  // Optional KIND argument reflected in result type.
  assert(args.size() >= 1);
  mlir::Value len;
  if (const auto *charBox = args[0].getCharBox()) {
    len = charBox->getLen();
  } else if (const auto *charBoxArray = args[0].getCharBox()) {
    len = charBoxArray->getLen();
  } else {
    Fortran::lower::CharacterExprHelper helper{builder, loc};
    len = helper.createUnboxChar(fir::getBase(args[0])).second;
  }

  return builder.createConvert(loc, resultType, len);
}

// LEN_TRIM
fir::ExtendedValue
IntrinsicLibrary::genLenTrim(mlir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  // Optional KIND argument reflected in result type.
  assert(args.size() >= 1);
  Fortran::lower::CharacterExprHelper helper{builder, loc};
  auto *charBox = args[0].getCharBox();
  if (!charBox)
    TODO(loc, "character array len_trim");
  auto len = helper.createLenTrim(*charBox);
  return builder.createConvert(loc, resultType, len);
}

// MERGE
mlir::Value IntrinsicLibrary::genMerge(mlir::Type,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 3);
  auto mask = builder.createConvert(loc, builder.getI1Type(), args[2]);
  return builder.create<mlir::SelectOp>(loc, mask, args[0], args[1]);
}

// MOD
mlir::Value IntrinsicLibrary::genMod(mlir::Type resultType,
                                     llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  if (resultType.isa<mlir::IntegerType>())
    return builder.create<mlir::SignedRemIOp>(loc, args[0], args[1]);

  // Use runtime. Note that mlir::RemFOp implements floating point
  // remainder, but it does not work with fir::Real type.
  // TODO: consider using mlir::RemFOp when possible, that may help folding
  // and  optimizations.
  return genRuntimeCall("mod", resultType, args);
}

// MODULO
mlir::Value IntrinsicLibrary::genModulo(mlir::Type resultType,
                                        llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  // No floored modulo op in LLVM/MLIR yet. TODO: add one to MLIR.
  // In the meantime, use a simple inlined implementation based on truncated
  // modulo (MOD(A, P) implemented by RemIOp, RemFOp). This avoids making manual
  // division and multiplication from MODULO formula.
  //  - If A/P > 0 or MOD(A,P)=0, then INT(A/P) = FLOOR(A/P), and MODULO = MOD.
  //  - Otherwise, when A/P < 0 and MOD(A,P) !=0, then MODULO(A, P) =
  //    A-FLOOR(A/P)*P = A-(INT(A/P)-1)*P = A-INT(A/P)*P+P = MOD(A,P)+P
  // Note that A/P < 0 if and only if A and P signs are different.
  if (resultType.isa<mlir::IntegerType>()) {
    auto remainder = builder.create<mlir::SignedRemIOp>(loc, args[0], args[1]);
    auto argXor = builder.create<mlir::XOrOp>(loc, args[0], args[1]);
    auto zero = builder.createIntegerConstant(loc, argXor.getType(), 0);
    auto argSignDifferent = builder.create<mlir::CmpIOp>(
        loc, mlir::CmpIPredicate::slt, argXor, zero);
    auto remainderIsNotZero = builder.create<mlir::CmpIOp>(
        loc, mlir::CmpIPredicate::ne, remainder, zero);
    auto mustAddP =
        builder.create<mlir::AndOp>(loc, remainderIsNotZero, argSignDifferent);
    auto remPlusP = builder.create<mlir::AddIOp>(loc, remainder, args[1]);
    return builder.create<mlir::SelectOp>(loc, mustAddP, remPlusP, remainder);
  }
  // Real case
  auto remainder = builder.create<mlir::RemFOp>(loc, args[0], args[1]);
  auto zero = builder.createRealZeroConstant(loc, remainder.getType());
  auto remainderIsNotZero = builder.create<mlir::CmpFOp>(
      loc, mlir::CmpFPredicate::UNE, remainder, zero);
  auto aLessThanZero = builder.create<mlir::CmpFOp>(
      loc, mlir::CmpFPredicate::OLT, args[0], zero);
  auto pLessThanZero = builder.create<mlir::CmpFOp>(
      loc, mlir::CmpFPredicate::OLT, args[1], zero);
  auto argSignDifferent =
      builder.create<mlir::XOrOp>(loc, aLessThanZero, pLessThanZero);
  auto mustAddP =
      builder.create<mlir::AndOp>(loc, remainderIsNotZero, argSignDifferent);
  auto remPlusP = builder.create<mlir::AddFOp>(loc, remainder, args[1]);
  return builder.create<mlir::SelectOp>(loc, mustAddP, remPlusP, remainder);
}

// NINT
mlir::Value IntrinsicLibrary::genNint(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() >= 1);
  // Skip optional kind argument to search the runtime; it is already reflected
  // in result type.
  return genRuntimeCall("nint", resultType, {args[0]});
}

// PRESENT
fir::ExtendedValue
IntrinsicLibrary::genPresent(mlir::Type,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  return builder.create<fir::IsPresentOp>(loc, builder.getI1Type(),
                                          fir::getBase(args[0]));
}

// SCAN
fir::ExtendedValue
IntrinsicLibrary::genScan(mlir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 4);

  if (isAbsent(args[3])) {
    // Kind not specified, so call scan/verify runtime routine that is 
    // specialized on the kind of characters in string.

    // Handle required string base arg
    auto stringBase = fir::getBase(args[0]);

    // Handle required set string base arg
    auto setBase = fir::getBase(args[1]);

    // Handle kind argument; it is the kind of character in this case
    auto kind =
      Fortran::lower::CharacterExprHelper{builder, loc}.getCharacterKind(  
          stringBase.getType());

    // Get string length argument
    auto stringLen = fir::getLen(args[0]);   

    // Get set string length argument
    auto setLen = fir::getLen(args[1]);

    // Handle optional back argument
    auto back = isAbsent(args[2])
                ? builder.createIntegerConstant(loc, builder.getI1Type(), 0) 
                : fir::getBase(args[2]);

    return builder.createConvert(
        loc, resultType,
        Fortran::lower::genScan(builder, loc, kind, stringBase, 
                                stringLen, setBase, setLen, back));
  }
  // else use the runtime descriptor version of scan/verify
 

  // Handle optional argument, back
  auto makeRefThenEmbox = [&](mlir::Value b) {
    auto logTy = fir::LogicalType::get(
      builder.getContext(), builder.getKindMap().defaultLogicalKind());
    auto temp = builder.createTemporary(loc, logTy);
    auto castb = builder.createConvert(loc, logTy, b);
    builder.create<fir::StoreOp>(loc, castb, temp);
    return builder.createBox(loc, temp);
  };
  auto back = fir::isUnboxedValue(args[2])
              ? makeRefThenEmbox(*args[2].getUnboxed())
              : builder.create<fir::AbsentOp>(
                loc, fir::BoxType::get(builder.getI1Type()));

  // Handle required string argument
  auto string = builder.createBox(loc, args[0]);

  // Handle required set argument
  auto set = builder.createBox(loc, args[1]);

  // Handle kind argument
  auto kind = fir::getBase(args[3]);
  
  // Create result descriptor    
  auto resultMutableBox =
       Fortran::lower::createTempMutableBox(builder, loc, resultType);
  auto resultIrBox =
       Fortran::lower::getMutableIRBox(builder, loc, resultMutableBox);

  Fortran::lower::genScanDescriptor(builder, loc, resultIrBox, string, set, 
                                    back, kind);

  // Handle cleanup of allocatable result descriptor and return
  auto res = Fortran::lower::genMutableBoxRead(builder, loc, resultMutableBox);

  return res.match(
        [&](const Fortran::lower::SymbolBox::Intrinsic &box)
            -> fir::ExtendedValue { 
            addCleanUpForTemp(loc, box.getAddr());
            return builder.create<fir::LoadOp>(loc, resultType, box.getAddr());
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc, "unexpected result for SCAN");
      });
}

// SIGN
mlir::Value IntrinsicLibrary::genSign(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  auto abs = genAbs(resultType, {args[0]});
  if (resultType.isa<mlir::IntegerType>()) {
    auto zero = builder.createIntegerConstant(loc, resultType, 0);
    auto neg = builder.create<mlir::SubIOp>(loc, zero, abs);
    auto cmp = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt,
                                            args[1], zero);
    return builder.create<mlir::SelectOp>(loc, cmp, neg, abs);
  }
  // TODO: Requirements when second argument is +0./0.
  auto zero = builder.createRealZeroConstant(loc, resultType);
  auto neg = builder.create<fir::NegfOp>(loc, abs);
  auto cmp =
      builder.create<fir::CmpfOp>(loc, mlir::CmpFPredicate::OLT, args[1], zero);
  return builder.create<mlir::SelectOp>(loc, cmp, neg, abs);
}

// TRIM
fir::ExtendedValue
IntrinsicLibrary::genTrim(mlir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  auto string = builder.createBox(loc, args[0]);
  // Create mutable fir.box to be passed to the runtime for the result.
  auto resultMutableBox =
      Fortran::lower::createTempMutableBox(builder, loc, resultType);
  auto resultIrBox =
      Fortran::lower::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  Fortran::lower::genTrim(builder, loc, resultIrBox, string);
  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  auto res = Fortran::lower::genMutableBoxRead(builder, loc, resultMutableBox);
  return res.match(
      [&](const fir::CharBoxValue &box) -> fir::ExtendedValue {
        addCleanUpForTemp(loc, fir::getBase(box));
        return box;
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc, "result of TRIM is not a scalar character");
      });
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
  if (fir::isa_real(type)) {
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
      // TODO: ieeeMinNum/ieeeMaxNum
      static_assert(behavior == ExtremumBehavior::IeeeMinMaxNum,
                    "ieeeMinNum/ieeeMaxNum behavior not implemented");
    }
  } else if (fir::isa_integer(type)) {
    result = builder.create<mlir::CmpIOp>(loc, integerPredicate, left, right);
  } else if (fir::isa_char(type)) {
    // TODO: ! character min and max is tricky because the result
    // length is the length of the longest argument!
    // So we may need a temp.
    TODO(loc, "CHARACTER min and max");
  }
  assert(result && "result must be defined");
  return result;
}


// VERIFY
fir::ExtendedValue
IntrinsicLibrary::genVerify(mlir::Type resultType,
                                llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 4);

  if (isAbsent(args[3])) {
    // Kind not specified, so call scan/verify runtime routine that is 
    // specialized on the kind of characters in string.

    // Handle required string base arg
    auto stringBase = fir::getBase(args[0]);

    // Handle required set string base arg
    auto setBase = fir::getBase(args[1]);

    // Handle kind argument; it is the kind of character in this case
    auto kind =
      Fortran::lower::CharacterExprHelper{builder, loc}.getCharacterKind(  
          stringBase.getType());

    // Get string length argument
    auto stringLen = fir::getLen(args[0]);   

    // Get set string length argument
    auto setLen = fir::getLen(args[1]);

    // Handle optional back argument
    auto back = isAbsent(args[2])
                ? builder.createIntegerConstant(loc, builder.getI1Type(), 0) 
                : fir::getBase(args[2]);

    return builder.createConvert(
        loc, resultType,
        Fortran::lower::genVerify(builder, loc, kind, stringBase, 
                                  stringLen, setBase, setLen, back));
  }
  // else use the runtime descriptor version of scan/verify
 

  // Handle optional argument, back
  auto makeRefThenEmbox = [&](mlir::Value b) {
    auto logTy = fir::LogicalType::get(
      builder.getContext(), builder.getKindMap().defaultLogicalKind());
    auto temp = builder.createTemporary(loc, logTy);
    auto castb = builder.createConvert(loc, logTy, b);
    builder.create<fir::StoreOp>(loc, castb, temp);
    return builder.createBox(loc, temp);
  };
  auto back = fir::isUnboxedValue(args[2])
              ? makeRefThenEmbox(*args[2].getUnboxed())
              : builder.create<fir::AbsentOp>(
                loc, fir::BoxType::get(builder.getI1Type()));

  // Handle required string argument
  auto string = builder.createBox(loc, args[0]);

  // Handle required set argument
  auto set = builder.createBox(loc, args[1]);

  // Handle kind argument
  auto kind = fir::getBase(args[3]);
  
  // Create result descriptor    
  auto resultMutableBox =
       Fortran::lower::createTempMutableBox(builder, loc, resultType);
  auto resultIrBox =
       Fortran::lower::getMutableIRBox(builder, loc, resultMutableBox);

  Fortran::lower::genVerifyDescriptor(builder, loc, resultIrBox, string, set, 
                                      back, kind);

  // Handle cleanup of allocatable result descriptor and return
  auto res = Fortran::lower::genMutableBoxRead(builder, loc, resultMutableBox);

  return res.match(
        [&](const Fortran::lower::SymbolBox::Intrinsic &box)
            -> fir::ExtendedValue { 
            addCleanUpForTemp(loc, box.getAddr());
            return builder.create<fir::LoadOp>(loc, resultType, box.getAddr());
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc, "unexpected result for VERIFY");
      });
}

// MIN and MAX
template <Extremum extremum, ExtremumBehavior behavior>
mlir::Value IntrinsicLibrary::genExtremum(mlir::Type,
                                          llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() >= 1);
  mlir::Value result = args[0];
  for (auto arg : args.drop_front()) {
    auto mask =
        createExtremumCompare<extremum, behavior>(loc, builder, result, arg);
    result = builder.create<mlir::SelectOp>(loc, mask, result, arg);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Argument lowering rules interface
//===----------------------------------------------------------------------===//

const Fortran::lower::IntrinsicArgumentLoweringRules *
Fortran::lower::getIntrinsicArgumentLowering(llvm::StringRef intrinsicName) {
  for (const auto &handler : handlers)
    if (intrinsicName == handler.name)
      if (!handler.argLoweringRules.hasDefaultRules())
        return &handler.argLoweringRules;
  return nullptr;
}

/// Return how argument \p argName should be lowered given the rules for the
/// intrinsic function.
Fortran::lower::LowerIntrinsicArgAs Fortran::lower::lowerIntrinsicArgumentAs(
    mlir::Location loc, const IntrinsicArgumentLoweringRules &rules,
    llvm::StringRef argName) {
  for (const auto &arg : rules.args) {
    if (arg.name && arg.name == argName)
      return arg.lowerAs;
  }
  fir::emitFatalError(
      loc, "internal: unknown intrinsic argument name in lowering '" + argName +
               "'");
}

//===----------------------------------------------------------------------===//
// Public intrinsic call helpers
//===----------------------------------------------------------------------===//

fir::ExtendedValue
Fortran::lower::genIntrinsicCall(Fortran::lower::FirOpBuilder &builder,
                                 mlir::Location loc, llvm::StringRef name,
                                 llvm::Optional<mlir::Type> resultType,
                                 llvm::ArrayRef<fir::ExtendedValue> args,
                                 Fortran::lower::StatementContext &stmtCtx) {
  return IntrinsicLibrary{builder, loc, &stmtCtx}.genIntrinsicCall(
      name, resultType, args);
}

mlir::Value Fortran::lower::genMax(Fortran::lower::FirOpBuilder &builder,
                                   mlir::Location loc,
                                   llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() > 0 && "max requires at least one argument");
  return IntrinsicLibrary{builder, loc}
      .genExtremum<Extremum::Max, ExtremumBehavior::MinMaxss>(args[0].getType(),
                                                              args);
}

mlir::Value Fortran::lower::genMin(Fortran::lower::FirOpBuilder &builder,
                                   mlir::Location loc,
                                   llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() > 0 && "min requires at least one argument");
  return IntrinsicLibrary{builder, loc}
      .genExtremum<Extremum::Min, ExtremumBehavior::MinMaxss>(args[0].getType(),
                                                              args);
}

mlir::Value Fortran::lower::genPow(Fortran::lower::FirOpBuilder &builder,
                                   mlir::Location loc, mlir::Type type,
                                   mlir::Value x, mlir::Value y) {
  return IntrinsicLibrary{builder, loc}.genRuntimeCall("pow", type, {x, y});
}

mlir::SymbolRefAttr Fortran::lower::getUnrestrictedIntrinsicSymbolRefAttr(
    Fortran::lower::FirOpBuilder &builder, mlir::Location loc,
    llvm::StringRef name, mlir::FunctionType signature) {
  return IntrinsicLibrary{builder, loc}.getUnrestrictedIntrinsicSymbolRefAttr(
      name, signature);
}
