//===-- Builder/IntrinsicCall.h -- lowering of intrinsics -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_INTRINSICCALL_H
#define FORTRAN_LOWER_INTRINSICCALL_H

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Character.h"
#include "flang/Optimizer/Builder/Runtime/Numeric.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/entry-names.h"
#include "flang/Runtime/iostat.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include <optional>

namespace fir {

class StatementContext;

// TODO: Error handling interface ?
// TODO: Implementation is incomplete. Many intrinsics to tbd.

/// Same as the other genIntrinsicCall version above, except that the result
/// deallocation, if required, is not added to a StatementContext. Instead, an
/// extra boolean result indicates if the result must be freed after use.
std::pair<fir::ExtendedValue, bool>
genIntrinsicCall(fir::FirOpBuilder &, mlir::Location, llvm::StringRef name,
                 std::optional<mlir::Type> resultType,
                 llvm::ArrayRef<fir::ExtendedValue> args);

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

/// Enum specifying how intrinsic argument evaluate::Expr should be
/// lowered to fir::ExtendedValue to be passed to genIntrinsicCall.
enum class LowerIntrinsicArgAs {
  /// Lower argument to a value. Mainly intended for scalar arguments.
  Value,
  /// Lower argument to an address. Only valid when the argument properties are
  /// fully defined (e.g. allocatable is allocated...).
  Addr,
  /// Lower argument to a box.
  Box,
  /// Lower argument without assuming that the argument is fully defined.
  /// It can be used on unallocated allocatable, disassociated pointer,
  /// or absent optional. This is meant for inquiry intrinsic arguments.
  Inquired
};

/// Define how a given intrinsic argument must be lowered.
struct ArgLoweringRule {
  LowerIntrinsicArgAs lowerAs;
  /// Value:
  //    - Numerical: 0
  //    - Logical : false
  //    - Derived/character: not possible. Need custom intrinsic lowering.
  //  Addr:
  //    - nullptr
  //  Box:
  //    - absent box
  //  AsInquired:
  //    - no-op
  bool handleDynamicOptional;
};

constexpr auto asValue = fir::LowerIntrinsicArgAs::Value;
constexpr auto asAddr = fir::LowerIntrinsicArgAs::Addr;
constexpr auto asBox = fir::LowerIntrinsicArgAs::Box;
constexpr auto asInquired = fir::LowerIntrinsicArgAs::Inquired;

/// Opaque class defining the argument lowering rules for all the argument of
/// an intrinsic.
struct IntrinsicArgumentLoweringRules;

// TODO error handling -> return a code or directly emit messages ?
struct IntrinsicLibrary {

  // Constructors.
  explicit IntrinsicLibrary(fir::FirOpBuilder &builder, mlir::Location loc)
      : builder{builder}, loc{loc} {}
  IntrinsicLibrary() = delete;
  IntrinsicLibrary(const IntrinsicLibrary &) = delete;

  /// Generate FIR for call to Fortran intrinsic \p name with arguments \p arg
  /// and expected result type \p resultType. Return the result and a boolean
  /// that, if true, indicates that the result must be freed after use.
  std::pair<fir::ExtendedValue, bool>
  genIntrinsicCall(llvm::StringRef name, std::optional<mlir::Type> resultType,
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

  void genAbort(llvm::ArrayRef<fir::ExtendedValue>);
  /// Lowering for the ABS intrinsic. The ABS intrinsic expects one argument in
  /// the llvm::ArrayRef. The ABS intrinsic is lowered into MLIR/FIR operation
  /// if the argument is an integer, into llvm intrinsics if the argument is
  /// real and to the `hypot` math routine if the argument is of complex type.
  mlir::Value genAbs(mlir::Type, llvm::ArrayRef<mlir::Value>);
  template <void (*CallRuntime)(fir::FirOpBuilder &, mlir::Location loc,
                                mlir::Value, mlir::Value)>
  fir::ExtendedValue genAdjustRtCall(mlir::Type,
                                     llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genAimag(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genAint(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genAll(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genAllocated(mlir::Type,
                                  llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genAnint(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genAny(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genAtand(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue
      genCommandArgumentCount(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genAssociated(mlir::Type,
                                   llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genBesselJn(mlir::Type,
                                 llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genBesselYn(mlir::Type,
                                 llvm::ArrayRef<fir::ExtendedValue>);
  /// Lower a bitwise comparison intrinsic using the given comparator.
  template <mlir::arith::CmpIPredicate pred>
  mlir::Value genBitwiseCompare(mlir::Type resultType,
                                llvm::ArrayRef<mlir::Value> args);

  mlir::Value genBtest(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genCeiling(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genChar(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  template <mlir::arith::CmpIPredicate pred>
  fir::ExtendedValue genCharacterCompare(mlir::Type,
                                         llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genCmplx(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genConjg(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genCount(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genCpuTime(llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genCshift(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genCAssociatedCFunPtr(mlir::Type,
                                           llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genCAssociatedCPtr(mlir::Type,
                                        llvm::ArrayRef<fir::ExtendedValue>);
  void genCFPointer(llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genCFunLoc(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genCLoc(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genDateAndTime(llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genDim(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genDotProduct(mlir::Type,
                                   llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genDprod(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genDshiftl(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genDshiftr(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genEoshift(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genExit(llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genExponent(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genExtendsTypeOf(mlir::Type,
                                      llvm::ArrayRef<fir::ExtendedValue>);
  template <Extremum, ExtremumBehavior>
  mlir::Value genExtremum(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genFloor(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genFraction(mlir::Type resultType,
                          mlir::ArrayRef<mlir::Value> args);
  void genGetCommand(mlir::ArrayRef<fir::ExtendedValue> args);
  void genGetCommandArgument(mlir::ArrayRef<fir::ExtendedValue> args);
  void genGetEnvironmentVariable(llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genIall(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  /// Lowering for the IAND intrinsic. The IAND intrinsic expects two arguments
  /// in the llvm::ArrayRef.
  mlir::Value genIand(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genIany(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genIbclr(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIbits(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIbset(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genIchar(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genFindloc(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genIeeeClass(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIeeeCopySign(mlir::Type, llvm::ArrayRef<mlir::Value>);
  void genIeeeGetRoundingMode(llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genIeeeIsFinite(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIeeeIsNan(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIeeeIsNegative(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIeeeIsNormal(mlir::Type, llvm::ArrayRef<mlir::Value>);
  void genIeeeSetRoundingMode(llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genIeeeSignbit(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIeeeSupportRounding(mlir::Type, llvm::ArrayRef<mlir::Value>);
  template <mlir::arith::CmpIPredicate pred>
  mlir::Value genIeeeTypeCompare(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIeeeUnordered(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIeeeValue(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIeor(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genIndex(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genIor(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genIparity(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genIsContiguous(mlir::Type,
                                     llvm::ArrayRef<fir::ExtendedValue>);
  template <Fortran::runtime::io::Iostat value>
  mlir::Value genIsIostatValue(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIsFPClass(mlir::Type, llvm::ArrayRef<mlir::Value>,
                           int fpclass);
  mlir::Value genIshft(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIshftc(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genLbound(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genLeadz(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genLen(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genLenTrim(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genLoc(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  template <typename Shift>
  mlir::Value genMask(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genMatmul(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMatmulTranspose(mlir::Type,
                                        llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMaxloc(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMaxval(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMerge(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genMergeBits(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genMinloc(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMinval(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genMod(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genModulo(mlir::Type, llvm::ArrayRef<mlir::Value>);
  void genMoveAlloc(llvm::ArrayRef<fir::ExtendedValue>);
  void genMvbits(llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genNearest(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genNint(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genNorm2(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genNot(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genNull(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genPack(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genParity(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genPopcnt(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genPoppar(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genPresent(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genProduct(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genRandomInit(llvm::ArrayRef<fir::ExtendedValue>);
  void genRandomNumber(llvm::ArrayRef<fir::ExtendedValue>);
  void genRandomSeed(llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genReduce(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genRepeat(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genReshape(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genRRSpacing(mlir::Type resultType,
                           llvm::ArrayRef<mlir::Value> args);
  fir::ExtendedValue genSameTypeAs(mlir::Type,
                                   llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genScale(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genScan(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genSelectedIntKind(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genSelectedRealKind(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genSetExponent(mlir::Type resultType,
                             llvm::ArrayRef<mlir::Value> args);
  template <typename Shift>
  mlir::Value genShift(mlir::Type resultType, llvm::ArrayRef<mlir::Value>);
  mlir::Value genShiftA(mlir::Type resultType, llvm::ArrayRef<mlir::Value>);
  mlir::Value genSign(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genSize(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genSpacing(mlir::Type resultType,
                         llvm::ArrayRef<mlir::Value> args);
  fir::ExtendedValue genSpread(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genStorageSize(mlir::Type,
                                    llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genSum(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genSystemClock(llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genTand(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genTrailz(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genTransfer(mlir::Type,
                                 llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genTranspose(mlir::Type,
                                  llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genTrim(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genUbound(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genUnpack(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genVerify(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  /// Implement all conversion functions like DBLE, the first argument is
  /// the value to convert. There may be an additional KIND arguments that
  /// is ignored because this is already reflected in the result type.
  mlir::Value genConversion(mlir::Type, llvm::ArrayRef<mlir::Value>);

  /// In the template helper below:
  ///  - "FN func" is a callback to generate the related intrinsic runtime call.
  ///  - "FD funcDim" is a callback to generate the "dim" runtime call.
  ///  - "FC funcChar" is a callback to generate the character runtime call.
  /// Helper for MinLoc/MaxLoc.
  template <typename FN, typename FD>
  fir::ExtendedValue genExtremumloc(FN func, FD funcDim, llvm::StringRef errMsg,
                                    mlir::Type,
                                    llvm::ArrayRef<fir::ExtendedValue>);
  template <typename FN, typename FD, typename FC>
  /// Helper for MinVal/MaxVal.
  fir::ExtendedValue genExtremumVal(FN func, FD funcDim, FC funcChar,
                                    llvm::StringRef errMsg,
                                    mlir::Type resultType,
                                    llvm::ArrayRef<fir::ExtendedValue> args);
  /// Process calls to Product, Sum, IAll, IAny, IParity intrinsic functions
  template <typename FN, typename FD>
  fir::ExtendedValue genReduction(FN func, FD funcDim, llvm::StringRef errMsg,
                                  mlir::Type resultType,
                                  llvm::ArrayRef<fir::ExtendedValue> args);

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
                           std::optional<mlir::Type> resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args);

  template <typename GeneratorType>
  mlir::func::FuncOp getWrapper(GeneratorType, llvm::StringRef name,
                                mlir::FunctionType,
                                bool loadRefArguments = false);

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

  /// Helper function for generating code clean-up for result descriptors
  fir::ExtendedValue readAndAddCleanUp(fir::MutableBoxValue resultMutableBox,
                                       mlir::Type resultType,
                                       llvm::StringRef errMsg);

  void setResultMustBeFreed() { resultMustBeFreed = true; }

  fir::FirOpBuilder &builder;
  mlir::Location loc;
  bool resultMustBeFreed = false;
};

struct IntrinsicDummyArgument {
  const char *name = nullptr;
  fir::LowerIntrinsicArgAs lowerAs = fir::LowerIntrinsicArgAs::Value;
  bool handleDynamicOptional = false;
};

/// This is shared by intrinsics and intrinsic module procedures.
struct IntrinsicArgumentLoweringRules {
  /// There is no more than 7 non repeated arguments in Fortran intrinsics.
  IntrinsicDummyArgument args[7];
  constexpr bool hasDefaultRules() const { return args[0].name == nullptr; }
};

/// Structure describing what needs to be done to lower intrinsic or intrinsic
/// module procedure "name".
struct IntrinsicHandler {
  const char *name;
  IntrinsicLibrary::Generator generator;
  // The following may be omitted in the table below.
  fir::IntrinsicArgumentLoweringRules argLoweringRules = {};
  bool isElemental = true;
  /// Code heavy intrinsic can be outlined to make FIR
  /// more readable.
  bool outline = false;
};

struct RuntimeFunction {
  // llvm::StringRef comparison operator are not constexpr, so use string_view.
  using Key = std::string_view;
  // Needed for implicit compare with keys.
  constexpr operator Key() const { return key; }
  Key key; // intrinsic name

  // Name of a runtime function that implements the operation.
  llvm::StringRef symbol;
  fir::runtime::FuncTypeBuilderFunc typeGenerator;
};

/// Callback type for generating lowering for a math operation.
using MathGeneratorTy = mlir::Value (*)(fir::FirOpBuilder &, mlir::Location,
                                        llvm::StringRef, mlir::FunctionType,
                                        llvm::ArrayRef<mlir::Value>);

struct MathOperation {
  // Overrides fir::runtime::FuncTypeBuilderFunc to add FirOpBuilder argument.
  using FuncTypeBuilderFunc = mlir::FunctionType (*)(mlir::MLIRContext *,
                                                     fir::FirOpBuilder &);

  // llvm::StringRef comparison operator are not constexpr, so use string_view.
  using Key = std::string_view;
  // Needed for implicit compare with keys.
  constexpr operator Key() const { return key; }
  // Intrinsic name.
  Key key;

  // Name of a runtime function that implements the operation.
  llvm::StringRef runtimeFunc;
  FuncTypeBuilderFunc typeGenerator;

  // A callback to generate FIR for the intrinsic defined by 'key'.
  // A callback may generate either dedicated MLIR operation(s) or
  // a function call to a runtime function with name defined by
  // 'runtimeFunc'.
  MathGeneratorTy funcGenerator;
};

// Enum of most supported intrinsic argument or return types.
enum class ParamTypeId {
  Void,
  Integer,
  Real,
  Complex,
  IntegerVector,
  UnsignedVector,
  RealVector,
};

// Helper function to get length of a 16-byte vector of element type eleTy.
static int getVecLen(mlir::Type eleTy) {
  assert((mlir::isa<mlir::IntegerType>(eleTy) ||
          mlir::isa<mlir::FloatType>(eleTy)) &&
         "unsupported vector element type");
  return 16 / (eleTy.getIntOrFloatBitWidth() / 8);
}

template <ParamTypeId t, int k>
struct ParamType {
  // Supported kinds can be checked with static asserts at compile time.
  static_assert(t != ParamTypeId::Integer || k == 1 || k == 2 || k == 4 ||
                    k == 8,
                "Unsupported integer kind");
  static_assert(t != ParamTypeId::Real || k == 4 || k == 8 || k == 10 ||
                    k == 16,
                "Unsupported real kind");
  static_assert(t != ParamTypeId::Complex || k == 2 || k == 3 || k == 4 ||
                    k == 8 || k == 10 || k == 16,
                "Unsupported complex kind");

  static const ParamTypeId ty = t;
  static const int kind = k;
};

// Namespace encapsulating type definitions for parameter types.
namespace Ty {
using Void = ParamType<ParamTypeId::Void, 0>;
template <int k>
using Real = ParamType<ParamTypeId::Real, k>;
template <int k>
using Integer = ParamType<ParamTypeId::Integer, k>;
template <int k>
using Complex = ParamType<ParamTypeId::Complex, k>;
template <int k>
using IntegerVector = ParamType<ParamTypeId::IntegerVector, k>;
template <int k>
using RealVector = ParamType<ParamTypeId::RealVector, k>;
template <int k>
using UnsignedVector = ParamType<ParamTypeId::UnsignedVector, k>;
} // namespace Ty

// Helper function that generates most types that are supported for intrinsic
// arguments and return type. Used by `genFuncType` to generate function
// types for most of the intrinsics.
static inline mlir::Type getTypeHelper(mlir::MLIRContext *context,
                                       fir::FirOpBuilder &builder,
                                       ParamTypeId typeId, int kind) {
  mlir::Type r;
  unsigned bits{0};
  switch (typeId) {
  case ParamTypeId::Void:
    llvm::report_fatal_error("can not get type of void");
    break;
  case ParamTypeId::Integer:
  case ParamTypeId::IntegerVector:
    bits = builder.getKindMap().getIntegerBitsize(kind);
    assert(bits != 0 && "failed to convert kind to integer bitsize");
    r = mlir::IntegerType::get(context, bits);
    break;
  case ParamTypeId::UnsignedVector:
    bits = builder.getKindMap().getIntegerBitsize(kind);
    assert(bits != 0 && "failed to convert kind to unsigned bitsize");
    r = mlir::IntegerType::get(context, bits, mlir::IntegerType::Unsigned);
    break;
  case ParamTypeId::Real:
  case ParamTypeId::RealVector:
    r = builder.getRealType(kind);
    break;
  case ParamTypeId::Complex:
    r = fir::ComplexType::get(context, kind);
    break;
  }

  mlir::Type fTy;
  switch (typeId) {
  case ParamTypeId::Void:
  case ParamTypeId::Integer:
  case ParamTypeId::Real:
  case ParamTypeId::Complex:
    // keep original type for void and non-vector
    fTy = r;
    break;
  case ParamTypeId::IntegerVector:
  case ParamTypeId::UnsignedVector:
  case ParamTypeId::RealVector:
    // convert to FIR vector type
    fTy = fir::VectorType::get(getVecLen(r), r);
    break;
  }
  return fTy;
}

// Generic function type generator that supports most of the function types
// used by intrinsics.
template <typename TyR, typename... ArgTys>
static inline mlir::FunctionType genFuncType(mlir::MLIRContext *context,
                                             fir::FirOpBuilder &builder) {
  llvm::SmallVector<ParamTypeId> argTys = {ArgTys::ty...};
  llvm::SmallVector<int> argKinds = {ArgTys::kind...};
  llvm::SmallVector<mlir::Type> argTypes;

  for (size_t i = 0; i < argTys.size(); ++i) {
    argTypes.push_back(getTypeHelper(context, builder, argTys[i], argKinds[i]));
  }

  if (TyR::ty == ParamTypeId::Void)
    return mlir::FunctionType::get(context, argTypes, std::nullopt);

  auto resType = getTypeHelper(context, builder, TyR::ty, TyR::kind);
  return mlir::FunctionType::get(context, argTypes, {resType});
}

//===----------------------------------------------------------------------===//
// Helper functions for argument handling.
//===----------------------------------------------------------------------===//
static inline mlir::Type getConvertedElementType(mlir::MLIRContext *context,
                                                 mlir::Type eleTy) {
  if (eleTy.isa<mlir::IntegerType>() && !eleTy.isSignlessInteger()) {
    const auto intTy{eleTy.dyn_cast<mlir::IntegerType>()};
    auto newEleTy{mlir::IntegerType::get(context, intTy.getWidth())};
    return newEleTy;
  }
  return eleTy;
}

static inline llvm::SmallVector<mlir::Value, 4>
getBasesForArgs(llvm::ArrayRef<fir::ExtendedValue> args) {
  llvm::SmallVector<mlir::Value, 4> baseVec;
  for (auto arg : args)
    baseVec.push_back(getBase(arg));
  return baseVec;
}

static inline llvm::SmallVector<mlir::Type, 4>
getTypesForArgs(llvm::ArrayRef<mlir::Value> args) {
  llvm::SmallVector<mlir::Type, 4> typeVec;
  for (auto arg : args)
    typeVec.push_back(arg.getType());
  return typeVec;
}

mlir::Value genLibCall(fir::FirOpBuilder &builder, mlir::Location loc,
                       llvm::StringRef libFuncName,
                       mlir::FunctionType libFuncType,
                       llvm::ArrayRef<mlir::Value> args);

template <typename T>
mlir::Value genMathOp(fir::FirOpBuilder &builder, mlir::Location loc,
                      llvm::StringRef mathLibFuncName,
                      mlir::FunctionType mathLibFuncType,
                      llvm::ArrayRef<mlir::Value> args);

template <typename T>
mlir::Value genComplexMathOp(fir::FirOpBuilder &builder, mlir::Location loc,
                             llvm::StringRef mathLibFuncName,
                             mlir::FunctionType mathLibFuncType,
                             llvm::ArrayRef<mlir::Value> args);

mlir::Value genLibSplitComplexArgsCall(fir::FirOpBuilder &builder,
                                       mlir::Location loc,
                                       llvm::StringRef libFuncName,
                                       mlir::FunctionType libFuncType,
                                       llvm::ArrayRef<mlir::Value> args);

/// Return argument lowering rules for an intrinsic.
/// Returns a nullptr if all the intrinsic arguments should be lowered by value.
const IntrinsicArgumentLoweringRules *
getIntrinsicArgumentLowering(llvm::StringRef intrinsicName);

/// Return how argument \p argName should be lowered given the rules for the
/// intrinsic function. The argument names are the one defined by the standard.
ArgLoweringRule lowerIntrinsicArgumentAs(const IntrinsicArgumentLoweringRules &,
                                         unsigned position);

/// Return place-holder for absent intrinsic arguments.
fir::ExtendedValue getAbsentIntrinsicArgument();

/// Get SymbolRefAttr of runtime (or wrapper function containing inlined
// implementation) of an unrestricted intrinsic (defined by its signature
// and generic name)
mlir::SymbolRefAttr
getUnrestrictedIntrinsicSymbolRefAttr(fir::FirOpBuilder &, mlir::Location,
                                      llvm::StringRef name,
                                      mlir::FunctionType signature);

//===----------------------------------------------------------------------===//
// Direct access to intrinsics that may be used by lowering outside
// of intrinsic call lowering.
//===----------------------------------------------------------------------===//

/// Generate maximum. There must be at least one argument and all arguments
/// must have the same type.
mlir::Value genMax(fir::FirOpBuilder &, mlir::Location,
                   llvm::ArrayRef<mlir::Value> args);

/// Generate minimum. Same constraints as genMax.
mlir::Value genMin(fir::FirOpBuilder &, mlir::Location,
                   llvm::ArrayRef<mlir::Value> args);

/// Generate Complex divide with the given expected
/// result type.
mlir::Value genDivC(fir::FirOpBuilder &builder, mlir::Location loc,
                    mlir::Type resultType, mlir::Value x, mlir::Value y);

/// Generate power function x**y with the given expected
/// result type.
mlir::Value genPow(fir::FirOpBuilder &, mlir::Location, mlir::Type resultType,
                   mlir::Value x, mlir::Value y);

} // namespace fir

#endif // FORTRAN_LOWER_INTRINSICCALL_H
