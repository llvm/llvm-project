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
#include "flang/Runtime/iostat-consts.h"
#include "aiir/Dialect/Complex/IR/Complex.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/Math/IR/Math.h"
#include <optional>

namespace Fortran {
namespace lower {
// TODO: remove the usage of AbstractConverter to avoid making IntrinsicCall.cpp
// depend upon Lower/Evaluate and use a data structure to pass options to
// IntrinsicLibrary.
class AbstractConverter;
} // namespace lower
} // namespace Fortran

namespace fir {

class StatementContext;
struct IntrinsicHandlerEntry;

/// Lower an intrinsic call given the intrinsic \p name, its \p resultType (that
/// must be std::nullopt if and only if this is a subroutine call), and its
/// lowered arguments \p args. The returned pair contains the result value
/// (null aiir::Value for subroutine calls), and a boolean that indicates if
/// this result must be freed after use.
std::pair<fir::ExtendedValue, bool>
genIntrinsicCall(fir::FirOpBuilder &, aiir::Location, llvm::StringRef name,
                 std::optional<aiir::Type> resultType,
                 llvm::ArrayRef<fir::ExtendedValue> args,
                 Fortran::lower::AbstractConverter *converter = nullptr);

/// Same as the entry above except that instead of an intrinsic name it takes an
/// IntrinsicHandlerEntry obtained by a previous lookup for a handler to lower
/// this intrinsic (see lookupIntrinsicHandler).
std::pair<fir::ExtendedValue, bool>
genIntrinsicCall(fir::FirOpBuilder &, aiir::Location,
                 const IntrinsicHandlerEntry &,
                 std::optional<aiir::Type> resultType,
                 llvm::ArrayRef<fir::ExtendedValue> args,
                 Fortran::lower::AbstractConverter *converter = nullptr);

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
  explicit IntrinsicLibrary(
      fir::FirOpBuilder &builder, aiir::Location loc,
      Fortran::lower::AbstractConverter *converter = nullptr)
      : builder{builder}, loc{loc}, converter{converter} {}
  IntrinsicLibrary() = delete;
  IntrinsicLibrary(const IntrinsicLibrary &) = delete;

  /// Generate FIR for call to Fortran intrinsic \p name with arguments \p arg
  /// and expected result type \p resultType. Return the result and a boolean
  /// that, if true, indicates that the result must be freed after use.
  std::pair<fir::ExtendedValue, bool>
  genIntrinsicCall(llvm::StringRef name, std::optional<aiir::Type> resultType,
                   llvm::ArrayRef<fir::ExtendedValue> arg);

  /// Search a runtime function that is associated to the generic intrinsic name
  /// and whose signature matches the intrinsic arguments and result types.
  /// If no such runtime function is found but a runtime function associated
  /// with the Fortran generic exists and has the same number of arguments,
  /// conversions will be inserted before and/or after the call. This is to
  /// mainly to allow 16 bits float support even-though little or no math
  /// runtime is currently available for it.
  aiir::Value genRuntimeCall(llvm::StringRef name, aiir::Type,
                             llvm::ArrayRef<aiir::Value>);

  using RuntimeCallGenerator = std::function<aiir::Value(
      fir::FirOpBuilder &, aiir::Location, llvm::ArrayRef<aiir::Value>)>;
  RuntimeCallGenerator
  getRuntimeCallGenerator(llvm::StringRef name,
                          aiir::FunctionType soughtFuncType);

  void genAbort(llvm::ArrayRef<fir::ExtendedValue>);
  /// Lowering for the ABS intrinsic. The ABS intrinsic expects one argument in
  /// the llvm::ArrayRef. The ABS intrinsic is lowered into AIIR/FIR operation
  /// if the argument is an integer, into llvm intrinsics if the argument is
  /// real and to the `hypot` math routine if the argument is of complex type.
  aiir::Value genAbs(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genAcosd(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genAcospi(aiir::Type, llvm::ArrayRef<aiir::Value>);
  template <void (*CallRuntime)(fir::FirOpBuilder &, aiir::Location loc,
                                aiir::Value, aiir::Value)>
  fir::ExtendedValue genAdjustRtCall(aiir::Type,
                                     llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genAimag(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genAint(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genAll(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genAllocated(aiir::Type,
                                  llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genAnint(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genAny(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genAtanpi(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue
      genCommandArgumentCount(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genAsind(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genAsinpi(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genAssociated(aiir::Type,
                                   llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genAtand(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genBesselJn(aiir::Type,
                                 llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genBesselYn(aiir::Type,
                                 llvm::ArrayRef<fir::ExtendedValue>);
  template <aiir::arith::CmpIPredicate pred>
  aiir::Value genBitwiseCompare(aiir::Type resultType,
                                llvm::ArrayRef<aiir::Value> args);

  aiir::Value genBtest(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genCeiling(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genChar(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genChdir(std::optional<aiir::Type> resultType,
                              llvm::ArrayRef<fir::ExtendedValue>);
  template <aiir::arith::CmpIPredicate pred>
  fir::ExtendedValue genCharacterCompare(aiir::Type,
                                         llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genCmplx(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genConjg(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genCount(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genCpuTime(llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genCshift(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genCAssociatedCFunPtr(aiir::Type,
                                           llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genCAssociatedCPtr(aiir::Type,
                                        llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genCDevLoc(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genErfcScaled(aiir::Type resultType,
                            llvm::ArrayRef<aiir::Value> args);
  void genCFPointer(llvm::ArrayRef<fir::ExtendedValue>);
  void genCFProcPointer(llvm::ArrayRef<fir::ExtendedValue>);
  void genCFStrPointer(llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genCFunLoc(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genCLoc(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  template <aiir::arith::CmpIPredicate pred>
  fir::ExtendedValue genCPtrCompare(aiir::Type,
                                    llvm::ArrayRef<fir::ExtendedValue>);
  void genCoBroadcast(llvm::ArrayRef<fir::ExtendedValue>);
  void genCoMax(llvm::ArrayRef<fir::ExtendedValue>);
  void genCoMin(llvm::ArrayRef<fir::ExtendedValue>);
  void genCoSum(llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genCosd(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genCospi(aiir::Type, llvm::ArrayRef<aiir::Value>);
  void genDateAndTime(llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genDsecnds(aiir::Type resultType,
                                llvm::ArrayRef<fir::ExtendedValue> args);
  aiir::Value genDim(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genDotProduct(aiir::Type,
                                   llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genDprod(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genDshiftl(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genDshiftr(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genEoshift(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genExit(llvm::ArrayRef<fir::ExtendedValue>);
  void genExecuteCommandLine(aiir::ArrayRef<fir::ExtendedValue> args);
  fir::ExtendedValue genEtime(std::optional<aiir::Type>,
                              aiir::ArrayRef<fir::ExtendedValue> args);
  aiir::Value genExponent(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genExtendsTypeOf(aiir::Type,
                                      llvm::ArrayRef<fir::ExtendedValue>);
  template <bool isMax>
  aiir::Value genExtremum(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genFCString(aiir::Type,
                                 llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genFloor(aiir::Type, llvm::ArrayRef<aiir::Value>);
  void genFlush(llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genFraction(aiir::Type resultType,
                          aiir::ArrayRef<aiir::Value> args);
  void genFree(aiir::ArrayRef<fir::ExtendedValue> args);
  fir::ExtendedValue genFseek(std::optional<aiir::Type>,
                              aiir::ArrayRef<fir::ExtendedValue> args);
  fir::ExtendedValue genFtell(std::optional<aiir::Type>,
                              aiir::ArrayRef<fir::ExtendedValue> args);
  fir::ExtendedValue genGetCwd(std::optional<aiir::Type> resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args);
  void genGetCommand(aiir::ArrayRef<fir::ExtendedValue> args);
  aiir::Value genGetPID(aiir::Type resultType,
                        llvm::ArrayRef<aiir::Value> args);
  void genGetCommandArgument(aiir::ArrayRef<fir::ExtendedValue> args);
  void genGetEnvironmentVariable(llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genGetGID(aiir::Type resultType,
                        llvm::ArrayRef<aiir::Value> args);
  aiir::Value genGetTeam(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genGetUID(aiir::Type resultType,
                        llvm::ArrayRef<aiir::Value> args);
  fir::ExtendedValue genHostnm(std::optional<aiir::Type> resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args);
  fir::ExtendedValue genIall(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genIand(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genIany(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genIbclr(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genIbits(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genIbset(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genIchar(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genFindloc(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genIeeeClass(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genIeeeCopySign(aiir::Type, llvm::ArrayRef<aiir::Value>);
  void genIeeeGetFlag(llvm::ArrayRef<fir::ExtendedValue>);
  void genIeeeGetHaltingMode(llvm::ArrayRef<fir::ExtendedValue>);
  template <bool isGet, bool isModes>
  void genIeeeGetOrSetModesOrStatus(llvm::ArrayRef<fir::ExtendedValue>);
  void genIeeeGetRoundingMode(llvm::ArrayRef<fir::ExtendedValue>);
  void genIeeeGetUnderflowMode(llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genIeeeInt(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genIeeeIsFinite(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genIeeeIsNan(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genIeeeIsNegative(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genIeeeIsNormal(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genIeeeLogb(aiir::Type, aiir::ArrayRef<aiir::Value>);
  template <bool isMax, bool isNum, bool isMag>
  aiir::Value genIeeeMaxMin(aiir::Type, llvm::ArrayRef<aiir::Value>);
  template <aiir::arith::CmpFPredicate pred>
  aiir::Value genIeeeQuietCompare(aiir::Type resultType,
                                  llvm::ArrayRef<aiir::Value>);
  aiir::Value genIeeeReal(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genIeeeRem(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genIeeeRint(aiir::Type, llvm::ArrayRef<aiir::Value>);
  template <bool isFlag>
  void genIeeeSetFlagOrHaltingMode(llvm::ArrayRef<fir::ExtendedValue>);
  void genIeeeSetRoundingMode(llvm::ArrayRef<fir::ExtendedValue>);
  void genIeeeSetUnderflowMode(llvm::ArrayRef<fir::ExtendedValue>);
  template <aiir::arith::CmpFPredicate pred>
  aiir::Value genIeeeSignalingCompare(aiir::Type resultType,
                                      llvm::ArrayRef<aiir::Value>);
  aiir::Value genIeeeSignbit(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genIeeeSupportFlag(aiir::Type,
                                        llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genIeeeSupportHalting(aiir::Type,
                                           llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genIeeeSupportRounding(aiir::Type,
                                            llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genIeeeSupportStandard(aiir::Type,
                                            llvm::ArrayRef<fir::ExtendedValue>);
  template <aiir::arith::CmpIPredicate pred>
  aiir::Value genIeeeTypeCompare(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genIeeeUnordered(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genIeeeValue(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genIeor(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genIndex(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genIor(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genIparity(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genIrand(aiir::Type resultType,
                              llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genIsContiguous(aiir::Type,
                                     llvm::ArrayRef<fir::ExtendedValue>);
  template <Fortran::runtime::io::Iostat value>
  aiir::Value genIsIostatValue(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genIsFPClass(aiir::Type, llvm::ArrayRef<aiir::Value>,
                           int fpclass);
  aiir::Value genIshft(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genIshftc(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genLbound(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genLeadz(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genLen(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genLenTrim(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genLoc(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genMalloc(aiir::Type, llvm::ArrayRef<aiir::Value>);
  template <typename Shift>
  aiir::Value genMask(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genMatmul(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMatmulTranspose(aiir::Type,
                                        llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMaxloc(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMaxval(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMerge(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genMergeBits(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genMinloc(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMinval(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genMod(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genModulo(aiir::Type, llvm::ArrayRef<aiir::Value>);
  void genMoveAlloc(llvm::ArrayRef<fir::ExtendedValue>);
  void genMvbits(llvm::ArrayRef<fir::ExtendedValue>);
  enum class NearestProc { Nearest, NextAfter, NextDown, NextUp };
  template <NearestProc>
  aiir::Value genNearest(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genNint(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genNorm2(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genNot(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genNull(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genNumImages(aiir::Type,
                                  llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genPack(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genParity(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genPerror(llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genPopcnt(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genPoppar(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genPresent(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genProduct(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genPutenv(std::optional<aiir::Type>,
                               llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genRand(aiir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue>);
  void genRandomInit(llvm::ArrayRef<fir::ExtendedValue>);
  void genRandomNumber(llvm::ArrayRef<fir::ExtendedValue>);
  void genRandomSeed(llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genReduce(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genReduceDim(aiir::Type,
                                  llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genRename(std::optional<aiir::Type>,
                               aiir::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genRepeat(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genReshape(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genRRSpacing(aiir::Type resultType,
                           llvm::ArrayRef<aiir::Value> args);
  aiir::Value genRtc(aiir::Type resultType, llvm::ArrayRef<aiir::Value> args);
  fir::ExtendedValue genSameTypeAs(aiir::Type,
                                   llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genScale(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genScan(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genSecnds(aiir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args);
  fir::ExtendedValue genSecond(std::optional<aiir::Type>,
                               aiir::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genSelectedCharKind(aiir::Type,
                                         llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genSelectedIntKind(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genSelectedLogicalKind(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genSelectedRealKind(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genSetExponent(aiir::Type resultType,
                             llvm::ArrayRef<aiir::Value> args);
  fir::ExtendedValue genShape(aiir::Type resultType,
                              llvm::ArrayRef<fir::ExtendedValue>);
  template <typename Shift>
  aiir::Value genShift(aiir::Type resultType, llvm::ArrayRef<aiir::Value>);
  aiir::Value genShiftA(aiir::Type resultType, llvm::ArrayRef<aiir::Value>);
  void genShowDescriptor(llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genSign(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genSind(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genSinpi(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genSize(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genSizeOf(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genSpacing(aiir::Type resultType,
                         llvm::ArrayRef<aiir::Value> args);
  void genSplit(llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genSpread(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genStorageSize(aiir::Type,
                                    llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genSum(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genSignalSubroutine(llvm::ArrayRef<fir::ExtendedValue>);
  void genSleep(llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genSystem(std::optional<aiir::Type>,
                               aiir::ArrayRef<fir::ExtendedValue> args);
  void genSystemClock(llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genTand(aiir::Type, llvm::ArrayRef<aiir::Value>);
  aiir::Value genTanpi(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genTeamNumber(aiir::Type,
                                   llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genTime(aiir::Type, llvm::ArrayRef<aiir::Value>);
  void genTokenize(llvm::ArrayRef<fir::ExtendedValue>);
  aiir::Value genTrailz(aiir::Type, llvm::ArrayRef<aiir::Value>);
  fir::ExtendedValue genTransfer(aiir::Type,
                                 llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genTranspose(aiir::Type,
                                  llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genThisImage(aiir::Type,
                                  llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genTrim(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genUbound(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genUnlink(std::optional<aiir::Type> resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args);
  fir::ExtendedValue genUnpack(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genVerify(aiir::Type, llvm::ArrayRef<fir::ExtendedValue>);

  /// Implement all conversion functions like DBLE, the first argument is
  /// the value to convert. There may be an additional KIND arguments that
  /// is ignored because this is already reflected in the result type.
  aiir::Value genConversion(aiir::Type, llvm::ArrayRef<aiir::Value>);

  /// In the template helper below:
  ///  - "FN func" is a callback to generate the related intrinsic runtime call.
  ///  - "FD funcDim" is a callback to generate the "dim" runtime call.
  ///  - "FC funcChar" is a callback to generate the character runtime call.
  /// Helper for MinLoc/MaxLoc.
  template <typename FN, typename FD>
  fir::ExtendedValue genExtremumloc(FN func, FD funcDim, llvm::StringRef errMsg,
                                    aiir::Type,
                                    llvm::ArrayRef<fir::ExtendedValue>);
  template <typename FN, typename FD, typename FC>
  /// Helper for MinVal/MaxVal.
  fir::ExtendedValue genExtremumVal(FN func, FD funcDim, FC funcChar,
                                    llvm::StringRef errMsg,
                                    aiir::Type resultType,
                                    llvm::ArrayRef<fir::ExtendedValue> args);
  /// Process calls to Product, Sum, IAll, IAny, IParity intrinsic functions
  template <typename FN, typename FD>
  fir::ExtendedValue genReduction(FN func, FD funcDim, llvm::StringRef errMsg,
                                  aiir::Type resultType,
                                  llvm::ArrayRef<fir::ExtendedValue> args);

  /// Generate code to raise \p excepts if \p cond is absent,
  /// or present and true.
  void genRaiseExcept(int excepts, aiir::Value cond = {});

  /// Generate a quiet NaN of a given floating point type.
  aiir::Value genQNan(aiir::Type resultType);

  /// Define the different FIR generators that can be mapped to intrinsic to
  /// generate the related code.
  using ElementalGenerator = decltype(&IntrinsicLibrary::genAbs);
  using ExtendedGenerator = decltype(&IntrinsicLibrary::genLenTrim);
  using SubroutineGenerator = decltype(&IntrinsicLibrary::genDateAndTime);
  /// The generator for intrinsic that has both function and subroutine form.
  using DualGenerator = decltype(&IntrinsicLibrary::genEtime);
  using Generator = std::variant<ElementalGenerator, ExtendedGenerator,
                                 SubroutineGenerator, DualGenerator>;

  /// All generators can be outlined. This will build a function named
  /// "fir."+ <generic name> + "." + <result type code> and generate the
  /// intrinsic implementation inside instead of at the intrinsic call sites.
  /// This can be used to keep the FIR more readable. Only one function will
  /// be generated for all the similar calls in a program.
  /// If the Generator is nullptr, the wrapper uses genRuntimeCall.
  template <typename GeneratorType>
  aiir::Value outlineInWrapper(GeneratorType, llvm::StringRef name,
                               aiir::Type resultType,
                               llvm::ArrayRef<aiir::Value> args);
  template <typename GeneratorType>
  fir::ExtendedValue
  outlineInExtendedWrapper(GeneratorType, llvm::StringRef name,
                           std::optional<aiir::Type> resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args);

  template <typename GeneratorType>
  aiir::func::FuncOp getWrapper(GeneratorType, llvm::StringRef name,
                                aiir::FunctionType,
                                bool loadRefArguments = false);

  /// Generate calls to ElementalGenerator, handling the elemental aspects
  template <typename GeneratorType>
  fir::ExtendedValue
  genElementalCall(GeneratorType, llvm::StringRef name, aiir::Type resultType,
                   llvm::ArrayRef<fir::ExtendedValue> args, bool outline);

  /// Helper to invoke code generator for the intrinsics given arguments.
  aiir::Value invokeGenerator(ElementalGenerator generator,
                              aiir::Type resultType,
                              llvm::ArrayRef<aiir::Value> args);
  aiir::Value invokeGenerator(RuntimeCallGenerator generator,
                              aiir::Type resultType,
                              llvm::ArrayRef<aiir::Value> args);
  aiir::Value invokeGenerator(ExtendedGenerator generator,
                              aiir::Type resultType,
                              llvm::ArrayRef<aiir::Value> args);
  aiir::Value invokeGenerator(SubroutineGenerator generator,
                              llvm::ArrayRef<aiir::Value> args);
  aiir::Value invokeGenerator(DualGenerator generator,
                              llvm::ArrayRef<aiir::Value> args);
  aiir::Value invokeGenerator(DualGenerator generator, aiir::Type resultType,
                              llvm::ArrayRef<aiir::Value> args);

  /// Get pointer to unrestricted intrinsic. Generate the related unrestricted
  /// intrinsic if it is not defined yet.
  aiir::SymbolRefAttr
  getUnrestrictedIntrinsicSymbolRefAttr(llvm::StringRef name,
                                        aiir::FunctionType signature);

  /// Helper function for generating code clean-up for result descriptors
  fir::ExtendedValue readAndAddCleanUp(fir::MutableBoxValue resultMutableBox,
                                       aiir::Type resultType,
                                       llvm::StringRef errMsg);

  void setResultMustBeFreed() { resultMustBeFreed = true; }

  fir::FirOpBuilder &builder;
  aiir::Location loc;
  bool resultMustBeFreed = false;
  Fortran::lower::AbstractConverter *converter = nullptr;
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

struct MathOperation {
  // Callback type for generating lowering for a math operation.
  using MathGeneratorTy = aiir::Value (*)(fir::FirOpBuilder &, aiir::Location,
                                          const MathOperation &,
                                          aiir::FunctionType,
                                          llvm::ArrayRef<aiir::Value>);

  // Overrides fir::runtime::FuncTypeBuilderFunc to add FirOpBuilder argument.
  using FuncTypeBuilderFunc = aiir::FunctionType (*)(aiir::AIIRContext *,
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
  // A callback may generate either dedicated AIIR operation(s) or
  // a function call to a runtime function with name defined by
  // 'runtimeFunc'.
  MathGeneratorTy funcGenerator;
};

// Enum of most supported intrinsic argument or return types.
enum class ParamTypeId {
  Void,
  Address, // pointer (to an [array of] Integers of some kind)
  Integer,
  Real,
  Complex,
  IntegerVector,
  UnsignedVector,
  RealVector,
};

// Helper function to get length of a 16-byte vector of element type eleTy.
static int getVecLen(aiir::Type eleTy) {
  assert((aiir::isa<aiir::IntegerType>(eleTy) ||
          aiir::isa<aiir::FloatType>(eleTy)) &&
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
using Address = ParamType<ParamTypeId::Address, k>;
template <int k>
using Integer = ParamType<ParamTypeId::Integer, k>;
template <int k>
using Real = ParamType<ParamTypeId::Real, k>;
template <int k>
using Complex = ParamType<ParamTypeId::Complex, k>;
template <int k>
using IntegerVector = ParamType<ParamTypeId::IntegerVector, k>;
template <int k>
using UnsignedVector = ParamType<ParamTypeId::UnsignedVector, k>;
template <int k>
using RealVector = ParamType<ParamTypeId::RealVector, k>;
} // namespace Ty

// Helper function that generates most types that are supported for intrinsic
// arguments and return type. Used by `genFuncType` to generate function
// types for most of the intrinsics.
static inline aiir::Type getTypeHelper(aiir::AIIRContext *context,
                                       fir::FirOpBuilder &builder,
                                       ParamTypeId typeId, int kind) {
  aiir::Type r;
  unsigned bits{0};
  switch (typeId) {
  case ParamTypeId::Void:
    llvm::report_fatal_error("can not get type of void");
    break;
  case ParamTypeId::Address:
    bits = builder.getKindMap().getIntegerBitsize(kind);
    assert(bits != 0 && "failed to convert address kind to integer bitsize");
    r = fir::ReferenceType::get(aiir::IntegerType::get(context, bits));
    break;
  case ParamTypeId::Integer:
  case ParamTypeId::IntegerVector:
    bits = builder.getKindMap().getIntegerBitsize(kind);
    assert(bits != 0 && "failed to convert kind to integer bitsize");
    r = aiir::IntegerType::get(context, bits);
    break;
  case ParamTypeId::UnsignedVector:
    bits = builder.getKindMap().getIntegerBitsize(kind);
    assert(bits != 0 && "failed to convert kind to unsigned bitsize");
    r = aiir::IntegerType::get(context, bits, aiir::IntegerType::Unsigned);
    break;
  case ParamTypeId::Real:
  case ParamTypeId::RealVector:
    r = builder.getRealType(kind);
    break;
  case ParamTypeId::Complex:
    r = aiir::ComplexType::get(builder.getRealType(kind));
    break;
  }

  switch (typeId) {
  case ParamTypeId::Void:
  case ParamTypeId::Address:
  case ParamTypeId::Integer:
  case ParamTypeId::Real:
  case ParamTypeId::Complex:
    break;
  case ParamTypeId::IntegerVector:
  case ParamTypeId::UnsignedVector:
  case ParamTypeId::RealVector:
    // convert to vector type
    r = fir::VectorType::get(getVecLen(r), r);
  }
  return r;
}

// Generic function type generator that supports most of the function types
// used by intrinsics.
template <typename TyR, typename... ArgTys>
static inline aiir::FunctionType genFuncType(aiir::AIIRContext *context,
                                             fir::FirOpBuilder &builder) {
  llvm::SmallVector<ParamTypeId> argTys = {ArgTys::ty...};
  llvm::SmallVector<int> argKinds = {ArgTys::kind...};
  llvm::SmallVector<aiir::Type> argTypes;

  for (size_t i = 0; i < argTys.size(); ++i) {
    argTypes.push_back(getTypeHelper(context, builder, argTys[i], argKinds[i]));
  }

  if (TyR::ty == ParamTypeId::Void)
    return aiir::FunctionType::get(context, argTypes, {});

  auto resType = getTypeHelper(context, builder, TyR::ty, TyR::kind);
  return aiir::FunctionType::get(context, argTypes, {resType});
}

/// Entry into the tables describing how an intrinsic must be lowered.
struct IntrinsicHandlerEntry {
  using RuntimeGeneratorRange =
      std::pair<const MathOperation *, const MathOperation *>;
  IntrinsicHandlerEntry(const IntrinsicHandler *handler) : entry{handler} {
    assert(handler && "handler must not be nullptr");
  };
  IntrinsicHandlerEntry(RuntimeGeneratorRange rt) : entry{rt} {};
  const IntrinsicArgumentLoweringRules *getArgumentLoweringRules() const;
  std::variant<const IntrinsicHandler *, RuntimeGeneratorRange> entry;
};

//===----------------------------------------------------------------------===//
// Helper functions for argument handling.
//===----------------------------------------------------------------------===//
static inline aiir::Type getConvertedElementType(aiir::AIIRContext *context,
                                                 aiir::Type eleTy) {
  if (aiir::isa<aiir::IntegerType>(eleTy) && !eleTy.isSignlessInteger()) {
    const auto intTy{aiir::dyn_cast<aiir::IntegerType>(eleTy)};
    auto newEleTy{aiir::IntegerType::get(context, intTy.getWidth())};
    return newEleTy;
  }
  return eleTy;
}

static inline llvm::SmallVector<aiir::Value, 4>
getBasesForArgs(llvm::ArrayRef<fir::ExtendedValue> args) {
  llvm::SmallVector<aiir::Value, 4> baseVec;
  for (auto arg : args)
    baseVec.push_back(getBase(arg));
  return baseVec;
}

static inline llvm::SmallVector<aiir::Type, 4>
getTypesForArgs(llvm::ArrayRef<aiir::Value> args) {
  llvm::SmallVector<aiir::Type, 4> typeVec;
  for (auto arg : args)
    typeVec.push_back(arg.getType());
  return typeVec;
}

aiir::Value genLibCall(fir::FirOpBuilder &builder, aiir::Location loc,
                       const MathOperation &mathOp,
                       aiir::FunctionType libFuncType,
                       llvm::ArrayRef<aiir::Value> args);

template <typename T>
aiir::Value genMathOp(fir::FirOpBuilder &builder, aiir::Location loc,
                      const MathOperation &mathOp,
                      aiir::FunctionType mathLibFuncType,
                      llvm::ArrayRef<aiir::Value> args);

template <typename T>
aiir::Value genComplexMathOp(fir::FirOpBuilder &builder, aiir::Location loc,
                             const MathOperation &mathOp,
                             aiir::FunctionType mathLibFuncType,
                             llvm::ArrayRef<aiir::Value> args);

aiir::Value genLibSplitComplexArgsCall(fir::FirOpBuilder &builder,
                                       aiir::Location loc,
                                       const MathOperation &mathOp,
                                       aiir::FunctionType libFuncType,
                                       llvm::ArrayRef<aiir::Value> args);

/// Lookup for a handler or runtime call generator to lower intrinsic
/// \p intrinsicName.
std::optional<IntrinsicHandlerEntry>
lookupIntrinsicHandler(fir::FirOpBuilder &, llvm::StringRef intrinsicName,
                       std::optional<aiir::Type> resultType);

/// Generate a TODO error message for an as yet unimplemented intrinsic.
void crashOnMissingIntrinsic(aiir::Location loc, llvm::StringRef name);

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
aiir::SymbolRefAttr
getUnrestrictedIntrinsicSymbolRefAttr(fir::FirOpBuilder &, aiir::Location,
                                      llvm::StringRef name,
                                      aiir::FunctionType signature);

//===----------------------------------------------------------------------===//
// Direct access to intrinsics that may be used by lowering outside
// of intrinsic call lowering.
//===----------------------------------------------------------------------===//

/// Generate maximum. There must be at least one argument and all arguments
/// must have the same type.
aiir::Value genMax(fir::FirOpBuilder &, aiir::Location,
                   llvm::ArrayRef<aiir::Value> args);

/// Generate minimum. Same constraints as genMax.
aiir::Value genMin(fir::FirOpBuilder &, aiir::Location,
                   llvm::ArrayRef<aiir::Value> args);

/// Generate Complex divide with the given expected
/// result type.
aiir::Value genDivC(fir::FirOpBuilder &builder, aiir::Location loc,
                    aiir::Type resultType, aiir::Value x, aiir::Value y);

/// Generate power function x**y with the given expected
/// result type.
aiir::Value genPow(fir::FirOpBuilder &, aiir::Location, aiir::Type resultType,
                   aiir::Value x, aiir::Value y);

} // namespace fir

#endif // FORTRAN_LOWER_INTRINSICCALL_H
