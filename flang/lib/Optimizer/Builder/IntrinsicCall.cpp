//===-- IntrinsicCall.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper routines for constructing the FIR dialect of AIIR. As FIR is a
// dialect of AIIR, it makes extensive use of AIIR interfaces and AIIR's coding
// style (https://aiir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Common/static-multimap-view.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/CUDAIntrinsicCall.h"
#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/PPCIntrinsicCall.h"
#include "flang/Optimizer/Builder/Runtime/Allocatable.h"
#include "flang/Optimizer/Builder/Runtime/CUDA/Descriptor.h"
#include "flang/Optimizer/Builder/Runtime/Character.h"
#include "flang/Optimizer/Builder/Runtime/Command.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Runtime/Exceptions.h"
#include "flang/Optimizer/Builder/Runtime/Execute.h"
#include "flang/Optimizer/Builder/Runtime/Inquiry.h"
#include "flang/Optimizer/Builder/Runtime/Intrinsics.h"
#include "flang/Optimizer/Builder/Runtime/Numeric.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Reduction.h"
#include "flang/Optimizer/Builder/Runtime/Stop.h"
#include "flang/Optimizer/Builder/Runtime/Transformational.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/MIF/MIFOps.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Runtime/entry-names.h"
#include "flang/Runtime/iostat-consts.h"
#include "aiir/Dialect/Complex/IR/Complex.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/LLVMIR/LLVMTypes.h"
#include "aiir/Dialect/Math/IR/Math.h"
#include "aiir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cfenv> // temporary -- only used in genIeeeGetOrSetModesOrStatus
#include <optional>

#define DEBUG_TYPE "flang-lower-intrinsic"

/// This file implements lowering of Fortran intrinsic procedures and Fortran
/// intrinsic module procedures.  A call may be inlined with a mix of FIR and
/// AIIR operations, or as a call to a runtime function or LLVM intrinsic.

/// Lowering of intrinsic procedure calls is based on a map that associates
/// Fortran intrinsic generic names to FIR generator functions.
/// All generator functions are member functions of the IntrinsicLibrary class
/// and have the same interface.
/// If no generator is given for an intrinsic name, a math runtime library
/// is searched for an implementation and, if a runtime function is found,
/// a call is generated for it. LLVM intrinsics are handled as a math
/// runtime library here.

namespace fir {

fir::ExtendedValue getAbsentIntrinsicArgument() { return fir::UnboxedValue{}; }

/// Test if an ExtendedValue is absent. This is used to test if an intrinsic
/// argument are absent at compile time.
static bool isStaticallyAbsent(const fir::ExtendedValue &exv) {
  return !fir::getBase(exv);
}
static bool isStaticallyAbsent(llvm::ArrayRef<fir::ExtendedValue> args,
                               size_t argIndex) {
  return args.size() <= argIndex || isStaticallyAbsent(args[argIndex]);
}
static bool isStaticallyAbsent(llvm::ArrayRef<aiir::Value> args,
                               size_t argIndex) {
  return args.size() <= argIndex || !args[argIndex];
}
static bool isOptional(aiir::Value value) {
  auto varIface = aiir::dyn_cast_or_null<fir::FortranVariableOpInterface>(
      value.getDefiningOp());
  return varIface && varIface.isOptional();
}

/// Test if an ExtendedValue is present. This is used to test if an intrinsic
/// argument is present at compile time. This does not imply that the related
/// value may not be an absent dummy optional, disassociated pointer, or a
/// deallocated allocatable. See `handleDynamicOptional` to deal with these
/// cases when it makes sense.
static bool isStaticallyPresent(const fir::ExtendedValue &exv) {
  return !isStaticallyAbsent(exv);
}

using I = IntrinsicLibrary;

/// Flag to indicate that an intrinsic argument has to be handled as
/// being dynamically optional (e.g. special handling when actual
/// argument is an optional variable in the current scope).
static constexpr bool handleDynamicOptional = true;

/// Table that drives the fir generation depending on the intrinsic or intrinsic
/// module procedure one to one mapping with Fortran arguments. If no mapping is
/// defined here for a generic intrinsic, genRuntimeCall will be called
/// to look for a match in the runtime and emit a call. Note that the argument
/// lowering rules for an intrinsic need to be provided only if at least one
/// argument must not be lowered by value. In which case, the lowering rules
/// should be provided for all the intrinsic arguments for completeness.
static constexpr IntrinsicHandler handlers[]{
    {"abort", &I::genAbort},
    {"abs", &I::genAbs},
    {"achar", &I::genChar},
    {"acosd", &I::genAcosd},
    {"acospi", &I::genAcospi},
    {"adjustl",
     &I::genAdjustRtCall<fir::runtime::genAdjustL>,
     {{{"string", asAddr}}},
     /*isElemental=*/true},
    {"adjustr",
     &I::genAdjustRtCall<fir::runtime::genAdjustR>,
     {{{"string", asAddr}}},
     /*isElemental=*/true},
    {"aimag", &I::genAimag},
    {"aint", &I::genAint},
    {"all",
     &I::genAll,
     {{{"mask", asAddr}, {"dim", asValue}}},
     /*isElemental=*/false},
    {"allocated",
     &I::genAllocated,
     {{{"array", asInquired}, {"scalar", asInquired}}},
     /*isElemental=*/false},
    {"anint", &I::genAnint},
    {"any",
     &I::genAny,
     {{{"mask", asAddr}, {"dim", asValue}}},
     /*isElemental=*/false},
    {"asind", &I::genAsind},
    {"asinpi", &I::genAsinpi},
    {"associated",
     &I::genAssociated,
     {{{"pointer", asInquired}, {"target", asInquired}}},
     /*isElemental=*/false},
    {"atan2d", &I::genAtand},
    {"atan2pi", &I::genAtanpi},
    {"atand", &I::genAtand},
    {"atanpi", &I::genAtanpi},
    {"bessel_jn",
     &I::genBesselJn,
     {{{"n1", asValue}, {"n2", asValue}, {"x", asValue}}},
     /*isElemental=*/false},
    {"bessel_yn",
     &I::genBesselYn,
     {{{"n1", asValue}, {"n2", asValue}, {"x", asValue}}},
     /*isElemental=*/false},
    {"bge", &I::genBitwiseCompare<aiir::arith::CmpIPredicate::uge>},
    {"bgt", &I::genBitwiseCompare<aiir::arith::CmpIPredicate::ugt>},
    {"ble", &I::genBitwiseCompare<aiir::arith::CmpIPredicate::ule>},
    {"blt", &I::genBitwiseCompare<aiir::arith::CmpIPredicate::ult>},
    {"btest", &I::genBtest},
    {"c_associated_c_funptr",
     &I::genCAssociatedCFunPtr,
     {{{"c_ptr_1", asAddr}, {"c_ptr_2", asAddr, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"c_associated_c_ptr",
     &I::genCAssociatedCPtr,
     {{{"c_ptr_1", asAddr}, {"c_ptr_2", asAddr, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"c_devloc", &I::genCDevLoc, {{{"x", asBox}}}, /*isElemental=*/false},
    {"c_f_pointer",
     &I::genCFPointer,
     {{{"cptr", asValue},
       {"fptr", asInquired},
       {"shape", asAddr, handleDynamicOptional},
       {"lower", asAddr, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"c_f_procpointer",
     &I::genCFProcPointer,
     {{{"cptr", asValue}, {"fptr", asInquired}}},
     /*isElemental=*/false},
    {"c_f_strpointer",
     &I::genCFStrPointer,
     {{{"cstrptr_or_cstrarray", asValue},
       {"fstrptr", asInquired},
       {"nchars", asValue, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"c_funloc", &I::genCFunLoc, {{{"x", asBox}}}, /*isElemental=*/false},
    {"c_loc", &I::genCLoc, {{{"x", asBox}}}, /*isElemental=*/false},
    {"c_ptr_eq", &I::genCPtrCompare<aiir::arith::CmpIPredicate::eq>},
    {"c_ptr_ne", &I::genCPtrCompare<aiir::arith::CmpIPredicate::ne>},
    {"ceiling", &I::genCeiling},
    {"char", &I::genChar},
    {"chdir",
     &I::genChdir,
     {{{"name", asAddr}, {"status", asAddr, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"cmplx",
     &I::genCmplx,
     {{{"x", asValue}, {"y", asValue, handleDynamicOptional}}}},
    {"co_broadcast",
     &I::genCoBroadcast,
     {{{"a", asBox},
       {"source_image", asValue},
       {"stat", asAddr, handleDynamicOptional},
       {"errmsg", asBox, handleDynamicOptional}}},
     /*isElemental*/ false},
    {"co_max",
     &I::genCoMax,
     {{{"a", asBox},
       {"result_image", asValue, handleDynamicOptional},
       {"stat", asAddr, handleDynamicOptional},
       {"errmsg", asBox, handleDynamicOptional}}},
     /*isElemental*/ false},
    {"co_min",
     &I::genCoMin,
     {{{"a", asBox},
       {"result_image", asValue, handleDynamicOptional},
       {"stat", asAddr, handleDynamicOptional},
       {"errmsg", asBox, handleDynamicOptional}}},
     /*isElemental*/ false},
    {"co_sum",
     &I::genCoSum,
     {{{"a", asBox},
       {"result_image", asValue, handleDynamicOptional},
       {"stat", asAddr, handleDynamicOptional},
       {"errmsg", asBox, handleDynamicOptional}}},
     /*isElemental*/ false},
    {"command_argument_count", &I::genCommandArgumentCount},
    {"conjg", &I::genConjg},
    {"cosd", &I::genCosd},
    {"cospi", &I::genCospi},
    {"count",
     &I::genCount,
     {{{"mask", asAddr}, {"dim", asValue}, {"kind", asValue}}},
     /*isElemental=*/false},
    {"cpu_time",
     &I::genCpuTime,
     {{{"time", asAddr}}},
     /*isElemental=*/false},
    {"cshift",
     &I::genCshift,
     {{{"array", asAddr},
       {"shift", asAddr},
       {"dim", asValue, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"date_and_time",
     &I::genDateAndTime,
     {{{"date", asAddr, handleDynamicOptional},
       {"time", asAddr, handleDynamicOptional},
       {"zone", asAddr, handleDynamicOptional},
       {"values", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"dble", &I::genConversion},
    {"dim", &I::genDim},
    {"dot_product",
     &I::genDotProduct,
     {{{"vector_a", asBox}, {"vector_b", asBox}}},
     /*isElemental=*/false},
    {"dprod", &I::genDprod},
    {"dsecnds",
     &I::genDsecnds,
     {{{"refTime", asAddr}}},
     /*isElemental=*/false},
    {"dshiftl", &I::genDshiftl},
    {"dshiftr", &I::genDshiftr},
    {"eoshift",
     &I::genEoshift,
     {{{"array", asBox},
       {"shift", asAddr},
       {"boundary", asBox, handleDynamicOptional},
       {"dim", asValue, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"erfc_scaled", &I::genErfcScaled},
    {"etime",
     &I::genEtime,
     {{{"values", asBox}, {"time", asBox}}},
     /*isElemental=*/false},
    {"execute_command_line",
     &I::genExecuteCommandLine,
     {{{"command", asBox},
       {"wait", asAddr, handleDynamicOptional},
       {"exitstat", asBox, handleDynamicOptional},
       {"cmdstat", asBox, handleDynamicOptional},
       {"cmdmsg", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"exit",
     &I::genExit,
     {{{"status", asValue, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"exponent", &I::genExponent},
    {"extends_type_of",
     &I::genExtendsTypeOf,
     {{{"a", asBox}, {"mold", asBox}}},
     /*isElemental=*/false},
    {"f_c_string",
     &I::genFCString,
     {{{"string", asAddr}, {"asis", asValue, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"findloc",
     &I::genFindloc,
     {{{"array", asBox},
       {"value", asAddr},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional},
       {"kind", asValue},
       {"back", asValue, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"floor", &I::genFloor},
    {"flush",
     &I::genFlush,
     {{{"unit", asAddr}}},
     /*isElemental=*/false},
    {"fraction", &I::genFraction},
    {"free", &I::genFree},
    {"fseek",
     &I::genFseek,
     {{{"unit", asValue},
       {"offset", asValue},
       {"whence", asValue},
       {"status", asAddr, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"ftell",
     &I::genFtell,
     {{{"unit", asValue}, {"offset", asAddr}}},
     /*isElemental=*/false},
    {"get_command",
     &I::genGetCommand,
     {{{"command", asBox, handleDynamicOptional},
       {"length", asBox, handleDynamicOptional},
       {"status", asAddr, handleDynamicOptional},
       {"errmsg", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"get_command_argument",
     &I::genGetCommandArgument,
     {{{"number", asValue},
       {"value", asBox, handleDynamicOptional},
       {"length", asBox, handleDynamicOptional},
       {"status", asAddr, handleDynamicOptional},
       {"errmsg", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"get_environment_variable",
     &I::genGetEnvironmentVariable,
     {{{"name", asBox},
       {"value", asBox, handleDynamicOptional},
       {"length", asBox, handleDynamicOptional},
       {"status", asAddr, handleDynamicOptional},
       {"trim_name", asAddr, handleDynamicOptional},
       {"errmsg", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"get_team",
     &I::genGetTeam,
     {{{"level", asValue, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"getcwd",
     &I::genGetCwd,
     {{{"c", asBox}, {"status", asAddr, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"getgid", &I::genGetGID},
    {"getpid", &I::genGetPID},
    {"getuid", &I::genGetUID},
    {"hostnm",
     &I::genHostnm,
     {{{"c", asBox}, {"status", asAddr, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"iachar", &I::genIchar},
    {"iall",
     &I::genIall,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"iand", &I::genIand},
    {"iany",
     &I::genIany,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"ibclr", &I::genIbclr},
    {"ibits", &I::genIbits},
    {"ibset", &I::genIbset},
    {"ichar", &I::genIchar},
    {"ieee_class", &I::genIeeeClass},
    {"ieee_class_eq", &I::genIeeeTypeCompare<aiir::arith::CmpIPredicate::eq>},
    {"ieee_class_ne", &I::genIeeeTypeCompare<aiir::arith::CmpIPredicate::ne>},
    {"ieee_copy_sign", &I::genIeeeCopySign},
    {"ieee_get_flag",
     &I::genIeeeGetFlag,
     {{{"flag", asValue}, {"flag_value", asAddr}}}},
    {"ieee_get_halting_mode",
     &I::genIeeeGetHaltingMode,
     {{{"flag", asValue}, {"halting", asAddr}}}},
    {"ieee_get_modes",
     &I::genIeeeGetOrSetModesOrStatus</*isGet=*/true, /*isModes=*/true>},
    {"ieee_get_rounding_mode",
     &I::genIeeeGetRoundingMode,
     {{{"round_value", asAddr, handleDynamicOptional},
       {"radix", asValue, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"ieee_get_status",
     &I::genIeeeGetOrSetModesOrStatus</*isGet=*/true, /*isModes=*/false>},
    {"ieee_get_underflow_mode",
     &I::genIeeeGetUnderflowMode,
     {{{"gradual", asAddr}}},
     /*isElemental=*/false},
    {"ieee_int", &I::genIeeeInt},
    {"ieee_is_finite", &I::genIeeeIsFinite},
    {"ieee_is_nan", &I::genIeeeIsNan},
    {"ieee_is_negative", &I::genIeeeIsNegative},
    {"ieee_is_normal", &I::genIeeeIsNormal},
    {"ieee_logb", &I::genIeeeLogb},
    {"ieee_max",
     &I::genIeeeMaxMin</*isMax=*/true, /*isNum=*/false, /*isMag=*/false>},
    {"ieee_max_mag",
     &I::genIeeeMaxMin</*isMax=*/true, /*isNum=*/false, /*isMag=*/true>},
    {"ieee_max_num",
     &I::genIeeeMaxMin</*isMax=*/true, /*isNum=*/true, /*isMag=*/false>},
    {"ieee_max_num_mag",
     &I::genIeeeMaxMin</*isMax=*/true, /*isNum=*/true, /*isMag=*/true>},
    {"ieee_min",
     &I::genIeeeMaxMin</*isMax=*/false, /*isNum=*/false, /*isMag=*/false>},
    {"ieee_min_mag",
     &I::genIeeeMaxMin</*isMax=*/false, /*isNum=*/false, /*isMag=*/true>},
    {"ieee_min_num",
     &I::genIeeeMaxMin</*isMax=*/false, /*isNum=*/true, /*isMag=*/false>},
    {"ieee_min_num_mag",
     &I::genIeeeMaxMin</*isMax=*/false, /*isNum=*/true, /*isMag=*/true>},
    {"ieee_next_after", &I::genNearest<I::NearestProc::NextAfter>},
    {"ieee_next_down", &I::genNearest<I::NearestProc::NextDown>},
    {"ieee_next_up", &I::genNearest<I::NearestProc::NextUp>},
    {"ieee_quiet_eq", &I::genIeeeQuietCompare<aiir::arith::CmpFPredicate::OEQ>},
    {"ieee_quiet_ge", &I::genIeeeQuietCompare<aiir::arith::CmpFPredicate::OGE>},
    {"ieee_quiet_gt", &I::genIeeeQuietCompare<aiir::arith::CmpFPredicate::OGT>},
    {"ieee_quiet_le", &I::genIeeeQuietCompare<aiir::arith::CmpFPredicate::OLE>},
    {"ieee_quiet_lt", &I::genIeeeQuietCompare<aiir::arith::CmpFPredicate::OLT>},
    {"ieee_quiet_ne", &I::genIeeeQuietCompare<aiir::arith::CmpFPredicate::UNE>},
    {"ieee_real", &I::genIeeeReal},
    {"ieee_rem", &I::genIeeeRem},
    {"ieee_rint", &I::genIeeeRint},
    {"ieee_round_eq", &I::genIeeeTypeCompare<aiir::arith::CmpIPredicate::eq>},
    {"ieee_round_ne", &I::genIeeeTypeCompare<aiir::arith::CmpIPredicate::ne>},
    {"ieee_set_flag", &I::genIeeeSetFlagOrHaltingMode</*isFlag=*/true>},
    {"ieee_set_halting_mode",
     &I::genIeeeSetFlagOrHaltingMode</*isFlag=*/false>},
    {"ieee_set_modes",
     &I::genIeeeGetOrSetModesOrStatus</*isGet=*/false, /*isModes=*/true>},
    {"ieee_set_rounding_mode",
     &I::genIeeeSetRoundingMode,
     {{{"round_value", asValue, handleDynamicOptional},
       {"radix", asValue, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"ieee_set_status",
     &I::genIeeeGetOrSetModesOrStatus</*isGet=*/false, /*isModes=*/false>},
    {"ieee_set_underflow_mode", &I::genIeeeSetUnderflowMode},
    {"ieee_signaling_eq",
     &I::genIeeeSignalingCompare<aiir::arith::CmpFPredicate::OEQ>},
    {"ieee_signaling_ge",
     &I::genIeeeSignalingCompare<aiir::arith::CmpFPredicate::OGE>},
    {"ieee_signaling_gt",
     &I::genIeeeSignalingCompare<aiir::arith::CmpFPredicate::OGT>},
    {"ieee_signaling_le",
     &I::genIeeeSignalingCompare<aiir::arith::CmpFPredicate::OLE>},
    {"ieee_signaling_lt",
     &I::genIeeeSignalingCompare<aiir::arith::CmpFPredicate::OLT>},
    {"ieee_signaling_ne",
     &I::genIeeeSignalingCompare<aiir::arith::CmpFPredicate::UNE>},
    {"ieee_signbit", &I::genIeeeSignbit},
    {"ieee_support_flag",
     &I::genIeeeSupportFlag,
     {{{"flag", asValue}, {"x", asInquired, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"ieee_support_halting",
     &I::genIeeeSupportHalting,
     {{{"flag", asValue}}},
     /*isElemental=*/false},
    {"ieee_support_rounding",
     &I::genIeeeSupportRounding,
     {{{"round_value", asValue}, {"x", asInquired, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"ieee_support_standard",
     &I::genIeeeSupportStandard,
     {{{"flag", asValue}, {"x", asInquired, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"ieee_unordered", &I::genIeeeUnordered},
    {"ieee_value", &I::genIeeeValue},
    {"ieor", &I::genIeor},
    {"index",
     &I::genIndex,
     {{{"string", asAddr},
       {"substring", asAddr},
       {"back", asValue, handleDynamicOptional},
       {"kind", asValue}}}},
    {"ior", &I::genIor},
    {"iparity",
     &I::genIparity,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"irand",
     &I::genIrand,
     {{{"i", asAddr, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"is_contiguous",
     &I::genIsContiguous,
     {{{"array", asBox}}},
     /*isElemental=*/false},
    {"is_iostat_end", &I::genIsIostatValue<Fortran::runtime::io::IostatEnd>},
    {"is_iostat_eor", &I::genIsIostatValue<Fortran::runtime::io::IostatEor>},
    {"ishft", &I::genIshft},
    {"ishftc", &I::genIshftc},
    {"isnan", &I::genIeeeIsNan},
    {"lbound",
     &I::genLbound,
     {{{"array", asInquired}, {"dim", asValue}, {"kind", asValue}}},
     /*isElemental=*/false},
    {"leadz", &I::genLeadz},
    {"len",
     &I::genLen,
     {{{"string", asInquired}, {"kind", asValue}}},
     /*isElemental=*/false},
    {"len_trim", &I::genLenTrim},
    {"lge", &I::genCharacterCompare<aiir::arith::CmpIPredicate::sge>},
    {"lgt", &I::genCharacterCompare<aiir::arith::CmpIPredicate::sgt>},
    {"lle", &I::genCharacterCompare<aiir::arith::CmpIPredicate::sle>},
    {"llt", &I::genCharacterCompare<aiir::arith::CmpIPredicate::slt>},
    {"lnblnk", &I::genLenTrim},
    {"loc", &I::genLoc, {{{"x", asBox}}}, /*isElemental=*/false},
    {"malloc", &I::genMalloc},
    {"maskl", &I::genMask<aiir::arith::ShLIOp>},
    {"maskr", &I::genMask<aiir::arith::ShRUIOp>},
    {"matmul",
     &I::genMatmul,
     {{{"matrix_a", asAddr}, {"matrix_b", asAddr}}},
     /*isElemental=*/false},
    {"matmul_transpose",
     &I::genMatmulTranspose,
     {{{"matrix_a", asAddr}, {"matrix_b", asAddr}}},
     /*isElemental=*/false},
    {"max", &I::genExtremum</*isMax=*/true>},
    {"maxloc",
     &I::genMaxloc,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional},
       {"kind", asValue},
       {"back", asValue, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"maxval",
     &I::genMaxval,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"merge", &I::genMerge},
    {"merge_bits", &I::genMergeBits},
    {"min", &I::genExtremum</*isMax=*/false>},
    {"minloc",
     &I::genMinloc,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional},
       {"kind", asValue},
       {"back", asValue, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"minval",
     &I::genMinval,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"mod", &I::genMod},
    {"modulo", &I::genModulo},
    {"move_alloc",
     &I::genMoveAlloc,
     {{{"from", asInquired},
       {"to", asInquired},
       {"status", asAddr, handleDynamicOptional},
       {"errMsg", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"mvbits",
     &I::genMvbits,
     {{{"from", asValue},
       {"frompos", asValue},
       {"len", asValue},
       {"to", asAddr},
       {"topos", asValue}}}},
    {"nearest", &I::genNearest<I::NearestProc::Nearest>},
    {"nint", &I::genNint},
    {"norm2",
     &I::genNorm2,
     {{{"array", asBox}, {"dim", asValue}}},
     /*isElemental=*/false},
    {"not", &I::genNot},
    {"null", &I::genNull, {{{"mold", asInquired}}}, /*isElemental=*/false},
    {"num_images",
     &I::genNumImages,
     {{{"team_number", asValue}, {"team", asBox}}},
     /*isElemental*/ false},
    {"pack",
     &I::genPack,
     {{{"array", asBox},
       {"mask", asBox},
       {"vector", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"parity",
     &I::genParity,
     {{{"mask", asBox}, {"dim", asValue}}},
     /*isElemental=*/false},
    {"perror",
     &I::genPerror,
     {{{"string", asBox}}},
     /*isElemental*/ false},
    {"popcnt", &I::genPopcnt},
    {"poppar", &I::genPoppar},
    {"present",
     &I::genPresent,
     {{{"a", asInquired}}},
     /*isElemental=*/false},
    {"product",
     &I::genProduct,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"putenv",
     &I::genPutenv,
     {{{"str", asAddr}, {"status", asAddr, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"rand",
     &I::genRand,
     {{{"i", asAddr, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"random_init",
     &I::genRandomInit,
     {{{"repeatable", asValue}, {"image_distinct", asValue}}},
     /*isElemental=*/false},
    {"random_number",
     &I::genRandomNumber,
     {{{"harvest", asBox}}},
     /*isElemental=*/false},
    {"random_seed",
     &I::genRandomSeed,
     {{{"size", asBox, handleDynamicOptional},
       {"put", asBox, handleDynamicOptional},
       {"get", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"reduce",
     &I::genReduce,
     {{{"array", asBox},
       {"operation", asAddr},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional},
       {"identity", asAddr, handleDynamicOptional},
       {"ordered", asValue, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"rename",
     &I::genRename,
     {{{"path1", asBox},
       {"path2", asBox},
       {"status", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"repeat",
     &I::genRepeat,
     {{{"string", asAddr}, {"ncopies", asValue}}},
     /*isElemental=*/false},
    {"reshape",
     &I::genReshape,
     {{{"source", asBox},
       {"shape", asBox},
       {"pad", asBox, handleDynamicOptional},
       {"order", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"rrspacing", &I::genRRSpacing},
    {"rtc", &I::genRtc, {}, /*isElemental=*/false},
    {"same_type_as",
     &I::genSameTypeAs,
     {{{"a", asBox}, {"b", asBox}}},
     /*isElemental=*/false},
    {"scale",
     &I::genScale,
     {{{"x", asValue}, {"i", asValue}}},
     /*isElemental=*/true},
    {"scan",
     &I::genScan,
     {{{"string", asAddr},
       {"set", asAddr},
       {"back", asValue, handleDynamicOptional},
       {"kind", asValue}}},
     /*isElemental=*/true},
    {"secnds",
     &I::genSecnds,
     {{{"refTime", asAddr}}},
     /*isElemental=*/false},
    {"second",
     &I::genSecond,
     {{{"time", asAddr}}},
     /*isElemental=*/false},
    {"selected_char_kind",
     &I::genSelectedCharKind,
     {{{"name", asAddr}}},
     /*isElemental=*/false},
    {"selected_int_kind",
     &I::genSelectedIntKind,
     {{{"scalar", asAddr}}},
     /*isElemental=*/false},
    {"selected_logical_kind",
     &I::genSelectedLogicalKind,
     {{{"bits", asAddr}}},
     /*isElemental=*/false},
    {"selected_real_kind",
     &I::genSelectedRealKind,
     {{{"precision", asAddr, handleDynamicOptional},
       {"range", asAddr, handleDynamicOptional},
       {"radix", asAddr, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"selected_unsigned_kind",
     &I::genSelectedIntKind, // same results as selected_int_kind
     {{{"scalar", asAddr}}},
     /*isElemental=*/false},
    {"set_exponent", &I::genSetExponent},
    {"shape",
     &I::genShape,
     {{{"source", asBox}, {"kind", asValue}}},
     /*isElemental=*/false},
    {"shifta", &I::genShiftA},
    {"shiftl", &I::genShift<aiir::arith::ShLIOp>},
    {"shiftr", &I::genShift<aiir::arith::ShRUIOp>},
    {"show_descriptor",
     &I::genShowDescriptor,
     {{{"d", asInquired}}},
     /*isElemental=*/false},
    {"sign", &I::genSign},
    {"signal",
     &I::genSignalSubroutine,
     {{{"number", asValue}, {"handler", asAddr}, {"status", asAddr}}},
     /*isElemental=*/false},
    {"sind", &I::genSind},
    {"sinpi", &I::genSinpi},
    {"size",
     &I::genSize,
     {{{"array", asBox},
       {"dim", asAddr, handleDynamicOptional},
       {"kind", asValue}}},
     /*isElemental=*/false},
    {"sizeof",
     &I::genSizeOf,
     {{{"a", asBox}}},
     /*isElemental=*/false},
    {"sleep", &I::genSleep, {{{"seconds", asValue}}}, /*isElemental=*/false},
    {"spacing", &I::genSpacing},
    {"split",
     &I::genSplit,
     {{{"string", asAddr},
       {"set", asAddr},
       {"pos", asAddr},
       {"back", asValue, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"spread",
     &I::genSpread,
     {{{"source", asBox}, {"dim", asValue}, {"ncopies", asValue}}},
     /*isElemental=*/false},
    {"storage_size",
     &I::genStorageSize,
     {{{"a", asInquired}, {"kind", asValue}}},
     /*isElemental=*/false},
    {"sum",
     &I::genSum,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"system",
     &I::genSystem,
     {{{"command", asBox}, {"exitstat", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"system_clock",
     &I::genSystemClock,
     {{{"count", asAddr}, {"count_rate", asAddr}, {"count_max", asAddr}}},
     /*isElemental=*/false},
    {"tand", &I::genTand},
    {"tanpi", &I::genTanpi},
    {"team_number",
     &I::genTeamNumber,
     {{{"team", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"this_image",
     &I::genThisImage,
     {{{"coarray", asBox},
       {"dim", asAddr},
       {"team", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"time", &I::genTime, {}, /*isElemental=*/false},
    {"tokenize",
     &I::genTokenize,
     {{{"string", asAddr},
       {"set", asAddr},
       {"out1", asInquired},
       {"out2", asInquired, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"trailz", &I::genTrailz},
    {"transfer",
     &I::genTransfer,
     {{{"source", asAddr}, {"mold", asAddr}, {"size", asValue}}},
     /*isElemental=*/false},
    {"transpose",
     &I::genTranspose,
     {{{"matrix", asAddr}}},
     /*isElemental=*/false},
    {"trim", &I::genTrim, {{{"string", asAddr}}}, /*isElemental=*/false},
    {"ubound",
     &I::genUbound,
     {{{"array", asBox}, {"dim", asValue}, {"kind", asValue}}},
     /*isElemental=*/false},
    {"umaskl", &I::genMask<aiir::arith::ShLIOp>},
    {"umaskr", &I::genMask<aiir::arith::ShRUIOp>},
    {"unlink",
     &I::genUnlink,
     {{{"path", asAddr}, {"status", asAddr, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"unpack",
     &I::genUnpack,
     {{{"vector", asBox}, {"mask", asBox}, {"field", asBox}}},
     /*isElemental=*/false},
    {"verify",
     &I::genVerify,
     {{{"string", asAddr},
       {"set", asAddr},
       {"back", asValue, handleDynamicOptional},
       {"kind", asValue}}},
     /*isElemental=*/true},
};

template <std::size_t N>
static constexpr bool isSorted(const IntrinsicHandler (&array)[N]) {
  // Replace by std::sorted when C++20 is default (will be constexpr).
  const IntrinsicHandler *lastSeen{nullptr};
  bool isSorted{true};
  for (const auto &x : array) {
    if (lastSeen)
      isSorted &= std::string_view{lastSeen->name} < std::string_view{x.name};
    lastSeen = &x;
  }
  return isSorted;
}
static_assert(isSorted(handlers) && "map must be sorted");

static const IntrinsicHandler *findIntrinsicHandler(llvm::StringRef name) {
  auto compare = [](const IntrinsicHandler &handler, llvm::StringRef name) {
    return name.compare(handler.name) > 0;
  };
  auto result = llvm::lower_bound(handlers, name, compare);
  return result != std::end(handlers) && result->name == name ? result
                                                              : nullptr;
}

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

/// Command line option to modify math runtime behavior used to implement
/// intrinsics. This option applies both to early and late math-lowering modes.
enum MathRuntimeVersion { fastVersion, relaxedVersion, preciseVersion };
llvm::cl::opt<MathRuntimeVersion> mathRuntimeVersion(
    "math-runtime", llvm::cl::desc("Select math operations' runtime behavior:"),
    llvm::cl::values(
        clEnumValN(fastVersion, "fast", "use fast runtime behavior"),
        clEnumValN(relaxedVersion, "relaxed", "use relaxed runtime behavior"),
        clEnumValN(preciseVersion, "precise", "use precise runtime behavior")),
    llvm::cl::init(fastVersion));

static llvm::cl::opt<bool>
    forceAiirComplex("force-aiir-complex",
                     llvm::cl::desc("Force using AIIR complex operations "
                                    "instead of libm complex operations"),
                     llvm::cl::init(false));

/// Return a string containing the given Fortran intrinsic name
/// with the type of its arguments specified in funcType
/// surrounded by the given prefix/suffix.
static std::string
prettyPrintIntrinsicName(fir::FirOpBuilder &builder, aiir::Location loc,
                         llvm::StringRef prefix, llvm::StringRef name,
                         llvm::StringRef suffix, aiir::FunctionType funcType) {
  std::string output = prefix.str();
  llvm::raw_string_ostream sstream(output);
  if (name == "pow" || name == "pow-unsigned") {
    assert(funcType.getNumInputs() == 2 && "power operator has two arguments");
    std::string displayName{" ** "};
    sstream << aiirTypeToIntrinsicFortran(builder, funcType.getInput(0), loc,
                                          displayName)
            << displayName
            << aiirTypeToIntrinsicFortran(builder, funcType.getInput(1), loc,
                                          displayName);
  } else {
    sstream << name.upper() << "(";
    if (funcType.getNumInputs() > 0)
      sstream << aiirTypeToIntrinsicFortran(builder, funcType.getInput(0), loc,
                                            name);
    for (aiir::Type argType : funcType.getInputs().drop_front()) {
      sstream << ", "
              << aiirTypeToIntrinsicFortran(builder, argType, loc, name);
    }
    sstream << ")";
  }
  sstream << suffix;
  return output;
}

// Generate a call to the Fortran runtime library providing
// support for 128-bit float math.
// On 'HAS_LDBL128' targets the implementation
// is provided by flang_rt, otherwise, it is done via the
// libflang_rt.quadmath library. In the latter case the compiler
// has to be built with FLANG_RUNTIME_F128_MATH_LIB to guarantee
// proper linking actions in the driver.
static aiir::Value genLibF128Call(fir::FirOpBuilder &builder,
                                  aiir::Location loc,
                                  const MathOperation &mathOp,
                                  aiir::FunctionType libFuncType,
                                  llvm::ArrayRef<aiir::Value> args) {
  // TODO: if we knew that the C 'long double' does not have 113-bit mantissa
  // on the target, we could have asserted that FLANG_RUNTIME_F128_MATH_LIB
  // must be specified. For now just always generate the call even
  // if it will be unresolved.
  return genLibCall(builder, loc, mathOp, libFuncType, args);
}

aiir::Value genLibCall(fir::FirOpBuilder &builder, aiir::Location loc,
                       const MathOperation &mathOp,
                       aiir::FunctionType libFuncType,
                       llvm::ArrayRef<aiir::Value> args) {
  llvm::StringRef libFuncName = mathOp.runtimeFunc;

  // On AIX, __clog is used in libm.
  if (fir::getTargetTriple(builder.getModule()).isOSAIX() &&
      libFuncName == "clog") {
    libFuncName = "__clog";
  }

  LLVM_DEBUG(llvm::dbgs() << "Generating '" << libFuncName
                          << "' call with type ";
             libFuncType.dump(); llvm::dbgs() << "\n");
  aiir::func::FuncOp funcOp = builder.getNamedFunction(libFuncName);

  if (!funcOp) {
    funcOp = builder.createFunction(loc, libFuncName, libFuncType);
    // C-interoperability rules apply to these library functions.
    funcOp->setAttr(fir::getSymbolAttrName(),
                    aiir::StringAttr::get(builder.getContext(), libFuncName));
    // Set fir.runtime attribute to distinguish the function that
    // was just created from user functions with the same name.
    funcOp->setAttr(fir::FIROpsDialect::getFirRuntimeAttrName(),
                    builder.getUnitAttr());
    auto libCall = fir::CallOp::create(builder, loc, funcOp, args);
    // TODO: ensure 'strictfp' setting on the call for "precise/strict"
    //       FP mode. Set appropriate Fast-Math Flags otherwise.
    // TODO: we should also mark as many libm function as possible
    //       with 'pure' attribute (of course, not in strict FP mode).
    LLVM_DEBUG(libCall.dump(); llvm::dbgs() << "\n");
    return libCall.getResult(0);
  }

  // The function with the same name already exists.
  fir::CallOp libCall;
  aiir::Type soughtFuncType = funcOp.getFunctionType();

  if (soughtFuncType == libFuncType) {
    libCall = fir::CallOp::create(builder, loc, funcOp, args);
  } else {
    // A function with the same name might have been declared
    // before (e.g. with an explicit interface and a binding label).
    // It is in general incorrect to use the same definition for the library
    // call, but we have no other options. Type cast the function to match
    // the requested signature and generate an indirect call to avoid
    // later failures caused by the signature mismatch.
    LLVM_DEBUG(aiir::emitWarning(
        loc, llvm::Twine("function signature mismatch for '") +
                 llvm::Twine(libFuncName) +
                 llvm::Twine("' may lead to undefined behavior.")));
    aiir::SymbolRefAttr funcSymbolAttr = builder.getSymbolRefAttr(libFuncName);
    aiir::Value funcPointer =
        fir::AddrOfOp::create(builder, loc, soughtFuncType, funcSymbolAttr);
    funcPointer = builder.createConvert(loc, libFuncType, funcPointer);

    llvm::SmallVector<aiir::Value, 3> operands{funcPointer};
    operands.append(args.begin(), args.end());
    libCall = fir::CallOp::create(builder, loc, aiir::SymbolRefAttr{},
                                  libFuncType.getResults(), operands);
  }

  LLVM_DEBUG(libCall.dump(); llvm::dbgs() << "\n");
  return libCall.getResult(0);
}

aiir::Value genLibSplitComplexArgsCall(fir::FirOpBuilder &builder,
                                       aiir::Location loc,
                                       const MathOperation &mathOp,
                                       aiir::FunctionType libFuncType,
                                       llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2 && "Incorrect #args to genLibSplitComplexArgsCall");

  auto getSplitComplexArgsType = [&builder, &args]() -> aiir::FunctionType {
    aiir::Type ctype = args[0].getType();
    auto ftype = aiir::cast<aiir::ComplexType>(ctype).getElementType();
    return builder.getFunctionType({ftype, ftype, ftype, ftype}, {ctype});
  };

  llvm::SmallVector<aiir::Value, 4> splitArgs;
  aiir::Value cplx1 = args[0];
  auto real1 = fir::factory::Complex{builder, loc}.extractComplexPart(
      cplx1, /*isImagPart=*/false);
  splitArgs.push_back(real1);
  auto imag1 = fir::factory::Complex{builder, loc}.extractComplexPart(
      cplx1, /*isImagPart=*/true);
  splitArgs.push_back(imag1);
  aiir::Value cplx2 = args[1];
  auto real2 = fir::factory::Complex{builder, loc}.extractComplexPart(
      cplx2, /*isImagPart=*/false);
  splitArgs.push_back(real2);
  auto imag2 = fir::factory::Complex{builder, loc}.extractComplexPart(
      cplx2, /*isImagPart=*/true);
  splitArgs.push_back(imag2);

  return genLibCall(builder, loc, mathOp, getSplitComplexArgsType(), splitArgs);
}

template <typename T>
aiir::Value genMathOp(fir::FirOpBuilder &builder, aiir::Location loc,
                      const MathOperation &mathOp,
                      aiir::FunctionType mathLibFuncType,
                      llvm::ArrayRef<aiir::Value> args) {
  // TODO: we have to annotate the math operations with flags
  //       that will allow to define FP accuracy/exception
  //       behavior per operation, so that after early multi-module
  //       AIIR inlining we can distiguish operation that were
  //       compiled with different settings.
  //       Suggestion:
  //         * For "relaxed" FP mode set all Fast-Math Flags
  //           (see "[RFC] FastMath flags support in AIIR (arith dialect)"
  //           topic at discourse.llvm.org).
  //         * For "fast" FP mode set all Fast-Math Flags except 'afn'.
  //         * For "precise/strict" FP mode generate fir.calls to libm
  //           entries and annotate them with an attribute that will
  //           end up transformed into 'strictfp' LLVM attribute (TBD).
  //           Elsewhere, "precise/strict" FP mode should also set
  //           'strictfp' for all user functions and calls so that
  //           LLVM backend does the right job.
  //         * Operations that cannot be reasonably optimized in AIIR
  //           can be also lowered to libm calls for "fast" and "relaxed"
  //           modes.
  aiir::Value result;
  llvm::StringRef mathLibFuncName = mathOp.runtimeFunc;
  if (mathRuntimeVersion == preciseVersion &&
      // Some operations do not have to be lowered as conservative
      // calls, since they do not affect strict FP behavior.
      // For example, purely integer operations like exponentiation
      // with integer operands fall into this class.
      !mathLibFuncName.empty()) {
    result = genLibCall(builder, loc, mathOp, mathLibFuncType, args);
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Generating '" << mathLibFuncName
                            << "' operation with type ";
               mathLibFuncType.dump(); llvm::dbgs() << "\n");
    result = T::create(builder, loc, args);
  }
  LLVM_DEBUG(result.dump(); llvm::dbgs() << "\n");
  return result;
}

template <typename T>
aiir::Value genComplexMathOp(fir::FirOpBuilder &builder, aiir::Location loc,
                             const MathOperation &mathOp,
                             aiir::FunctionType mathLibFuncType,
                             llvm::ArrayRef<aiir::Value> args) {
  aiir::Value result;
  bool canUseApprox = aiir::arith::bitEnumContainsAny(
      builder.getFastMathFlags(), aiir::arith::FastMathFlags::afn);

  // If we have libm functions, we can attempt to generate the more precise
  // version of the complex math operation.
  llvm::StringRef mathLibFuncName = mathOp.runtimeFunc;
  if (!mathLibFuncName.empty()) {
    // If we enabled AIIR complex or can use approximate operations, we should
    // NOT use libm. Avoid libm when targeting AMDGPU as those symbols are not
    // available on the device and we rely on AIIR complex operations to
    // later map to OCML calls.
    bool isAMDGPU = fir::getTargetTriple(builder.getModule()).isAMDGCN();
    if (!forceAiirComplex && !canUseApprox && !isAMDGPU) {
      result = genLibCall(builder, loc, mathOp, mathLibFuncType, args);
      LLVM_DEBUG(result.dump(); llvm::dbgs() << "\n");
      return result;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Generating '" << mathLibFuncName
                          << "' operation with type ";
             mathLibFuncType.dump(); llvm::dbgs() << "\n");
  // Builder expects an extra return type to be provided if different to
  // the argument types for an operation
  if constexpr (T::template hasTrait<
                    aiir::OpTrait::SameOperandsAndResultType>()) {
    result = T::create(builder, loc, args);
    result = builder.createConvert(loc, mathLibFuncType.getResult(0), result);
  } else {
    auto complexTy = aiir::cast<aiir::ComplexType>(mathLibFuncType.getInput(0));
    auto realTy = complexTy.getElementType();
    result = T::create(builder, loc, realTy, args);
    result = builder.createConvert(loc, mathLibFuncType.getResult(0), result);
  }

  LLVM_DEBUG(result.dump(); llvm::dbgs() << "\n");
  return result;
}

/// Mapping between mathematical intrinsic operations and AIIR operations
/// of some appropriate dialect (math, complex, etc.) or libm calls.
/// TODO: support remaining Fortran math intrinsics.
///       See https://gcc.gnu.org/onlinedocs/gcc-12.1.0/gfortran/\
///       Intrinsic-Procedures.html for a reference.
constexpr auto FuncTypeReal16Real16 = genFuncType<Ty::Real<16>, Ty::Real<16>>;
constexpr auto FuncTypeReal16Real16Real16 =
    genFuncType<Ty::Real<16>, Ty::Real<16>, Ty::Real<16>>;
constexpr auto FuncTypeReal16Real16Real16Real16 =
    genFuncType<Ty::Real<16>, Ty::Real<16>, Ty::Real<16>, Ty::Real<16>>;
constexpr auto FuncTypeReal16Integer4Real16 =
    genFuncType<Ty::Real<16>, Ty::Integer<4>, Ty::Real<16>>;
constexpr auto FuncTypeInteger4Real16 =
    genFuncType<Ty::Integer<4>, Ty::Real<16>>;
constexpr auto FuncTypeInteger8Real16 =
    genFuncType<Ty::Integer<8>, Ty::Real<16>>;
constexpr auto FuncTypeReal16Complex16 =
    genFuncType<Ty::Real<16>, Ty::Complex<16>>;
constexpr auto FuncTypeComplex16Complex16 =
    genFuncType<Ty::Complex<16>, Ty::Complex<16>>;
constexpr auto FuncTypeComplex16Complex16Complex16 =
    genFuncType<Ty::Complex<16>, Ty::Complex<16>, Ty::Complex<16>>;
constexpr auto FuncTypeComplex16Complex16Integer4 =
    genFuncType<Ty::Complex<16>, Ty::Complex<16>, Ty::Integer<4>>;
constexpr auto FuncTypeComplex16Complex16Integer8 =
    genFuncType<Ty::Complex<16>, Ty::Complex<16>, Ty::Integer<8>>;

static constexpr MathOperation mathOperations[] = {
    {"abs", "fabsf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::AbsFOp>},
    {"abs", "fabs", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::AbsFOp>},
    {"abs", "llvm.fabs.f128", genFuncType<Ty::Real<16>, Ty::Real<16>>,
     genMathOp<aiir::math::AbsFOp>},
    {"abs", "cabsf", genFuncType<Ty::Real<4>, Ty::Complex<4>>,
     genComplexMathOp<aiir::complex::AbsOp>},
    {"abs", "cabs", genFuncType<Ty::Real<8>, Ty::Complex<8>>,
     genComplexMathOp<aiir::complex::AbsOp>},
    {"abs", RTNAME_STRING(CAbsF128), FuncTypeReal16Complex16, genLibF128Call},
    {"acos", "acosf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::AcosOp>},
    {"acos", "acos", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::AcosOp>},
    {"acos", RTNAME_STRING(AcosF128), FuncTypeReal16Real16, genLibF128Call},
    {"acos", "cacosf", genFuncType<Ty::Complex<4>, Ty::Complex<4>>, genLibCall},
    {"acos", "cacos", genFuncType<Ty::Complex<8>, Ty::Complex<8>>, genLibCall},
    {"acos", RTNAME_STRING(CAcosF128), FuncTypeComplex16Complex16,
     genLibF128Call},
    {"acosh", "acoshf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::AcoshOp>},
    {"acosh", "acosh", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::AcoshOp>},
    {"acosh", RTNAME_STRING(AcoshF128), FuncTypeReal16Real16, genLibF128Call},
    {"acosh", "cacoshf", genFuncType<Ty::Complex<4>, Ty::Complex<4>>,
     genLibCall},
    {"acosh", "cacosh", genFuncType<Ty::Complex<8>, Ty::Complex<8>>,
     genLibCall},
    {"acosh", RTNAME_STRING(CAcoshF128), FuncTypeComplex16Complex16,
     genLibF128Call},
    // llvm.trunc behaves the same way as libm's trunc.
    {"aint", "llvm.trunc.f32", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genLibCall},
    {"aint", "llvm.trunc.f64", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"aint", "llvm.trunc.f80", genFuncType<Ty::Real<10>, Ty::Real<10>>,
     genLibCall},
    {"aint", RTNAME_STRING(TruncF128), FuncTypeReal16Real16, genLibF128Call},
    // llvm.round behaves the same way as libm's round.
    {"anint", "llvm.round.f32", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::RoundOp>},
    {"anint", "llvm.round.f64", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::RoundOp>},
    {"anint", "llvm.round.f80", genFuncType<Ty::Real<10>, Ty::Real<10>>,
     genMathOp<aiir::math::RoundOp>},
    {"anint", RTNAME_STRING(RoundF128), FuncTypeReal16Real16, genLibF128Call},
    {"asin", "asinf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::AsinOp>},
    {"asin", "asin", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::AsinOp>},
    {"asin", RTNAME_STRING(AsinF128), FuncTypeReal16Real16, genLibF128Call},
    {"asin", "casinf", genFuncType<Ty::Complex<4>, Ty::Complex<4>>, genLibCall},
    {"asin", "casin", genFuncType<Ty::Complex<8>, Ty::Complex<8>>, genLibCall},
    {"asin", RTNAME_STRING(CAsinF128), FuncTypeComplex16Complex16,
     genLibF128Call},
    {"asinh", "asinhf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::AsinhOp>},
    {"asinh", "asinh", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::AsinhOp>},
    {"asinh", RTNAME_STRING(AsinhF128), FuncTypeReal16Real16, genLibF128Call},
    {"asinh", "casinhf", genFuncType<Ty::Complex<4>, Ty::Complex<4>>,
     genLibCall},
    {"asinh", "casinh", genFuncType<Ty::Complex<8>, Ty::Complex<8>>,
     genLibCall},
    {"asinh", RTNAME_STRING(CAsinhF128), FuncTypeComplex16Complex16,
     genLibF128Call},
    {"atan", "atanf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::AtanOp>},
    {"atan", "atan", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::AtanOp>},
    {"atan", RTNAME_STRING(AtanF128), FuncTypeReal16Real16, genLibF128Call},
    {"atan", "catanf", genFuncType<Ty::Complex<4>, Ty::Complex<4>>, genLibCall},
    {"atan", "catan", genFuncType<Ty::Complex<8>, Ty::Complex<8>>, genLibCall},
    {"atan", RTNAME_STRING(CAtanF128), FuncTypeComplex16Complex16,
     genLibF128Call},
    {"atan", "atan2f", genFuncType<Ty::Real<4>, Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::Atan2Op>},
    {"atan", "atan2", genFuncType<Ty::Real<8>, Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::Atan2Op>},
    {"atan", RTNAME_STRING(Atan2F128), FuncTypeReal16Real16Real16,
     genLibF128Call},
    {"atan2", "atan2f", genFuncType<Ty::Real<4>, Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::Atan2Op>},
    {"atan2", "atan2", genFuncType<Ty::Real<8>, Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::Atan2Op>},
    {"atan2", RTNAME_STRING(Atan2F128), FuncTypeReal16Real16Real16,
     genLibF128Call},
    {"atanh", "atanhf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::AtanhOp>},
    {"atanh", "atanh", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::AtanhOp>},
    {"atanh", RTNAME_STRING(AtanhF128), FuncTypeReal16Real16, genLibF128Call},
    {"atanh", "catanhf", genFuncType<Ty::Complex<4>, Ty::Complex<4>>,
     genLibCall},
    {"atanh", "catanh", genFuncType<Ty::Complex<8>, Ty::Complex<8>>,
     genLibCall},
    {"atanh", RTNAME_STRING(CAtanhF128), FuncTypeComplex16Complex16,
     genLibF128Call},
    {"bessel_j0", "j0f", genFuncType<Ty::Real<4>, Ty::Real<4>>, genLibCall},
    {"bessel_j0", "j0", genFuncType<Ty::Real<8>, Ty::Real<8>>, genLibCall},
    {"bessel_j0", RTNAME_STRING(J0F128), FuncTypeReal16Real16, genLibF128Call},
    {"bessel_j1", "j1f", genFuncType<Ty::Real<4>, Ty::Real<4>>, genLibCall},
    {"bessel_j1", "j1", genFuncType<Ty::Real<8>, Ty::Real<8>>, genLibCall},
    {"bessel_j1", RTNAME_STRING(J1F128), FuncTypeReal16Real16, genLibF128Call},
    {"bessel_jn", "jnf", genFuncType<Ty::Real<4>, Ty::Integer<4>, Ty::Real<4>>,
     genLibCall},
    {"bessel_jn", "jn", genFuncType<Ty::Real<8>, Ty::Integer<4>, Ty::Real<8>>,
     genLibCall},
    {"bessel_jn", RTNAME_STRING(JnF128), FuncTypeReal16Integer4Real16,
     genLibF128Call},
    {"bessel_y0", "y0f", genFuncType<Ty::Real<4>, Ty::Real<4>>, genLibCall},
    {"bessel_y0", "y0", genFuncType<Ty::Real<8>, Ty::Real<8>>, genLibCall},
    {"bessel_y0", RTNAME_STRING(Y0F128), FuncTypeReal16Real16, genLibF128Call},
    {"bessel_y1", "y1f", genFuncType<Ty::Real<4>, Ty::Real<4>>, genLibCall},
    {"bessel_y1", "y1", genFuncType<Ty::Real<8>, Ty::Real<8>>, genLibCall},
    {"bessel_y1", RTNAME_STRING(Y1F128), FuncTypeReal16Real16, genLibF128Call},
    {"bessel_yn", "ynf", genFuncType<Ty::Real<4>, Ty::Integer<4>, Ty::Real<4>>,
     genLibCall},
    {"bessel_yn", "yn", genFuncType<Ty::Real<8>, Ty::Integer<4>, Ty::Real<8>>,
     genLibCall},
    {"bessel_yn", RTNAME_STRING(YnF128), FuncTypeReal16Integer4Real16,
     genLibF128Call},
    // math::CeilOp returns a real, while Fortran CEILING returns integer.
    {"ceil", "ceilf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::CeilOp>},
    {"ceil", "ceil", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::CeilOp>},
    {"ceil", RTNAME_STRING(CeilF128), FuncTypeReal16Real16, genLibF128Call},
    {"cos", "cosf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::CosOp>},
    {"cos", "cos", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::CosOp>},
    {"cos", RTNAME_STRING(CosF128), FuncTypeReal16Real16, genLibF128Call},
    {"cos", "ccosf", genFuncType<Ty::Complex<4>, Ty::Complex<4>>,
     genComplexMathOp<aiir::complex::CosOp>},
    {"cos", "ccos", genFuncType<Ty::Complex<8>, Ty::Complex<8>>,
     genComplexMathOp<aiir::complex::CosOp>},
    {"cos", RTNAME_STRING(CCosF128), FuncTypeComplex16Complex16,
     genLibF128Call},
    {"cosh", "coshf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::CoshOp>},
    {"cosh", "cosh", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::CoshOp>},
    {"cosh", RTNAME_STRING(CoshF128), FuncTypeReal16Real16, genLibF128Call},
    {"cosh", "ccoshf", genFuncType<Ty::Complex<4>, Ty::Complex<4>>, genLibCall},
    {"cosh", "ccosh", genFuncType<Ty::Complex<8>, Ty::Complex<8>>, genLibCall},
    {"cosh", RTNAME_STRING(CCoshF128), FuncTypeComplex16Complex16,
     genLibF128Call},
    {"divc",
     {},
     genFuncType<Ty::Complex<2>, Ty::Complex<2>, Ty::Complex<2>>,
     genComplexMathOp<aiir::complex::DivOp>},
    {"divc",
     {},
     genFuncType<Ty::Complex<3>, Ty::Complex<3>, Ty::Complex<3>>,
     genComplexMathOp<aiir::complex::DivOp>},
    {"divc", "__divsc3",
     genFuncType<Ty::Complex<4>, Ty::Complex<4>, Ty::Complex<4>>,
     genLibSplitComplexArgsCall},
    {"divc", "__divdc3",
     genFuncType<Ty::Complex<8>, Ty::Complex<8>, Ty::Complex<8>>,
     genLibSplitComplexArgsCall},
    {"divc", "__divxc3",
     genFuncType<Ty::Complex<10>, Ty::Complex<10>, Ty::Complex<10>>,
     genLibSplitComplexArgsCall},
    {"divc", "__divtc3",
     genFuncType<Ty::Complex<16>, Ty::Complex<16>, Ty::Complex<16>>,
     genLibSplitComplexArgsCall},
    {"erf", "erff", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::ErfOp>},
    {"erf", "erf", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::ErfOp>},
    {"erf", RTNAME_STRING(ErfF128), FuncTypeReal16Real16, genLibF128Call},
    {"erfc", "erfcf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::ErfcOp>},
    {"erfc", "erfc", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::ErfcOp>},
    {"erfc", RTNAME_STRING(ErfcF128), FuncTypeReal16Real16, genLibF128Call},
    {"exp", "expf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::ExpOp>},
    {"exp", "exp", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::ExpOp>},
    {"exp", RTNAME_STRING(ExpF128), FuncTypeReal16Real16, genLibF128Call},
    {"exp", "cexpf", genFuncType<Ty::Complex<4>, Ty::Complex<4>>,
     genComplexMathOp<aiir::complex::ExpOp>},
    {"exp", "cexp", genFuncType<Ty::Complex<8>, Ty::Complex<8>>,
     genComplexMathOp<aiir::complex::ExpOp>},
    {"exp", RTNAME_STRING(CExpF128), FuncTypeComplex16Complex16,
     genLibF128Call},
    {"feclearexcept", "feclearexcept",
     genFuncType<Ty::Integer<4>, Ty::Integer<4>>, genLibCall},
    {"fedisableexcept", "fedisableexcept",
     genFuncType<Ty::Integer<4>, Ty::Integer<4>>, genLibCall},
    {"feenableexcept", "feenableexcept",
     genFuncType<Ty::Integer<4>, Ty::Integer<4>>, genLibCall},
    {"fegetenv", "fegetenv", genFuncType<Ty::Integer<4>, Ty::Address<4>>,
     genLibCall},
    {"fegetexcept", "fegetexcept", genFuncType<Ty::Integer<4>>, genLibCall},
    {"fegetmode", "fegetmode", genFuncType<Ty::Integer<4>, Ty::Address<4>>,
     genLibCall},
    {"feraiseexcept", "feraiseexcept",
     genFuncType<Ty::Integer<4>, Ty::Integer<4>>, genLibCall},
    {"fesetenv", "fesetenv", genFuncType<Ty::Integer<4>, Ty::Address<4>>,
     genLibCall},
    {"fesetmode", "fesetmode", genFuncType<Ty::Integer<4>, Ty::Address<4>>,
     genLibCall},
    {"fetestexcept", "fetestexcept",
     genFuncType<Ty::Integer<4>, Ty::Integer<4>>, genLibCall},
    {"feupdateenv", "feupdateenv", genFuncType<Ty::Integer<4>, Ty::Address<4>>,
     genLibCall},
    // math::FloorOp returns a real, while Fortran FLOOR returns integer.
    {"floor", "floorf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::FloorOp>},
    {"floor", "floor", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::FloorOp>},
    {"floor", RTNAME_STRING(FloorF128), FuncTypeReal16Real16, genLibF128Call},
    {"fma", "llvm.fma.f32",
     genFuncType<Ty::Real<4>, Ty::Real<4>, Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::FmaOp>},
    {"fma", "llvm.fma.f64",
     genFuncType<Ty::Real<8>, Ty::Real<8>, Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::FmaOp>},
    {"fma", RTNAME_STRING(FmaF128), FuncTypeReal16Real16Real16Real16,
     genLibF128Call},
    {"gamma", "tgammaf", genFuncType<Ty::Real<4>, Ty::Real<4>>, genLibCall},
    {"gamma", "tgamma", genFuncType<Ty::Real<8>, Ty::Real<8>>, genLibCall},
    {"gamma", RTNAME_STRING(TgammaF128), FuncTypeReal16Real16, genLibF128Call},
    {"hypot", "hypotf", genFuncType<Ty::Real<4>, Ty::Real<4>, Ty::Real<4>>,
     genLibCall},
    {"hypot", "hypot", genFuncType<Ty::Real<8>, Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"hypot", RTNAME_STRING(HypotF128), FuncTypeReal16Real16Real16,
     genLibF128Call},
    {"log", "logf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::LogOp>},
    {"log", "log", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::LogOp>},
    {"log", RTNAME_STRING(LogF128), FuncTypeReal16Real16, genLibF128Call},
    {"log", "clogf", genFuncType<Ty::Complex<4>, Ty::Complex<4>>,
     genComplexMathOp<aiir::complex::LogOp>},
    {"log", "clog", genFuncType<Ty::Complex<8>, Ty::Complex<8>>,
     genComplexMathOp<aiir::complex::LogOp>},
    {"log", RTNAME_STRING(CLogF128), FuncTypeComplex16Complex16,
     genLibF128Call},
    {"log10", "log10f", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::Log10Op>},
    {"log10", "log10", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::Log10Op>},
    {"log10", RTNAME_STRING(Log10F128), FuncTypeReal16Real16, genLibF128Call},
    {"log_gamma", "lgammaf", genFuncType<Ty::Real<4>, Ty::Real<4>>, genLibCall},
    {"log_gamma", "lgamma", genFuncType<Ty::Real<8>, Ty::Real<8>>, genLibCall},
    {"log_gamma", RTNAME_STRING(LgammaF128), FuncTypeReal16Real16,
     genLibF128Call},
    {"nearbyint", "llvm.nearbyint.f32", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genLibCall},
    {"nearbyint", "llvm.nearbyint.f64", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"nearbyint", "llvm.nearbyint.f80", genFuncType<Ty::Real<10>, Ty::Real<10>>,
     genLibCall},
    {"nearbyint", RTNAME_STRING(NearbyintF128), FuncTypeReal16Real16,
     genLibF128Call},
    // llvm.lround behaves the same way as libm's lround.
    {"nint", "llvm.lround.i64.f64", genFuncType<Ty::Integer<8>, Ty::Real<8>>,
     genLibCall},
    {"nint", "llvm.lround.i64.f32", genFuncType<Ty::Integer<8>, Ty::Real<4>>,
     genLibCall},
    {"nint", RTNAME_STRING(LlroundF128), FuncTypeInteger8Real16,
     genLibF128Call},
    {"nint", "llvm.lround.i32.f64", genFuncType<Ty::Integer<4>, Ty::Real<8>>,
     genLibCall},
    {"nint", "llvm.lround.i32.f32", genFuncType<Ty::Integer<4>, Ty::Real<4>>,
     genLibCall},
    {"nint", RTNAME_STRING(LroundF128), FuncTypeInteger4Real16, genLibF128Call},
    {"pow",
     {},
     genFuncType<Ty::Integer<1>, Ty::Integer<1>, Ty::Integer<1>>,
     genMathOp<aiir::math::IPowIOp>},
    {"pow",
     {},
     genFuncType<Ty::Integer<2>, Ty::Integer<2>, Ty::Integer<2>>,
     genMathOp<aiir::math::IPowIOp>},
    {"pow",
     {},
     genFuncType<Ty::Integer<4>, Ty::Integer<4>, Ty::Integer<4>>,
     genMathOp<aiir::math::IPowIOp>},
    {"pow",
     {},
     genFuncType<Ty::Integer<8>, Ty::Integer<8>, Ty::Integer<8>>,
     genMathOp<aiir::math::IPowIOp>},
    {"pow", "powf", genFuncType<Ty::Real<4>, Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::PowFOp>},
    {"pow", "pow", genFuncType<Ty::Real<8>, Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::PowFOp>},
    {"pow", RTNAME_STRING(PowF128), FuncTypeReal16Real16Real16, genLibF128Call},
    {"pow", "cpowf",
     genFuncType<Ty::Complex<4>, Ty::Complex<4>, Ty::Complex<4>>,
     genMathOp<aiir::complex::PowOp>},
    {"pow", "cpow", genFuncType<Ty::Complex<8>, Ty::Complex<8>, Ty::Complex<8>>,
     genMathOp<aiir::complex::PowOp>},
    {"pow", RTNAME_STRING(CPowF128), FuncTypeComplex16Complex16Complex16,
     genMathOp<aiir::complex::PowOp>},
    {"pow", RTNAME_STRING(FPow4i),
     genFuncType<Ty::Real<4>, Ty::Real<4>, Ty::Integer<4>>,
     genMathOp<aiir::math::FPowIOp>},
    {"pow", RTNAME_STRING(FPow8i),
     genFuncType<Ty::Real<8>, Ty::Real<8>, Ty::Integer<4>>,
     genMathOp<aiir::math::FPowIOp>},
    {"pow", RTNAME_STRING(FPow16i),
     genFuncType<Ty::Real<16>, Ty::Real<16>, Ty::Integer<4>>,
     genMathOp<aiir::math::FPowIOp>},
    {"pow", RTNAME_STRING(FPow4k),
     genFuncType<Ty::Real<4>, Ty::Real<4>, Ty::Integer<8>>,
     genMathOp<aiir::math::FPowIOp>},
    {"pow", RTNAME_STRING(FPow8k),
     genFuncType<Ty::Real<8>, Ty::Real<8>, Ty::Integer<8>>,
     genMathOp<aiir::math::FPowIOp>},
    {"pow", RTNAME_STRING(FPow16k),
     genFuncType<Ty::Real<16>, Ty::Real<16>, Ty::Integer<8>>,
     genMathOp<aiir::math::FPowIOp>},
    {"pow", RTNAME_STRING(cpowi),
     genFuncType<Ty::Complex<4>, Ty::Complex<4>, Ty::Integer<4>>,
     genMathOp<aiir::complex::PowiOp>},
    {"pow", RTNAME_STRING(zpowi),
     genFuncType<Ty::Complex<8>, Ty::Complex<8>, Ty::Integer<4>>,
     genMathOp<aiir::complex::PowiOp>},
    {"pow", RTNAME_STRING(cqpowi), FuncTypeComplex16Complex16Integer4,
     genMathOp<aiir::complex::PowiOp>},
    {"pow", RTNAME_STRING(cpowk),
     genFuncType<Ty::Complex<4>, Ty::Complex<4>, Ty::Integer<8>>,
     genMathOp<aiir::complex::PowiOp>},
    {"pow", RTNAME_STRING(zpowk),
     genFuncType<Ty::Complex<8>, Ty::Complex<8>, Ty::Integer<8>>,
     genMathOp<aiir::complex::PowiOp>},
    {"pow", RTNAME_STRING(cqpowk), FuncTypeComplex16Complex16Integer8,
     genMathOp<aiir::complex::PowiOp>},
    {"pow-unsigned", RTNAME_STRING(UPow1),
     genFuncType<Ty::Integer<1>, Ty::Integer<1>, Ty::Integer<1>>, genLibCall},
    {"pow-unsigned", RTNAME_STRING(UPow2),
     genFuncType<Ty::Integer<2>, Ty::Integer<2>, Ty::Integer<2>>, genLibCall},
    {"pow-unsigned", RTNAME_STRING(UPow4),
     genFuncType<Ty::Integer<4>, Ty::Integer<4>, Ty::Integer<4>>, genLibCall},
    {"pow-unsigned", RTNAME_STRING(UPow8),
     genFuncType<Ty::Integer<8>, Ty::Integer<8>, Ty::Integer<8>>, genLibCall},
    {"remainder", "remainderf",
     genFuncType<Ty::Real<4>, Ty::Real<4>, Ty::Real<4>>, genLibCall},
    {"remainder", "remainder",
     genFuncType<Ty::Real<8>, Ty::Real<8>, Ty::Real<8>>, genLibCall},
    {"remainder", "remainderl",
     genFuncType<Ty::Real<10>, Ty::Real<10>, Ty::Real<10>>, genLibCall},
    {"remainder", RTNAME_STRING(RemainderF128), FuncTypeReal16Real16Real16,
     genLibF128Call},
    {"sign", "copysignf", genFuncType<Ty::Real<4>, Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::CopySignOp>},
    {"sign", "copysign", genFuncType<Ty::Real<8>, Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::CopySignOp>},
    {"sign", "copysignl", genFuncType<Ty::Real<10>, Ty::Real<10>, Ty::Real<10>>,
     genMathOp<aiir::math::CopySignOp>},
    {"sign", "llvm.copysign.f128",
     genFuncType<Ty::Real<16>, Ty::Real<16>, Ty::Real<16>>,
     genMathOp<aiir::math::CopySignOp>},
    {"sin", "sinf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::SinOp>},
    {"sin", "sin", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::SinOp>},
    {"sin", RTNAME_STRING(SinF128), FuncTypeReal16Real16, genLibF128Call},
    {"sin", "csinf", genFuncType<Ty::Complex<4>, Ty::Complex<4>>,
     genComplexMathOp<aiir::complex::SinOp>},
    {"sin", "csin", genFuncType<Ty::Complex<8>, Ty::Complex<8>>,
     genComplexMathOp<aiir::complex::SinOp>},
    {"sin", RTNAME_STRING(CSinF128), FuncTypeComplex16Complex16,
     genLibF128Call},
    {"sinh", "sinhf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::SinhOp>},
    {"sinh", "sinh", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::SinhOp>},
    {"sinh", RTNAME_STRING(SinhF128), FuncTypeReal16Real16, genLibF128Call},
    {"sinh", "csinhf", genFuncType<Ty::Complex<4>, Ty::Complex<4>>, genLibCall},
    {"sinh", "csinh", genFuncType<Ty::Complex<8>, Ty::Complex<8>>, genLibCall},
    {"sinh", RTNAME_STRING(CSinhF128), FuncTypeComplex16Complex16,
     genLibF128Call},
    {"sqrt", "sqrtf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::SqrtOp>},
    {"sqrt", "sqrt", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::SqrtOp>},
    {"sqrt", RTNAME_STRING(SqrtF128), FuncTypeReal16Real16, genLibF128Call},
    {"sqrt", "csqrtf", genFuncType<Ty::Complex<4>, Ty::Complex<4>>,
     genComplexMathOp<aiir::complex::SqrtOp>},
    {"sqrt", "csqrt", genFuncType<Ty::Complex<8>, Ty::Complex<8>>,
     genComplexMathOp<aiir::complex::SqrtOp>},
    {"sqrt", RTNAME_STRING(CSqrtF128), FuncTypeComplex16Complex16,
     genLibF128Call},
    {"tan", "tanf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::TanOp>},
    {"tan", "tan", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::TanOp>},
    {"tan", RTNAME_STRING(TanF128), FuncTypeReal16Real16, genLibF128Call},
    {"tan", "ctanf", genFuncType<Ty::Complex<4>, Ty::Complex<4>>,
     genComplexMathOp<aiir::complex::TanOp>},
    {"tan", "ctan", genFuncType<Ty::Complex<8>, Ty::Complex<8>>,
     genComplexMathOp<aiir::complex::TanOp>},
    {"tan", RTNAME_STRING(CTanF128), FuncTypeComplex16Complex16,
     genLibF128Call},
    {"tanh", "tanhf", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genMathOp<aiir::math::TanhOp>},
    {"tanh", "tanh", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genMathOp<aiir::math::TanhOp>},
    {"tanh", RTNAME_STRING(TanhF128), FuncTypeReal16Real16, genLibF128Call},
    {"tanh", "ctanhf", genFuncType<Ty::Complex<4>, Ty::Complex<4>>,
     genComplexMathOp<aiir::complex::TanhOp>},
    {"tanh", "ctanh", genFuncType<Ty::Complex<8>, Ty::Complex<8>>,
     genComplexMathOp<aiir::complex::TanhOp>},
    {"tanh", RTNAME_STRING(CTanhF128), FuncTypeComplex16Complex16,
     genLibF128Call},
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

  FunctionDistance(aiir::FunctionType from, aiir::FunctionType to) {
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

  void addArgumentDistance(aiir::Type from, aiir::Type to) {
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

  void addResultDistance(aiir::Type from, aiir::Type to) {
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

  // Floating point can be aiir Float or Complex Type.
  static unsigned getFloatingPointWidth(aiir::Type t) {
    if (auto f{aiir::dyn_cast<aiir::FloatType>(t)})
      return f.getWidth();
    if (auto cplx{aiir::dyn_cast<aiir::ComplexType>(t)})
      return aiir::cast<aiir::FloatType>(cplx.getElementType()).getWidth();
    llvm_unreachable("not a floating-point type");
  }

  static Conversion conversionBetweenTypes(aiir::Type from, aiir::Type to) {
    if (from == to)
      return Conversion::None;

    if (auto fromIntTy{aiir::dyn_cast<aiir::IntegerType>(from)}) {
      if (auto toIntTy{aiir::dyn_cast<aiir::IntegerType>(to)}) {
        return fromIntTy.getWidth() > toIntTy.getWidth() ? Conversion::Narrow
                                                         : Conversion::Extend;
      }
    }

    if (fir::isa_real(from) && fir::isa_real(to)) {
      return getFloatingPointWidth(from) > getFloatingPointWidth(to)
                 ? Conversion::Narrow
                 : Conversion::Extend;
    }

    if (fir::isa_complex(from) && fir::isa_complex(to)) {
      return getFloatingPointWidth(from) > getFloatingPointWidth(to)
                 ? Conversion::Narrow
                 : Conversion::Extend;
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

using RtMap = Fortran::common::StaticMultimapView<MathOperation>;
static constexpr RtMap mathOps(mathOperations);
static_assert(mathOps.Verify() && "map must be sorted");

/// Look for a MathOperation entry specifying how to lower a mathematical
/// operation defined by \p name with its result' and operands' types
/// specified in the form of a FunctionType \p funcType.
/// If exact match for the given types is found, then the function
/// returns a pointer to the corresponding MathOperation.
/// Otherwise, the function returns nullptr.
/// If there is a MathOperation that can be used with additional
/// type casts for the operands or/and result (non-exact match),
/// then it is returned via \p bestNearMatch argument, and
/// \p bestMatchDistance specifies the FunctionDistance between
/// the requested operation and the non-exact match.
static const MathOperation *
searchMathOperation(fir::FirOpBuilder &builder,
                    const IntrinsicHandlerEntry::RuntimeGeneratorRange &range,
                    aiir::FunctionType funcType,
                    const MathOperation **bestNearMatch,
                    FunctionDistance &bestMatchDistance) {
  for (auto iter = range.first; iter != range.second && iter; ++iter) {
    const auto &impl = *iter;
    auto implType = impl.typeGenerator(builder.getContext(), builder);
    if (funcType == implType) {
      return &impl; // exact match
    }

    FunctionDistance distance(funcType, implType);
    if (distance.isSmallerThan(bestMatchDistance)) {
      *bestNearMatch = &impl;
      bestMatchDistance = std::move(distance);
    }
  }
  return nullptr;
}

/// Implementation of the operation defined by \p name with type
/// \p funcType is not precise, and the actual available implementation
/// is \p distance away from the requested. If using the available
/// implementation results in a precision loss, emit an error message
/// with the given code location \p loc.
static void checkPrecisionLoss(llvm::StringRef name,
                               aiir::FunctionType funcType,
                               const FunctionDistance &distance,
                               fir::FirOpBuilder &builder, aiir::Location loc) {
  if (!distance.isLosingPrecision())
    return;

  // Using this runtime version requires narrowing the arguments
  // or extending the result. It is not numerically safe. There
  // is currently no quad math library that was described in
  // lowering and could be used here. Emit an error and continue
  // generating the code with the narrowing cast so that the user
  // can get a complete list of the problematic intrinsic calls.
  std::string message = prettyPrintIntrinsicName(
      builder, loc, "not yet implemented: no math runtime available for '",
      name, "'", funcType);
  aiir::emitError(loc, message);
}

/// Helpers to get function type from arguments and result type.
static aiir::FunctionType getFunctionType(std::optional<aiir::Type> resultType,
                                          llvm::ArrayRef<aiir::Value> arguments,
                                          fir::FirOpBuilder &builder) {
  llvm::SmallVector<aiir::Type> argTypes;
  for (aiir::Value arg : arguments)
    argTypes.push_back(arg.getType());
  llvm::SmallVector<aiir::Type> resTypes;
  if (resultType)
    resTypes.push_back(*resultType);
  return aiir::FunctionType::get(builder.getModule().getContext(), argTypes,
                                 resTypes);
}

/// fir::ExtendedValue to aiir::Value translation layer

fir::ExtendedValue toExtendedValue(aiir::Value val, fir::FirOpBuilder &builder,
                                   aiir::Location loc) {
  assert(val && "optional unhandled here");
  aiir::Type type = val.getType();
  aiir::Value base = val;
  aiir::IndexType indexType = builder.getIndexType();
  llvm::SmallVector<aiir::Value> extents;

  fir::factory::CharacterExprHelper charHelper{builder, loc};
  // FIXME: we may want to allow non character scalar here.
  if (charHelper.isCharacterScalar(type))
    return charHelper.toExtendedValue(val);

  if (auto refType = aiir::dyn_cast<fir::ReferenceType>(type))
    type = refType.getEleTy();

  if (auto arrayType = aiir::dyn_cast<fir::SequenceType>(type)) {
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
      aiir::emitError(loc, "cannot retrieve array extents from type");
  } else if (aiir::isa<fir::BoxType>(type) ||
             aiir::isa<fir::RecordType>(type)) {
    fir::emitFatalError(loc, "not yet implemented: descriptor or derived type");
  }

  if (!extents.empty())
    return fir::ArrayBoxValue{base, extents};
  return base;
}

aiir::Value toValue(const fir::ExtendedValue &val, fir::FirOpBuilder &builder,
                    aiir::Location loc) {
  if (const fir::CharBoxValue *charBox = val.getCharBox()) {
    aiir::Value buffer = charBox->getBuffer();
    auto buffTy = buffer.getType();
    if (aiir::isa<aiir::FunctionType>(buffTy))
      fir::emitFatalError(
          loc, "A character's buffer type cannot be a function type.");
    if (aiir::isa<fir::BoxCharType>(buffTy))
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

static bool isIntrinsicModuleProcedure(llvm::StringRef name) {
  return name.starts_with("c_") || name.starts_with("compiler_") ||
         name.starts_with("ieee_") || name.starts_with("__ppc_");
}

static bool isCoarrayIntrinsic(llvm::StringRef name) {
  return name.starts_with("atomic_") || name.starts_with("co_") ||
         name.contains("image") || name.ends_with("cobound") ||
         name == "team_number";
}

/// Return the generic name of an intrinsic module procedure specific name.
/// Remove any "__builtin_" prefix, and any specific suffix of the form
/// {_[ail]?[0-9]+}*, such as _1 or _a4.
llvm::StringRef genericName(llvm::StringRef specificName) {
  const std::string builtin = "__builtin_";
  llvm::StringRef name = specificName.starts_with(builtin)
                             ? specificName.drop_front(builtin.size())
                             : specificName;
  size_t size = name.size();
  if (isIntrinsicModuleProcedure(name))
    while (isdigit(name[size - 1]))
      while (name[--size] != '_')
        ;
  return name.drop_back(name.size() - size);
}

std::optional<IntrinsicHandlerEntry::RuntimeGeneratorRange>
lookupRuntimeGenerator(llvm::StringRef name, bool isPPCTarget) {
  if (auto range = mathOps.equal_range(name); range.first != range.second)
    return std::make_optional<IntrinsicHandlerEntry::RuntimeGeneratorRange>(
        range);
  // Search ppcMathOps only if targetting PowerPC arch
  if (isPPCTarget)
    if (auto range = checkPPCMathOperationsRange(name);
        range.first != range.second)
      return std::make_optional<IntrinsicHandlerEntry::RuntimeGeneratorRange>(
          range);
  return std::nullopt;
}

std::optional<IntrinsicHandlerEntry>
lookupIntrinsicHandler(fir::FirOpBuilder &builder,
                       llvm::StringRef intrinsicName,
                       std::optional<aiir::Type> resultType) {
  llvm::StringRef name = genericName(intrinsicName);
  if (const IntrinsicHandler *handler = findIntrinsicHandler(name))
    return std::make_optional<IntrinsicHandlerEntry>(handler);
  bool isPPCTarget = fir::getTargetTriple(builder.getModule()).isPPC();
  // If targeting PowerPC, check PPC intrinsic handlers.
  if (isPPCTarget)
    if (const IntrinsicHandler *ppcHandler = findPPCIntrinsicHandler(name))
      return std::make_optional<IntrinsicHandlerEntry>(ppcHandler);
  // TODO: Look for CUDA intrinsic handlers only if CUDA is enabled.
  if (const IntrinsicHandler *cudaHandler = findCUDAIntrinsicHandler(name))
    return std::make_optional<IntrinsicHandlerEntry>(cudaHandler);
  // Subroutines should have a handler.
  if (!resultType)
    return std::nullopt;
  // Try the runtime if no special handler was defined for the
  // intrinsic being called. Maths runtime only has numerical elemental.
  if (auto runtimeGeneratorRange = lookupRuntimeGenerator(name, isPPCTarget))
    return std::make_optional<IntrinsicHandlerEntry>(*runtimeGeneratorRange);
  return std::nullopt;
}

/// Generate a TODO error message for an as yet unimplemented intrinsic.
void crashOnMissingIntrinsic(aiir::Location loc,
                             llvm::StringRef intrinsicName) {
  llvm::StringRef name = genericName(intrinsicName);
  if (isIntrinsicModuleProcedure(name))
    TODO(loc, "intrinsic module procedure: " + llvm::Twine(name));
  else if (isCoarrayIntrinsic(name))
    TODO(loc, "coarray: intrinsic " + llvm::Twine(name));
  else
    TODO(loc, "intrinsic: " + llvm::Twine(name.upper()));
}

template <typename GeneratorType>
fir::ExtendedValue IntrinsicLibrary::genElementalCall(
    GeneratorType generator, llvm::StringRef name, aiir::Type resultType,
    llvm::ArrayRef<fir::ExtendedValue> args, bool outline) {
  llvm::SmallVector<aiir::Value> scalarArgs;
  for (const fir::ExtendedValue &arg : args)
    if (arg.getUnboxed() || arg.getCharBox())
      scalarArgs.emplace_back(fir::getBase(arg));
    else
      fir::emitFatalError(loc, "nonscalar intrinsic argument");
  if (outline)
    return outlineInWrapper(generator, name, resultType, scalarArgs);
  return invokeGenerator(generator, resultType, scalarArgs);
}

template <>
fir::ExtendedValue
IntrinsicLibrary::genElementalCall<IntrinsicLibrary::ExtendedGenerator>(
    ExtendedGenerator generator, llvm::StringRef name, aiir::Type resultType,
    llvm::ArrayRef<fir::ExtendedValue> args, bool outline) {
  for (const fir::ExtendedValue &arg : args) {
    auto *box = arg.getBoxOf<fir::BoxValue>();
    if (!arg.getUnboxed() && !arg.getCharBox() &&
        !(box && (fir::isScalarBoxedRecordType(fir::getBase(*box).getType()) ||
                  fir::isClassStarType(fir::getBase(*box).getType()))))
      fir::emitFatalError(loc, "nonscalar intrinsic argument");
  }
  if (outline)
    return outlineInExtendedWrapper(generator, name, resultType, args);
  return std::invoke(generator, *this, resultType, args);
}

template <>
fir::ExtendedValue
IntrinsicLibrary::genElementalCall<IntrinsicLibrary::SubroutineGenerator>(
    SubroutineGenerator generator, llvm::StringRef name, aiir::Type resultType,
    llvm::ArrayRef<fir::ExtendedValue> args, bool outline) {
  for (const fir::ExtendedValue &arg : args)
    if (!arg.getUnboxed() && !arg.getCharBox())
      // fir::emitFatalError(loc, "nonscalar intrinsic argument");
      crashOnMissingIntrinsic(loc, name);
  if (outline)
    return outlineInExtendedWrapper(generator, name, resultType, args);
  std::invoke(generator, *this, args);
  return aiir::Value();
}

template <>
fir::ExtendedValue
IntrinsicLibrary::genElementalCall<IntrinsicLibrary::DualGenerator>(
    DualGenerator generator, llvm::StringRef name, aiir::Type resultType,
    llvm::ArrayRef<fir::ExtendedValue> args, bool outline) {
  assert(resultType.getImpl() && "expect elemental intrinsic to be functions");

  for (const fir::ExtendedValue &arg : args)
    if (!arg.getUnboxed() && !arg.getCharBox())
      // fir::emitFatalError(loc, "nonscalar intrinsic argument");
      crashOnMissingIntrinsic(loc, name);
  if (outline)
    return outlineInExtendedWrapper(generator, name, resultType, args);

  return std::invoke(generator, *this, std::optional<aiir::Type>{resultType},
                     args);
}

static fir::ExtendedValue
invokeHandler(IntrinsicLibrary::ElementalGenerator generator,
              const IntrinsicHandler &handler,
              std::optional<aiir::Type> resultType,
              llvm::ArrayRef<fir::ExtendedValue> args, bool outline,
              IntrinsicLibrary &lib) {
  assert(resultType && "expect elemental intrinsic to be functions");
  return lib.genElementalCall(generator, handler.name, *resultType, args,
                              outline);
}

static fir::ExtendedValue
invokeHandler(IntrinsicLibrary::ExtendedGenerator generator,
              const IntrinsicHandler &handler,
              std::optional<aiir::Type> resultType,
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
              std::optional<aiir::Type> resultType,
              llvm::ArrayRef<fir::ExtendedValue> args, bool outline,
              IntrinsicLibrary &lib) {
  if (handler.isElemental)
    return lib.genElementalCall(generator, handler.name, aiir::Type{}, args,
                                outline);
  if (outline)
    return lib.outlineInExtendedWrapper(generator, handler.name, resultType,
                                        args);
  std::invoke(generator, lib, args);
  return aiir::Value{};
}

static fir::ExtendedValue
invokeHandler(IntrinsicLibrary::DualGenerator generator,
              const IntrinsicHandler &handler,
              std::optional<aiir::Type> resultType,
              llvm::ArrayRef<fir::ExtendedValue> args, bool outline,
              IntrinsicLibrary &lib) {
  if (handler.isElemental)
    return lib.genElementalCall(generator, handler.name, aiir::Type{}, args,
                                outline);
  if (outline)
    return lib.outlineInExtendedWrapper(generator, handler.name, resultType,
                                        args);

  return std::invoke(generator, lib, resultType, args);
}

static std::pair<fir::ExtendedValue, bool> genIntrinsicCallHelper(
    const IntrinsicHandler *handler, std::optional<aiir::Type> resultType,
    llvm::ArrayRef<fir::ExtendedValue> args, IntrinsicLibrary &lib) {
  assert(handler && "must be set");
  bool outline = handler->outline || outlineAllIntrinsics;
  return {Fortran::common::visit(
              [&](auto &generator) -> fir::ExtendedValue {
                return invokeHandler(generator, *handler, resultType, args,
                                     outline, lib);
              },
              handler->generator),
          lib.resultMustBeFreed};
}

static IntrinsicLibrary::RuntimeCallGenerator getRuntimeCallGeneratorHelper(
    const IntrinsicHandlerEntry::RuntimeGeneratorRange &, aiir::FunctionType,
    fir::FirOpBuilder &, aiir::Location);

static std::pair<fir::ExtendedValue, bool> genIntrinsicCallHelper(
    const IntrinsicHandlerEntry::RuntimeGeneratorRange &range,
    std::optional<aiir::Type> resultType,
    llvm::ArrayRef<fir::ExtendedValue> args, IntrinsicLibrary &lib) {
  assert(resultType.has_value() && "RuntimeGenerator are for functions only");
  assert(range.first != nullptr && "range should not be empty");
  fir::FirOpBuilder &builder = lib.builder;
  aiir::Location loc = lib.loc;
  llvm::StringRef name = range.first->key;
  // FIXME: using toValue to get the type won't work with array arguments.
  llvm::SmallVector<aiir::Value> aiirArgs;
  for (const fir::ExtendedValue &extendedVal : args) {
    aiir::Value val = toValue(extendedVal, builder, loc);
    if (!val)
      // If an absent optional gets there, most likely its handler has just
      // not yet been defined.
      crashOnMissingIntrinsic(loc, name);
    aiirArgs.emplace_back(val);
  }
  aiir::FunctionType soughtFuncType =
      getFunctionType(*resultType, aiirArgs, builder);

  IntrinsicLibrary::RuntimeCallGenerator runtimeCallGenerator =
      getRuntimeCallGeneratorHelper(range, soughtFuncType, builder, loc);
  return {lib.genElementalCall(runtimeCallGenerator, name, *resultType, args,
                               /*outline=*/outlineAllIntrinsics),
          lib.resultMustBeFreed};
}

std::pair<fir::ExtendedValue, bool>
genIntrinsicCall(fir::FirOpBuilder &builder, aiir::Location loc,
                 const IntrinsicHandlerEntry &intrinsic,
                 std::optional<aiir::Type> resultType,
                 llvm::ArrayRef<fir::ExtendedValue> args,
                 Fortran::lower::AbstractConverter *converter) {
  IntrinsicLibrary library{builder, loc, converter};
  return std::visit(
      [&](auto handler) -> auto {
        return genIntrinsicCallHelper(handler, resultType, args, library);
      },
      intrinsic.entry);
}

std::pair<fir::ExtendedValue, bool>
IntrinsicLibrary::genIntrinsicCall(llvm::StringRef specificName,
                                   std::optional<aiir::Type> resultType,
                                   llvm::ArrayRef<fir::ExtendedValue> args) {
  std::optional<IntrinsicHandlerEntry> intrinsic =
      lookupIntrinsicHandler(builder, specificName, resultType);
  if (!intrinsic.has_value())
    crashOnMissingIntrinsic(loc, specificName);
  return std::visit(
      [&](auto handler) -> auto {
        return genIntrinsicCallHelper(handler, resultType, args, *this);
      },
      intrinsic->entry);
}

aiir::Value
IntrinsicLibrary::invokeGenerator(ElementalGenerator generator,
                                  aiir::Type resultType,
                                  llvm::ArrayRef<aiir::Value> args) {
  return std::invoke(generator, *this, resultType, args);
}

aiir::Value
IntrinsicLibrary::invokeGenerator(RuntimeCallGenerator generator,
                                  aiir::Type resultType,
                                  llvm::ArrayRef<aiir::Value> args) {
  return generator(builder, loc, args);
}

aiir::Value
IntrinsicLibrary::invokeGenerator(ExtendedGenerator generator,
                                  aiir::Type resultType,
                                  llvm::ArrayRef<aiir::Value> args) {
  llvm::SmallVector<fir::ExtendedValue> extendedArgs;
  for (aiir::Value arg : args)
    extendedArgs.emplace_back(toExtendedValue(arg, builder, loc));
  auto extendedResult = std::invoke(generator, *this, resultType, extendedArgs);
  return toValue(extendedResult, builder, loc);
}

aiir::Value
IntrinsicLibrary::invokeGenerator(SubroutineGenerator generator,
                                  llvm::ArrayRef<aiir::Value> args) {
  llvm::SmallVector<fir::ExtendedValue> extendedArgs;
  for (aiir::Value arg : args)
    extendedArgs.emplace_back(toExtendedValue(arg, builder, loc));
  std::invoke(generator, *this, extendedArgs);
  return {};
}

aiir::Value
IntrinsicLibrary::invokeGenerator(DualGenerator generator,
                                  llvm::ArrayRef<aiir::Value> args) {
  llvm::SmallVector<fir::ExtendedValue> extendedArgs;
  for (aiir::Value arg : args)
    extendedArgs.emplace_back(toExtendedValue(arg, builder, loc));
  std::invoke(generator, *this, std::optional<aiir::Type>{}, extendedArgs);
  return {};
}

aiir::Value
IntrinsicLibrary::invokeGenerator(DualGenerator generator,
                                  aiir::Type resultType,
                                  llvm::ArrayRef<aiir::Value> args) {
  llvm::SmallVector<fir::ExtendedValue> extendedArgs;
  for (aiir::Value arg : args)
    extendedArgs.emplace_back(toExtendedValue(arg, builder, loc));

  if (resultType.getImpl() == nullptr) {
    // TODO:
    assert(false && "result type is null");
  }

  auto extendedResult = std::invoke(
      generator, *this, std::optional<aiir::Type>{resultType}, extendedArgs);
  return toValue(extendedResult, builder, loc);
}

//===----------------------------------------------------------------------===//
// Intrinsic Procedure Mangling
//===----------------------------------------------------------------------===//

/// Helper to encode type into string for intrinsic procedure names.
/// Note: aiir has Type::dump(ostream) methods but it may add "!" that is not
/// suitable for function names.
static std::string typeToString(aiir::Type t) {
  if (auto refT{aiir::dyn_cast<fir::ReferenceType>(t)})
    return "ref_" + typeToString(refT.getEleTy());
  if (auto i{aiir::dyn_cast<aiir::IntegerType>(t)}) {
    return "i" + std::to_string(i.getWidth());
  }
  if (auto cplx{aiir::dyn_cast<aiir::ComplexType>(t)}) {
    auto eleTy = aiir::cast<aiir::FloatType>(cplx.getElementType());
    return "z" + std::to_string(eleTy.getWidth());
  }
  if (auto f{aiir::dyn_cast<aiir::FloatType>(t)}) {
    return "f" + std::to_string(f.getWidth());
  }
  if (auto logical{aiir::dyn_cast<fir::LogicalType>(t)}) {
    return "l" + std::to_string(logical.getFKind());
  }
  if (auto character{aiir::dyn_cast<fir::CharacterType>(t)}) {
    return "c" + std::to_string(character.getFKind());
  }
  if (auto boxCharacter{aiir::dyn_cast<fir::BoxCharType>(t)}) {
    return "bc" + std::to_string(boxCharacter.getEleTy().getFKind());
  }
  llvm_unreachable("no mangling for type");
}

/// Returns a name suitable to define aiir functions for Fortran intrinsic
/// Procedure. These names are guaranteed to not conflict with user defined
/// procedures. This is needed to implement Fortran generic intrinsics as
/// several aiir functions specialized for the argument types.
/// The result is guaranteed to be distinct for different aiir::FunctionType
/// arguments. The mangling pattern is:
///    fir.<generic name>.<result type>.<arg type>...
/// e.g ACOS(COMPLEX(4)) is mangled as fir.acos.z4.z4
/// For subroutines no result type is return but in order to still provide
/// a unique mangled name, we use "void" as the return type. As in:
///    fir.<generic name>.void.<arg type>...
/// e.g. FREE(INTEGER(4)) is mangled as fir.free.void.i4
static std::string mangleIntrinsicProcedure(llvm::StringRef intrinsic,
                                            aiir::FunctionType funTy) {
  std::string name = "fir.";
  name.append(intrinsic.str()).append(".");
  if (funTy.getNumResults() == 1)
    name.append(typeToString(funTy.getResult(0)));
  else if (funTy.getNumResults() == 0)
    name.append("void");
  else
    llvm_unreachable("more than one result value for function");
  unsigned e = funTy.getNumInputs();
  for (decltype(e) i = 0; i < e; ++i)
    name.append(".").append(typeToString(funTy.getInput(i)));
  return name;
}

template <typename GeneratorType>
aiir::func::FuncOp IntrinsicLibrary::getWrapper(GeneratorType generator,
                                                llvm::StringRef name,
                                                aiir::FunctionType funcType,
                                                bool loadRefArguments) {
  std::string wrapperName = mangleIntrinsicProcedure(name, funcType);
  aiir::func::FuncOp function = builder.getNamedFunction(wrapperName);
  if (!function) {
    // First time this wrapper is needed, build it.
    function = builder.createFunction(loc, wrapperName, funcType);
    function->setAttr("fir.intrinsic", builder.getUnitAttr());
    fir::factory::setInternalLinkage(function);
    function.addEntryBlock();

    // Create local context to emit code into the newly created function
    // This new function is not linked to a source file location, only
    // its calls will be.
    auto localBuilder = std::make_unique<fir::FirOpBuilder>(
        function, builder.getKindMap(), builder.getAIIRSymbolTable());
    localBuilder->setFastMathFlags(builder.getFastMathFlags());
    localBuilder->setInsertionPointToStart(&function.front());
    // Location of code inside wrapper of the wrapper is independent from
    // the location of the intrinsic call.
    aiir::Location localLoc = localBuilder->getUnknownLoc();
    llvm::SmallVector<aiir::Value> localArguments;
    for (aiir::BlockArgument bArg : function.front().getArguments()) {
      auto refType = aiir::dyn_cast<fir::ReferenceType>(bArg.getType());
      if (loadRefArguments && refType) {
        auto loaded = fir::LoadOp::create(*localBuilder, localLoc, bArg);
        localArguments.push_back(loaded);
      } else {
        localArguments.push_back(bArg);
      }
    }

    IntrinsicLibrary localLib{*localBuilder, localLoc};

    if constexpr (std::is_same_v<GeneratorType, SubroutineGenerator>) {
      localLib.invokeGenerator(generator, localArguments);
      aiir::func::ReturnOp::create(*localBuilder, localLoc);
    } else {
      assert(funcType.getNumResults() == 1 &&
             "expect one result for intrinsic function wrapper type");
      aiir::Type resultType = funcType.getResult(0);
      auto result =
          localLib.invokeGenerator(generator, resultType, localArguments);
      aiir::func::ReturnOp::create(*localBuilder, localLoc, result);
    }
  } else {
    // Wrapper was already built, ensure it has the sought type
    assert(function.getFunctionType() == funcType &&
           "conflict between intrinsic wrapper types");
  }
  return function;
}

/// Helpers to detect absent optional (not yet supported in outlining).
bool static hasAbsentOptional(llvm::ArrayRef<aiir::Value> args) {
  for (const aiir::Value &arg : args)
    if (!arg)
      return true;
  return false;
}
bool static hasAbsentOptional(llvm::ArrayRef<fir::ExtendedValue> args) {
  for (const fir::ExtendedValue &arg : args)
    if (!fir::getBase(arg))
      return true;
  return false;
}

template <typename GeneratorType>
aiir::Value
IntrinsicLibrary::outlineInWrapper(GeneratorType generator,
                                   llvm::StringRef name, aiir::Type resultType,
                                   llvm::ArrayRef<aiir::Value> args) {
  if (hasAbsentOptional(args)) {
    // TODO: absent optional in outlining is an issue: we cannot just ignore
    // them. Needs a better interface here. The issue is that we cannot easily
    // tell that a value is optional or not here if it is presents. And if it is
    // absent, we cannot tell what it type should be.
    TODO(loc, "cannot outline call to intrinsic " + llvm::Twine(name) +
                  " with absent optional argument");
  }

  aiir::FunctionType funcType = getFunctionType(resultType, args, builder);
  std::string funcName{name};
  llvm::raw_string_ostream nameOS{funcName};
  if (std::string fmfString{builder.getFastMathFlagsString()};
      !fmfString.empty()) {
    nameOS << '.' << fmfString;
  }
  aiir::func::FuncOp wrapper = getWrapper(generator, funcName, funcType);
  return fir::CallOp::create(builder, loc, wrapper, args).getResult(0);
}

template <typename GeneratorType>
fir::ExtendedValue IntrinsicLibrary::outlineInExtendedWrapper(
    GeneratorType generator, llvm::StringRef name,
    std::optional<aiir::Type> resultType,
    llvm::ArrayRef<fir::ExtendedValue> args) {
  if (hasAbsentOptional(args))
    TODO(loc, "cannot outline call to intrinsic " + llvm::Twine(name) +
                  " with absent optional argument");
  llvm::SmallVector<aiir::Value> aiirArgs;
  for (const auto &extendedVal : args)
    aiirArgs.emplace_back(toValue(extendedVal, builder, loc));
  aiir::FunctionType funcType = getFunctionType(resultType, aiirArgs, builder);
  aiir::func::FuncOp wrapper = getWrapper(generator, name, funcType);
  auto call = fir::CallOp::create(builder, loc, wrapper, aiirArgs);
  if (resultType)
    return toExtendedValue(call.getResult(0), builder, loc);
  // Subroutine calls
  return aiir::Value{};
}

static IntrinsicLibrary::RuntimeCallGenerator getRuntimeCallGeneratorHelper(
    const IntrinsicHandlerEntry::RuntimeGeneratorRange &range,
    aiir::FunctionType soughtFuncType, fir::FirOpBuilder &builder,
    aiir::Location loc) {
  assert(range.first != nullptr && "range should not be empty");
  llvm::StringRef name = range.first->key;
  // Look for a dedicated math operation generator, which
  // normally produces a single AIIR operation implementing
  // the math operation.
  const MathOperation *bestNearMatch = nullptr;
  FunctionDistance bestMatchDistance;
  const MathOperation *mathOp = searchMathOperation(
      builder, range, soughtFuncType, &bestNearMatch, bestMatchDistance);
  if (!mathOp && bestNearMatch) {
    // Use the best near match, optionally issuing an error,
    // if types conversions cause precision loss.
    checkPrecisionLoss(name, soughtFuncType, bestMatchDistance, builder, loc);
    mathOp = bestNearMatch;
  }

  if (!mathOp) {
    std::string nameAndType;
    llvm::raw_string_ostream sstream(nameAndType);
    sstream << name << "\nrequested type: " << soughtFuncType;
    crashOnMissingIntrinsic(loc, nameAndType);
  }

  aiir::FunctionType actualFuncType =
      mathOp->typeGenerator(builder.getContext(), builder);

  assert(actualFuncType.getNumResults() == soughtFuncType.getNumResults() &&
         actualFuncType.getNumInputs() == soughtFuncType.getNumInputs() &&
         actualFuncType.getNumResults() == 1 && "Bad intrinsic match");

  return [actualFuncType, mathOp,
          soughtFuncType](fir::FirOpBuilder &builder, aiir::Location loc,
                          llvm::ArrayRef<aiir::Value> args) {
    llvm::SmallVector<aiir::Value> convertedArguments;
    for (auto [fst, snd] : llvm::zip(actualFuncType.getInputs(), args))
      convertedArguments.push_back(builder.createConvert(loc, fst, snd));
    aiir::Value result = mathOp->funcGenerator(
        builder, loc, *mathOp, actualFuncType, convertedArguments);
    aiir::Type soughtType = soughtFuncType.getResult(0);
    return builder.createConvert(loc, soughtType, result);
  };
}

IntrinsicLibrary::RuntimeCallGenerator
IntrinsicLibrary::getRuntimeCallGenerator(llvm::StringRef name,
                                          aiir::FunctionType soughtFuncType) {
  bool isPPCTarget = fir::getTargetTriple(builder.getModule()).isPPC();
  std::optional<IntrinsicHandlerEntry::RuntimeGeneratorRange> range =
      lookupRuntimeGenerator(name, isPPCTarget);
  if (!range.has_value())
    crashOnMissingIntrinsic(loc, name);
  return getRuntimeCallGeneratorHelper(*range, soughtFuncType, builder, loc);
}

aiir::SymbolRefAttr IntrinsicLibrary::getUnrestrictedIntrinsicSymbolRefAttr(
    llvm::StringRef name, aiir::FunctionType signature) {
  // Unrestricted intrinsics signature follows implicit rules: argument
  // are passed by references. But the runtime versions expect values.
  // So instead of duplicating the runtime, just have the wrappers loading
  // this before calling the code generators.
  bool loadRefArguments = true;
  aiir::func::FuncOp funcOp;
  if (const IntrinsicHandler *handler = findIntrinsicHandler(name))
    funcOp = Fortran::common::visit(
        [&](auto generator) {
          return getWrapper(generator, name, signature, loadRefArguments);
        },
        handler->generator);

  if (!funcOp) {
    llvm::SmallVector<aiir::Type> argTypes;
    for (aiir::Type type : signature.getInputs()) {
      if (auto refType = aiir::dyn_cast<fir::ReferenceType>(type))
        argTypes.push_back(refType.getEleTy());
      else
        argTypes.push_back(type);
    }
    aiir::FunctionType soughtFuncType =
        builder.getFunctionType(argTypes, signature.getResults());
    IntrinsicLibrary::RuntimeCallGenerator rtCallGenerator =
        getRuntimeCallGenerator(name, soughtFuncType);
    funcOp = getWrapper(rtCallGenerator, name, signature, loadRefArguments);
  }

  return aiir::SymbolRefAttr::get(funcOp);
}

fir::ExtendedValue
IntrinsicLibrary::readAndAddCleanUp(fir::MutableBoxValue resultMutableBox,
                                    aiir::Type resultType,
                                    llvm::StringRef intrinsicName) {
  fir::ExtendedValue res =
      fir::factory::genMutableBoxRead(builder, loc, resultMutableBox);
  return res.match(
      [&](const fir::ArrayBoxValue &box) -> fir::ExtendedValue {
        setResultMustBeFreed();
        return box;
      },
      [&](const fir::BoxValue &box) -> fir::ExtendedValue {
        setResultMustBeFreed();
        return box;
      },
      [&](const fir::CharArrayBoxValue &box) -> fir::ExtendedValue {
        setResultMustBeFreed();
        return box;
      },
      [&](const aiir::Value &tempAddr) -> fir::ExtendedValue {
        auto load = fir::LoadOp::create(builder, loc, resultType, tempAddr);
        // Temp can be freed right away since it was loaded.
        fir::FreeMemOp::create(builder, loc, tempAddr);
        return load;
      },
      [&](const fir::CharBoxValue &box) -> fir::ExtendedValue {
        setResultMustBeFreed();
        return box;
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc, "unexpected result for " + intrinsicName);
      });
}

//===----------------------------------------------------------------------===//
// Code generators for the intrinsic
//===----------------------------------------------------------------------===//

aiir::Value IntrinsicLibrary::genRuntimeCall(llvm::StringRef name,
                                             aiir::Type resultType,
                                             llvm::ArrayRef<aiir::Value> args) {
  aiir::FunctionType soughtFuncType =
      getFunctionType(resultType, args, builder);
  return getRuntimeCallGenerator(name, soughtFuncType)(builder, loc, args);
}

aiir::Value IntrinsicLibrary::genConversion(aiir::Type resultType,
                                            llvm::ArrayRef<aiir::Value> args) {
  // There can be an optional kind in second argument.
  assert(args.size() >= 1);
  return builder.convertWithSemantics(loc, resultType, args[0]);
}

// ABORT
void IntrinsicLibrary::genAbort(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 0);
  fir::runtime::genAbort(builder, loc);
}

// ABS
aiir::Value IntrinsicLibrary::genAbs(aiir::Type resultType,
                                     llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);
  aiir::Value arg = args[0];
  aiir::Type type = arg.getType();
  if (fir::isa_real(type) || fir::isa_complex(type)) {
    // Runtime call to fp abs. An alternative would be to use aiir
    // math::AbsFOp but it does not support all fir floating point types.
    return genRuntimeCall("abs", resultType, args);
  }
  if (auto intType = aiir::dyn_cast<aiir::IntegerType>(type)) {
    // At the time of this implementation there is no abs op in aiir.
    // So, implement abs here without branching.
    aiir::Value shift =
        builder.createIntegerConstant(loc, intType, intType.getWidth() - 1);
    auto mask = aiir::arith::ShRSIOp::create(builder, loc, arg, shift);
    auto xored = aiir::arith::XOrIOp::create(builder, loc, arg, mask);
    return aiir::arith::SubIOp::create(builder, loc, xored, mask);
  }
  llvm_unreachable("unexpected type in ABS argument");
}

// ACOSD
aiir::Value IntrinsicLibrary::genAcosd(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  // maps ACOSD to ACOS * 180 / pi
  assert(args.size() == 1);
  aiir::AIIRContext *context = builder.getContext();
  aiir::FunctionType ftype =
      aiir::FunctionType::get(context, {resultType}, {args[0].getType()});
  aiir::Value result =
      getRuntimeCallGenerator("acos", ftype)(builder, loc, {args[0]});
  const llvm::fltSemantics &fltSem =
      llvm::cast<aiir::FloatType>(resultType).getFloatSemantics();
  llvm::APFloat pi = llvm::APFloat(fltSem, llvm::numbers::pis);
  aiir::Value factor = builder.createRealConstant(
      loc, resultType, llvm::APFloat(fltSem, "180.0") / pi);
  return aiir::arith::MulFOp::create(builder, loc, result, factor);
}

// ACOSPI
aiir::Value IntrinsicLibrary::genAcospi(aiir::Type resultType,
                                        llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);
  aiir::AIIRContext *context = builder.getContext();
  aiir::FunctionType ftype =
      aiir::FunctionType::get(context, {resultType}, {args[0].getType()});
  aiir::Value acos = getRuntimeCallGenerator("acos", ftype)(builder, loc, args);
  llvm::APFloat inv_pi =
      llvm::APFloat(llvm::cast<aiir::FloatType>(resultType).getFloatSemantics(),
                    llvm::numbers::inv_pis);
  aiir::Value factor = builder.createRealConstant(loc, resultType, inv_pi);
  return aiir::arith::MulFOp::create(builder, loc, acos, factor);
}

// ADJUSTL & ADJUSTR
template <void (*CallRuntime)(fir::FirOpBuilder &, aiir::Location loc,
                              aiir::Value, aiir::Value)>
fir::ExtendedValue
IntrinsicLibrary::genAdjustRtCall(aiir::Type resultType,
                                  llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  aiir::Value string = builder.createBox(loc, args[0]);
  // Create a mutable fir.box to be passed to the runtime for the result.
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  // Call the runtime -- the runtime will allocate the result.
  CallRuntime(builder, loc, resultIrBox, string);
  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  return readAndAddCleanUp(resultMutableBox, resultType, "ADJUSTL or ADJUSTR");
}

// AIMAG
aiir::Value IntrinsicLibrary::genAimag(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);
  return fir::factory::Complex{builder, loc}.extractComplexPart(
      args[0], /*isImagPart=*/true);
}

// AINT
aiir::Value IntrinsicLibrary::genAint(aiir::Type resultType,
                                      llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() >= 1 && args.size() <= 2);
  // Skip optional kind argument to search the runtime; it is already reflected
  // in result type.
  return genRuntimeCall("aint", resultType, {args[0]});
}

// ALL
fir::ExtendedValue
IntrinsicLibrary::genAll(aiir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 2);
  // Handle required mask argument
  aiir::Value mask = builder.createBox(loc, args[0]);

  fir::BoxValue maskArry = builder.createBox(loc, args[0]);
  int rank = maskArry.rank();
  assert(rank >= 1);

  // Handle optional dim argument
  bool absentDim = isStaticallyAbsent(args[1]);
  aiir::Value dim =
      absentDim ? builder.createIntegerConstant(loc, builder.getIndexType(), 1)
                : fir::getBase(args[1]);

  if (rank == 1 || absentDim)
    return builder.createConvert(loc, resultType,
                                 fir::runtime::genAll(builder, loc, mask, dim));

  // else use the result descriptor AllDim() intrinsic

  // Create mutable fir.box to be passed to the runtime for the result.

  aiir::Type resultArrayType = builder.getVarLenSeqTy(resultType, rank - 1);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genAllDescriptor(builder, loc, resultIrBox, mask, dim);
  return readAndAddCleanUp(resultMutableBox, resultType, "ALL");
}

// ALLOCATED
fir::ExtendedValue
IntrinsicLibrary::genAllocated(aiir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  return args[0].match(
      [&](const fir::MutableBoxValue &x) -> fir::ExtendedValue {
        return fir::factory::genIsAllocatedOrAssociatedTest(builder, loc, x);
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc,
                            "allocated arg not lowered to MutableBoxValue");
      });
}

// ANINT
aiir::Value IntrinsicLibrary::genAnint(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() >= 1 && args.size() <= 2);
  // Skip optional kind argument to search the runtime; it is already reflected
  // in result type.
  return genRuntimeCall("anint", resultType, {args[0]});
}

// ANY
fir::ExtendedValue
IntrinsicLibrary::genAny(aiir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 2);
  // Handle required mask argument
  aiir::Value mask = builder.createBox(loc, args[0]);

  fir::BoxValue maskArry = builder.createBox(loc, args[0]);
  int rank = maskArry.rank();
  assert(rank >= 1);

  // Handle optional dim argument
  bool absentDim = isStaticallyAbsent(args[1]);
  aiir::Value dim =
      absentDim ? builder.createIntegerConstant(loc, builder.getIndexType(), 1)
                : fir::getBase(args[1]);

  if (rank == 1 || absentDim)
    return builder.createConvert(loc, resultType,
                                 fir::runtime::genAny(builder, loc, mask, dim));

  // else use the result descriptor AnyDim() intrinsic

  // Create mutable fir.box to be passed to the runtime for the result.

  aiir::Type resultArrayType = builder.getVarLenSeqTy(resultType, rank - 1);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genAnyDescriptor(builder, loc, resultIrBox, mask, dim);
  return readAndAddCleanUp(resultMutableBox, resultType, "ANY");
}

// ASIND
aiir::Value IntrinsicLibrary::genAsind(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  // maps ASIND to ASIN * 180 / pi
  assert(args.size() == 1);
  aiir::AIIRContext *context = builder.getContext();
  aiir::FunctionType ftype =
      aiir::FunctionType::get(context, {resultType}, {args[0].getType()});
  aiir::Value result =
      getRuntimeCallGenerator("asin", ftype)(builder, loc, {args[0]});
  const llvm::fltSemantics &fltSem =
      llvm::cast<aiir::FloatType>(resultType).getFloatSemantics();
  llvm::APFloat pi = llvm::APFloat(fltSem, llvm::numbers::pis);
  aiir::Value factor = builder.createRealConstant(
      loc, resultType, llvm::APFloat(fltSem, "180.0") / pi);
  return aiir::arith::MulFOp::create(builder, loc, result, factor);
}

// ASINPI
aiir::Value IntrinsicLibrary::genAsinpi(aiir::Type resultType,
                                        llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);
  aiir::AIIRContext *context = builder.getContext();
  aiir::FunctionType ftype =
      aiir::FunctionType::get(context, {resultType}, {args[0].getType()});
  aiir::Value asin = getRuntimeCallGenerator("asin", ftype)(builder, loc, args);
  llvm::APFloat inv_pi =
      llvm::APFloat(llvm::cast<aiir::FloatType>(resultType).getFloatSemantics(),
                    llvm::numbers::inv_pis);
  aiir::Value factor = builder.createRealConstant(loc, resultType, inv_pi);
  return aiir::arith::MulFOp::create(builder, loc, asin, factor);
}

// ATAND, ATAN2D
aiir::Value IntrinsicLibrary::genAtand(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  // assert for: atand(X), atand(Y,X), atan2d(Y,X)
  assert(args.size() >= 1 && args.size() <= 2);

  aiir::AIIRContext *context = builder.getContext();
  aiir::Value atan;

  // atand = atan * 180/pi
  if (args.size() == 2) {
    atan = aiir::math::Atan2Op::create(builder, loc, fir::getBase(args[0]),
                                       fir::getBase(args[1]));
  } else {
    aiir::FunctionType ftype =
        aiir::FunctionType::get(context, {resultType}, {args[0].getType()});
    atan = getRuntimeCallGenerator("atan", ftype)(builder, loc, args);
  }
  const llvm::fltSemantics &fltSem =
      llvm::cast<aiir::FloatType>(resultType).getFloatSemantics();
  llvm::APFloat pi = llvm::APFloat(fltSem, llvm::numbers::pis);
  aiir::Value factor = builder.createRealConstant(
      loc, resultType, llvm::APFloat(fltSem, "180.0") / pi);
  return aiir::arith::MulFOp::create(builder, loc, atan, factor);
}

// ATANPI, ATAN2PI
aiir::Value IntrinsicLibrary::genAtanpi(aiir::Type resultType,
                                        llvm::ArrayRef<aiir::Value> args) {
  // assert for: atanpi(X), atanpi(Y,X), atan2pi(Y,X)
  assert(args.size() >= 1 && args.size() <= 2);

  aiir::Value atan;
  aiir::AIIRContext *context = builder.getContext();

  // atanpi = atan / pi
  if (args.size() == 2) {
    atan = aiir::math::Atan2Op::create(builder, loc, fir::getBase(args[0]),
                                       fir::getBase(args[1]));
  } else {
    aiir::FunctionType ftype =
        aiir::FunctionType::get(context, {resultType}, {args[0].getType()});
    atan = getRuntimeCallGenerator("atan", ftype)(builder, loc, args);
  }
  llvm::APFloat inv_pi =
      llvm::APFloat(llvm::cast<aiir::FloatType>(resultType).getFloatSemantics(),
                    llvm::numbers::inv_pis);
  aiir::Value factor = builder.createRealConstant(loc, resultType, inv_pi);
  return aiir::arith::MulFOp::create(builder, loc, atan, factor);
}

// ASSOCIATED
fir::ExtendedValue
IntrinsicLibrary::genAssociated(aiir::Type resultType,
                                llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  aiir::Type ptrTy = fir::getBase(args[0]).getType();
  if (ptrTy && (fir::isBoxProcAddressType(ptrTy) ||
                aiir::isa<fir::BoxProcType>(ptrTy))) {
    aiir::Value pointerBoxProc =
        fir::isBoxProcAddressType(ptrTy)
            ? fir::LoadOp::create(builder, loc, fir::getBase(args[0]))
            : fir::getBase(args[0]);
    aiir::Value pointerTarget =
        fir::BoxAddrOp::create(builder, loc, pointerBoxProc);
    if (isStaticallyAbsent(args[1]))
      return builder.genIsNotNullAddr(loc, pointerTarget);
    aiir::Value target = fir::getBase(args[1]);
    if (fir::isBoxProcAddressType(target.getType()))
      target = fir::LoadOp::create(builder, loc, target);
    if (aiir::isa<fir::BoxProcType>(target.getType()))
      target = fir::BoxAddrOp::create(builder, loc, target);
    aiir::Type intPtrTy = builder.getIntPtrType();
    aiir::Value pointerInt =
        builder.createConvert(loc, intPtrTy, pointerTarget);
    aiir::Value targetInt = builder.createConvert(loc, intPtrTy, target);
    aiir::Value sameTarget = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::eq, pointerInt, targetInt);
    aiir::Value zero = builder.createIntegerConstant(loc, intPtrTy, 0);
    aiir::Value notNull = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::ne, zero, pointerInt);
    // The not notNull test covers the following two cases:
    // - TARGET is a procedure that is OPTIONAL and absent at runtime.
    // - TARGET is a procedure pointer that is NULL.
    // In both cases, ASSOCIATED should be false if POINTER is NULL.
    return aiir::arith::AndIOp::create(builder, loc, sameTarget, notNull);
  }
  auto *pointer =
      args[0].match([&](const fir::MutableBoxValue &x) { return &x; },
                    [&](const auto &) -> const fir::MutableBoxValue * {
                      fir::emitFatalError(loc, "pointer not a MutableBoxValue");
                    });
  const fir::ExtendedValue &target = args[1];
  if (isStaticallyAbsent(target))
    return fir::factory::genIsAllocatedOrAssociatedTest(builder, loc, *pointer);
  aiir::Value targetBox = builder.createBox(loc, target);
  aiir::Value pointerBoxRef =
      fir::factory::getMutableIRBox(builder, loc, *pointer);
  auto pointerBox = fir::LoadOp::create(builder, loc, pointerBoxRef);
  return fir::runtime::genAssociated(builder, loc, pointerBox, targetBox);
}

// BESSEL_JN
fir::ExtendedValue
IntrinsicLibrary::genBesselJn(aiir::Type resultType,
                              llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2 || args.size() == 3);

  aiir::Value x = fir::getBase(args.back());

  if (args.size() == 2) {
    aiir::Value n = fir::getBase(args[0]);

    return genRuntimeCall("bessel_jn", resultType, {n, x});
  } else {
    aiir::Value n1 = fir::getBase(args[0]);
    aiir::Value n2 = fir::getBase(args[1]);

    aiir::Type intTy = n1.getType();
    aiir::Type floatTy = x.getType();
    aiir::Value zero = builder.createRealZeroConstant(loc, floatTy);
    aiir::Value one = builder.createIntegerConstant(loc, intTy, 1);

    aiir::Type resultArrayType = builder.getVarLenSeqTy(resultType, 1);
    fir::MutableBoxValue resultMutableBox =
        fir::factory::createTempMutableBox(builder, loc, resultArrayType);
    aiir::Value resultBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    aiir::Value cmpXEq0 = aiir::arith::CmpFOp::create(
        builder, loc, aiir::arith::CmpFPredicate::UEQ, x, zero);
    aiir::Value cmpN1LtN2 = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::slt, n1, n2);
    aiir::Value cmpN1EqN2 = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::eq, n1, n2);

    auto genXEq0 = [&]() {
      fir::runtime::genBesselJnX0(builder, loc, floatTy, resultBox, n1, n2);
    };

    auto genN1LtN2 = [&]() {
      // The runtime generates the values in the range using a backward
      // recursion from n2 to n1. (see https://dlmf.nist.gov/10.74.iv and
      // https://dlmf.nist.gov/10.6.E1). When n1 < n2, this requires
      // the values of BESSEL_JN(n2) and BESSEL_JN(n2 - 1) since they
      // are the anchors of the recursion.
      aiir::Value n2_1 = aiir::arith::SubIOp::create(builder, loc, n2, one);
      aiir::Value bn2 = genRuntimeCall("bessel_jn", resultType, {n2, x});
      aiir::Value bn2_1 = genRuntimeCall("bessel_jn", resultType, {n2_1, x});
      fir::runtime::genBesselJn(builder, loc, resultBox, n1, n2, x, bn2, bn2_1);
    };

    auto genN1EqN2 = [&]() {
      // When n1 == n2, only BESSEL_JN(n2) is needed.
      aiir::Value bn2 = genRuntimeCall("bessel_jn", resultType, {n2, x});
      fir::runtime::genBesselJn(builder, loc, resultBox, n1, n2, x, bn2, zero);
    };

    auto genN1GtN2 = [&]() {
      // The standard requires n1 <= n2. However, we still need to allocate
      // a zero-length array and return it when n1 > n2, so we do need to call
      // the runtime function.
      fir::runtime::genBesselJn(builder, loc, resultBox, n1, n2, x, zero, zero);
    };

    auto genN1GeN2 = [&] {
      builder.genIfThenElse(loc, cmpN1EqN2)
          .genThen(genN1EqN2)
          .genElse(genN1GtN2)
          .end();
    };

    auto genXNeq0 = [&]() {
      builder.genIfThenElse(loc, cmpN1LtN2)
          .genThen(genN1LtN2)
          .genElse(genN1GeN2)
          .end();
    };

    builder.genIfThenElse(loc, cmpXEq0)
        .genThen(genXEq0)
        .genElse(genXNeq0)
        .end();
    return readAndAddCleanUp(resultMutableBox, resultType, "BESSEL_JN");
  }
}

// BESSEL_YN
fir::ExtendedValue
IntrinsicLibrary::genBesselYn(aiir::Type resultType,
                              llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2 || args.size() == 3);

  aiir::Value x = fir::getBase(args.back());

  if (args.size() == 2) {
    aiir::Value n = fir::getBase(args[0]);

    return genRuntimeCall("bessel_yn", resultType, {n, x});
  } else {
    aiir::Value n1 = fir::getBase(args[0]);
    aiir::Value n2 = fir::getBase(args[1]);

    aiir::Type floatTy = x.getType();
    aiir::Type intTy = n1.getType();
    aiir::Value zero = builder.createRealZeroConstant(loc, floatTy);
    aiir::Value one = builder.createIntegerConstant(loc, intTy, 1);

    aiir::Type resultArrayType = builder.getVarLenSeqTy(resultType, 1);
    fir::MutableBoxValue resultMutableBox =
        fir::factory::createTempMutableBox(builder, loc, resultArrayType);
    aiir::Value resultBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    aiir::Value cmpXEq0 = aiir::arith::CmpFOp::create(
        builder, loc, aiir::arith::CmpFPredicate::UEQ, x, zero);
    aiir::Value cmpN1LtN2 = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::slt, n1, n2);
    aiir::Value cmpN1EqN2 = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::eq, n1, n2);

    auto genXEq0 = [&]() {
      fir::runtime::genBesselYnX0(builder, loc, floatTy, resultBox, n1, n2);
    };

    auto genN1LtN2 = [&]() {
      // The runtime generates the values in the range using a forward
      // recursion from n1 to n2. (see https://dlmf.nist.gov/10.74.iv and
      // https://dlmf.nist.gov/10.6.E1). When n1 < n2, this requires
      // the values of BESSEL_YN(n1) and BESSEL_YN(n1 + 1) since they
      // are the anchors of the recursion.
      aiir::Value n1_1 = aiir::arith::AddIOp::create(builder, loc, n1, one);
      aiir::Value bn1 = genRuntimeCall("bessel_yn", resultType, {n1, x});
      aiir::Value bn1_1 = genRuntimeCall("bessel_yn", resultType, {n1_1, x});
      fir::runtime::genBesselYn(builder, loc, resultBox, n1, n2, x, bn1, bn1_1);
    };

    auto genN1EqN2 = [&]() {
      // When n1 == n2, only BESSEL_YN(n1) is needed.
      aiir::Value bn1 = genRuntimeCall("bessel_yn", resultType, {n1, x});
      fir::runtime::genBesselYn(builder, loc, resultBox, n1, n2, x, bn1, zero);
    };

    auto genN1GtN2 = [&]() {
      // The standard requires n1 <= n2. However, we still need to allocate
      // a zero-length array and return it when n1 > n2, so we do need to call
      // the runtime function.
      fir::runtime::genBesselYn(builder, loc, resultBox, n1, n2, x, zero, zero);
    };

    auto genN1GeN2 = [&] {
      builder.genIfThenElse(loc, cmpN1EqN2)
          .genThen(genN1EqN2)
          .genElse(genN1GtN2)
          .end();
    };

    auto genXNeq0 = [&]() {
      builder.genIfThenElse(loc, cmpN1LtN2)
          .genThen(genN1LtN2)
          .genElse(genN1GeN2)
          .end();
    };

    builder.genIfThenElse(loc, cmpXEq0)
        .genThen(genXEq0)
        .genElse(genXNeq0)
        .end();
    return readAndAddCleanUp(resultMutableBox, resultType, "BESSEL_YN");
  }
}

// BGE, BGT, BLE, BLT
template <aiir::arith::CmpIPredicate pred>
aiir::Value
IntrinsicLibrary::genBitwiseCompare(aiir::Type resultType,
                                    llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);

  aiir::Value arg0 = args[0];
  aiir::Value arg1 = args[1];
  aiir::Type arg0Ty = arg0.getType();
  aiir::Type arg1Ty = arg1.getType();
  int bits0 = arg0Ty.getIntOrFloatBitWidth();
  int bits1 = arg1Ty.getIntOrFloatBitWidth();

  // Arguments do not have to be of the same integer type. However, if neither
  // of the arguments is a BOZ literal, then the shorter of the two needs
  // to be converted to the longer by zero-extending (not sign-extending)
  // to the left [Fortran 2008, 13.3.2].
  //
  // In the case of BOZ literals, the standard describes zero-extension or
  // truncation depending on the kind of the result [Fortran 2008, 13.3.3].
  // However, that seems to be relevant for the case where the type of the
  // result must match the type of the BOZ literal. That is not the case for
  // these intrinsics, so, again, zero-extend to the larger type.
  int widest = bits0 > bits1 ? bits0 : bits1;
  aiir::Type signlessType =
      aiir::IntegerType::get(builder.getContext(), widest,
                             aiir::IntegerType::SignednessSemantics::Signless);
  if (arg0Ty.isUnsignedInteger())
    arg0 = builder.createConvert(loc, signlessType, arg0);
  else if (bits0 < widest)
    arg0 = aiir::arith::ExtUIOp::create(builder, loc, signlessType, arg0);
  if (arg1Ty.isUnsignedInteger())
    arg1 = builder.createConvert(loc, signlessType, arg1);
  else if (bits1 < widest)
    arg1 = aiir::arith::ExtUIOp::create(builder, loc, signlessType, arg1);
  return aiir::arith::CmpIOp::create(builder, loc, pred, arg0, arg1);
}

// BTEST
aiir::Value IntrinsicLibrary::genBtest(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  // A conformant BTEST(I,POS) call satisfies:
  //     POS >= 0
  //     POS < BIT_SIZE(I)
  // Return:  (I >> POS) & 1
  assert(args.size() == 2);
  aiir::Value word = args[0];
  aiir::Type signlessType = aiir::IntegerType::get(
      builder.getContext(), word.getType().getIntOrFloatBitWidth(),
      aiir::IntegerType::SignednessSemantics::Signless);
  if (word.getType().isUnsignedInteger())
    word = builder.createConvert(loc, signlessType, word);
  aiir::Value shiftCount = builder.createConvert(loc, signlessType, args[1]);
  aiir::Value shifted =
      aiir::arith::ShRUIOp::create(builder, loc, word, shiftCount);
  aiir::Value one = builder.createIntegerConstant(loc, signlessType, 1);
  aiir::Value bit = aiir::arith::AndIOp::create(builder, loc, shifted, one);
  return builder.createConvert(loc, resultType, bit);
}

static aiir::Value getAddrFromBox(fir::FirOpBuilder &builder,
                                  aiir::Location loc, fir::ExtendedValue arg,
                                  bool isFunc) {
  aiir::Value argValue = fir::getBase(arg);
  aiir::Value addr{nullptr};
  if (isFunc) {
    auto funcTy = aiir::cast<fir::BoxProcType>(argValue.getType()).getEleTy();
    addr = fir::BoxAddrOp::create(builder, loc, funcTy, argValue);
  } else {
    const auto *box = arg.getBoxOf<fir::BoxValue>();
    addr = fir::BoxAddrOp::create(builder, loc, box->getMemTy(),
                                  fir::getBase(*box));
  }
  return addr;
}

static void clocDeviceArgRewrite(fir::ExtendedValue arg) {
  // Special case for device address in c_loc.
  if (auto emboxOp = aiir::dyn_cast_or_null<fir::EmboxOp>(
          fir::getBase(arg).getDefiningOp()))
    if (auto declareOp = aiir::dyn_cast_or_null<hlfir::DeclareOp>(
            emboxOp.getMemref().getDefiningOp()))
      if (declareOp.getDataAttr() &&
          declareOp.getDataAttr() == cuf::DataAttribute::Device)
        emboxOp.getMemrefMutable().assign(declareOp.getMemref());
}

static fir::ExtendedValue
genCLocOrCFunLoc(fir::FirOpBuilder &builder, aiir::Location loc,
                 aiir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args,
                 bool isFunc = false, bool isDevLoc = false) {
  assert(args.size() == 1);
  clocDeviceArgRewrite(args[0]);
  aiir::Value res = fir::AllocaOp::create(builder, loc, resultType);
  aiir::Value resAddr;
  if (isDevLoc)
    resAddr = fir::factory::genCDevPtrAddr(builder, loc, res, resultType);
  else
    resAddr = fir::factory::genCPtrOrCFunptrAddr(builder, loc, res, resultType);
  assert(fir::isa_box_type(fir::getBase(args[0]).getType()) &&
         "argument must have been lowered to box type");
  aiir::Value argAddr = getAddrFromBox(builder, loc, args[0], isFunc);
  aiir::Value argAddrVal = builder.createConvert(
      loc, fir::unwrapRefType(resAddr.getType()), argAddr);
  fir::StoreOp::create(builder, loc, argAddrVal, resAddr);
  return res;
}

/// C_ASSOCIATED
static fir::ExtendedValue
genCAssociated(fir::FirOpBuilder &builder, aiir::Location loc,
               aiir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  aiir::Value cPtr1 = fir::getBase(args[0]);
  aiir::Value cPtrVal1 =
      fir::factory::genCPtrOrCFunptrValue(builder, loc, cPtr1);
  aiir::Value zero = builder.createIntegerConstant(loc, cPtrVal1.getType(), 0);
  aiir::Value res = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::ne, cPtrVal1, zero);

  if (isStaticallyPresent(args[1])) {
    aiir::Type i1Ty = builder.getI1Type();
    aiir::Value cPtr2 = fir::getBase(args[1]);
    aiir::Value isDynamicallyAbsent = builder.genIsNullAddr(loc, cPtr2);
    res =
        builder
            .genIfOp(loc, {i1Ty}, isDynamicallyAbsent, /*withElseRegion=*/true)
            .genThen([&]() { fir::ResultOp::create(builder, loc, res); })
            .genElse([&]() {
              aiir::Value cPtrVal2 =
                  fir::factory::genCPtrOrCFunptrValue(builder, loc, cPtr2);
              aiir::Value cmpVal = aiir::arith::CmpIOp::create(
                  builder, loc, aiir::arith::CmpIPredicate::eq, cPtrVal1,
                  cPtrVal2);
              aiir::Value newRes =
                  aiir::arith::AndIOp::create(builder, loc, res, cmpVal);
              fir::ResultOp::create(builder, loc, newRes);
            })
            .getResults()[0];
  }
  return builder.createConvert(loc, resultType, res);
}

/// C_ASSOCIATED (C_FUNPTR [, C_FUNPTR])
fir::ExtendedValue IntrinsicLibrary::genCAssociatedCFunPtr(
    aiir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args) {
  return genCAssociated(builder, loc, resultType, args);
}

/// C_ASSOCIATED (C_PTR [, C_PTR])
fir::ExtendedValue
IntrinsicLibrary::genCAssociatedCPtr(aiir::Type resultType,
                                     llvm::ArrayRef<fir::ExtendedValue> args) {
  return genCAssociated(builder, loc, resultType, args);
}

// C_DEVLOC
fir::ExtendedValue
IntrinsicLibrary::genCDevLoc(aiir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  return genCLocOrCFunLoc(builder, loc, resultType, args, /*isFunc=*/false,
                          /*isDevLoc=*/true);
}

// C_F_POINTER
void IntrinsicLibrary::genCFPointer(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  // Handle CPTR argument
  // Get the value of the C address or the result of a reference to C_LOC.
  aiir::Value cPtr = fir::getBase(args[0]);
  aiir::Value cPtrAddrVal =
      fir::factory::genCPtrOrCFunptrValue(builder, loc, cPtr);

  // Handle FPTR argument
  const auto *fPtr = args[1].getBoxOf<fir::MutableBoxValue>();
  assert(fPtr && "FPTR must be a pointer");

  auto getCPtrExtVal = [&](fir::MutableBoxValue box) -> fir::ExtendedValue {
    aiir::Value addr =
        builder.createConvert(loc, fPtr->getMemTy(), cPtrAddrVal);
    aiir::SmallVector<aiir::Value> extents;
    aiir::SmallVector<aiir::Value> lbounds;
    if (box.hasRank()) {
      assert(isStaticallyPresent(args[2]) &&
             "FPTR argument must be an array if SHAPE argument exists");

      // Handle and unpack SHAPE argument
      aiir::Value shape = fir::getBase(args[2]);
      int arrayRank = box.rank();
      aiir::Type shapeElementType =
          fir::unwrapSequenceType(fir::unwrapPassByRefType(shape.getType()));
      aiir::Type idxType = builder.getIndexType();
      for (int i = 0; i < arrayRank; ++i) {
        aiir::Value index = builder.createIntegerConstant(loc, idxType, i);
        aiir::Value var = fir::CoordinateOp::create(
            builder, loc, builder.getRefType(shapeElementType), shape, index);
        aiir::Value load = fir::LoadOp::create(builder, loc, var);
        extents.push_back(builder.createConvert(loc, idxType, load));
      }

      // Handle and unpack LOWER argument if present
      if (isStaticallyPresent(args[3])) {
        aiir::Value lower = fir::getBase(args[3]);
        aiir::Type lowerElementType =
            fir::unwrapSequenceType(fir::unwrapPassByRefType(lower.getType()));
        for (int i = 0; i < arrayRank; ++i) {
          aiir::Value index = builder.createIntegerConstant(loc, idxType, i);
          aiir::Value var = fir::CoordinateOp::create(
              builder, loc, builder.getRefType(lowerElementType), lower, index);
          aiir::Value load = fir::LoadOp::create(builder, loc, var);
          lbounds.push_back(builder.createConvert(loc, idxType, load));
        }
      }
    }
    if (box.isCharacter()) {
      aiir::Value len = box.nonDeferredLenParams()[0];
      if (box.hasRank())
        return fir::CharArrayBoxValue{addr, len, extents, lbounds};
      return fir::CharBoxValue{addr, len};
    }
    if (box.isDerivedWithLenParameters())
      TODO(loc, "get length parameters of derived type");
    if (box.hasRank())
      return fir::ArrayBoxValue{addr, extents, lbounds};
    return addr;
  };

  fir::factory::associateMutableBox(builder, loc, *fPtr, getCPtrExtVal(*fPtr),
                                    /*lbounds=*/aiir::ValueRange{});

  // If the pointer is a registered CUDA fortran variable, the descriptor needs
  // to be synced.
  if (auto declare = aiir::dyn_cast_or_null<hlfir::DeclareOp>(
          fPtr->getAddr().getDefiningOp()))
    if (declare.getMemref().getDefiningOp() &&
        aiir::isa<fir::AddrOfOp>(declare.getMemref().getDefiningOp()))
      if (cuf::isRegisteredDeviceAttr(declare.getDataAttr()) &&
          !cuf::isCUDADeviceContext(builder.getRegion()))
        fir::runtime::cuda::genSyncGlobalDescriptor(builder, loc,
                                                    declare.getMemref());
}

// C_F_PROCPOINTER
void IntrinsicLibrary::genCFProcPointer(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  aiir::Value cptr =
      fir::factory::genCPtrOrCFunptrValue(builder, loc, fir::getBase(args[0]));
  aiir::Value fptr = fir::getBase(args[1]);
  auto boxProcType =
      aiir::cast<fir::BoxProcType>(fir::unwrapRefType(fptr.getType()));
  aiir::Value cptrCast =
      builder.createConvert(loc, boxProcType.getEleTy(), cptr);
  aiir::Value cptrBox =
      fir::EmboxProcOp::create(builder, loc, boxProcType, cptrCast);
  fir::StoreOp::create(builder, loc, cptrBox, fptr);
}

// C_F_STRPOINTER
void IntrinsicLibrary::genCFStrPointer(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);

  aiir::Value cStrAddr;
  aiir::Value strLen;

  const aiir::Value firstArg = fir::getBase(args[0]);
  const aiir::Type firstArgType = fir::unwrapRefType(firstArg.getType());
  const bool isCstrptr = aiir::isa<fir::RecordType>(firstArgType);

  if (isCstrptr) {
    // CSTRPTR form: Extract address from C_PTR
    cStrAddr = fir::factory::genCPtrOrCFunptrValue(builder, loc, firstArg);

    assert(isStaticallyPresent(args[2]));
    aiir::Value nchars = fir::getBase(args[2]);
    if (fir::isa_ref_type(nchars.getType())) {
      strLen = fir::LoadOp::create(builder, loc, nchars);
    } else {
      strLen = nchars;
    }
  } else {
    // CSTRARRAY form: Get address from CHARACTER array
    if (const auto boxCharTy =
            aiir::dyn_cast<fir::BoxCharType>(firstArg.getType())) {
      const auto charTy = aiir::cast<fir::CharacterType>(boxCharTy.getEleTy());
      const auto addrTy = builder.getRefType(charTy);
      auto unboxed = fir::UnboxCharOp::create(
          builder, loc, aiir::TypeRange{addrTy, builder.getIndexType()},
          firstArg);
      cStrAddr = unboxed.getResult(0);
    } else if (aiir::isa<fir::BoxType>(firstArg.getType())) {
      cStrAddr = fir::BoxAddrOp::create(builder, loc, firstArg);
    } else {
      cStrAddr = firstArg;
    }

    // Handle optional NCHARS argument
    if (isStaticallyPresent(args[2])) {
      aiir::Value nchars = fir::getBase(args[2]);
      if (fir::isa_ref_type(nchars.getType())) {
        strLen = fir::LoadOp::create(builder, loc, nchars);
      } else {
        strLen = nchars;
      }
    } else {
      const aiir::Type i8PtrTy = builder.getRefType(builder.getIntegerType(8));
      const aiir::Value strPtr = builder.createConvert(loc, i8PtrTy, cStrAddr);

      const aiir::Type i64Ty = builder.getIntegerType(64);
      const aiir::FunctionType strlenType =
          aiir::FunctionType::get(builder.getContext(), {i8PtrTy}, {i64Ty});

      aiir::func::FuncOp strlenFunc = builder.getNamedFunction("strlen");
      if (!strlenFunc) {
        strlenFunc = builder.createFunction(loc, "strlen", strlenType);
        strlenFunc->setAttr(
            fir::getSymbolAttrName(),
            aiir::StringAttr::get(builder.getContext(), "strlen"));
      }
      auto call = fir::CallOp::create(builder, loc, strlenFunc, {strPtr});
      strLen = call.getResult(0);
    }
  }

  // Handle FSTRPTR (second argument)
  const auto *fStrPtr = args[1].getBoxOf<fir::MutableBoxValue>();
  assert(fStrPtr && "FSTRPTR must be a pointer");

  const aiir::Value lenIdx =
      builder.createConvert(loc, builder.getIndexType(), strLen);

  const aiir::Type charPtrType = fir::PointerType::get(fir::CharacterType::get(
      builder.getContext(), 1, fir::CharacterType::unknownLen()));
  const aiir::Value charPtr = builder.createConvert(loc, charPtrType, cStrAddr);

  const fir::CharBoxValue charBox{charPtr, lenIdx};
  fir::factory::associateMutableBox(builder, loc, *fStrPtr, charBox,
                                    /*lbounds=*/aiir::ValueRange{});

  // CUDA synchronization if needed
  if (auto declare = aiir::dyn_cast_or_null<hlfir::DeclareOp>(
          fStrPtr->getAddr().getDefiningOp()))
    if (declare.getMemref().getDefiningOp() &&
        aiir::isa<fir::AddrOfOp>(declare.getMemref().getDefiningOp()))
      if (cuf::isRegisteredDeviceAttr(declare.getDataAttr()) &&
          !cuf::isCUDADeviceContext(builder.getRegion()))
        fir::runtime::cuda::genSyncGlobalDescriptor(builder, loc,
                                                    declare.getMemref());
}

// C_FUNLOC
fir::ExtendedValue
IntrinsicLibrary::genCFunLoc(aiir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  return genCLocOrCFunLoc(builder, loc, resultType, args, /*isFunc=*/true);
}

// C_LOC
fir::ExtendedValue
IntrinsicLibrary::genCLoc(aiir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  return genCLocOrCFunLoc(builder, loc, resultType, args);
}

// C_PTR_EQ and C_PTR_NE
template <aiir::arith::CmpIPredicate pred>
fir::ExtendedValue
IntrinsicLibrary::genCPtrCompare(aiir::Type resultType,
                                 llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  aiir::Value cPtr1 = fir::getBase(args[0]);
  aiir::Value cPtrVal1 =
      fir::factory::genCPtrOrCFunptrValue(builder, loc, cPtr1);
  aiir::Value cPtr2 = fir::getBase(args[1]);
  aiir::Value cPtrVal2 =
      fir::factory::genCPtrOrCFunptrValue(builder, loc, cPtr2);
  aiir::Value cmp =
      aiir::arith::CmpIOp::create(builder, loc, pred, cPtrVal1, cPtrVal2);
  return builder.createConvert(loc, resultType, cmp);
}

// CEILING
aiir::Value IntrinsicLibrary::genCeiling(aiir::Type resultType,
                                         llvm::ArrayRef<aiir::Value> args) {
  // Optional KIND argument.
  assert(args.size() >= 1);
  aiir::Value arg = args[0];
  // Use ceil that is not an actual Fortran intrinsic but that is
  // an llvm intrinsic that does the same, but return a floating
  // point.
  aiir::Value ceil = genRuntimeCall("ceil", arg.getType(), {arg});
  return builder.createConvert(loc, resultType, ceil);
}

// CHAR
fir::ExtendedValue
IntrinsicLibrary::genChar(aiir::Type type,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  // Optional KIND argument.
  assert(args.size() >= 1);
  const aiir::Value *arg = args[0].getUnboxed();
  // expect argument to be a scalar integer
  if (!arg)
    aiir::emitError(loc, "CHAR intrinsic argument not unboxed");
  fir::factory::CharacterExprHelper helper{builder, loc};
  fir::CharacterType::KindTy kind = helper.getCharacterType(type).getFKind();
  aiir::Value cast = helper.createSingletonFromCode(*arg, kind);
  aiir::Value len =
      builder.createIntegerConstant(loc, builder.getCharacterLengthType(), 1);
  return fir::CharBoxValue{cast, len};
}

// CHDIR
fir::ExtendedValue
IntrinsicLibrary::genChdir(std::optional<aiir::Type> resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  assert((args.size() == 1 && resultType.has_value()) ||
         (args.size() >= 1 && !resultType.has_value()));
  aiir::Value name = fir::getBase(args[0]);
  aiir::Value status = fir::runtime::genChdir(builder, loc, name);

  if (resultType.has_value()) {
    return status;
  } else {
    // Subroutine form, store status and return none.
    if (!isStaticallyAbsent(args[1])) {
      aiir::Value statusAddr = fir::getBase(args[1]);
      statusAddr.dump();
      aiir::Value statusIsPresentAtRuntime =
          builder.genIsNotNullAddr(loc, statusAddr);
      builder.genIfThen(loc, statusIsPresentAtRuntime)
          .genThen([&]() {
            builder.createStoreWithConvert(loc, status, statusAddr);
          })
          .end();
    }
  }

  return {};
}

// CMPLX
aiir::Value IntrinsicLibrary::genCmplx(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() >= 1);
  fir::factory::Complex complexHelper(builder, loc);
  aiir::Type partType = complexHelper.getComplexPartType(resultType);
  aiir::Value real = builder.createConvert(loc, partType, args[0]);
  aiir::Value imag = isStaticallyAbsent(args, 1)
                         ? builder.createRealZeroConstant(loc, partType)
                         : builder.createConvert(loc, partType, args[1]);
  return fir::factory::Complex{builder, loc}.createComplex(resultType, real,
                                                           imag);
}

// CO_BROADCAST
void IntrinsicLibrary::genCoBroadcast(llvm::ArrayRef<fir::ExtendedValue> args) {
  converter->checkCoarrayEnabled();
  assert(args.size() == 4);
  mif::CoBroadcastOp::create(builder, loc, fir::getBase(args[0]),
                             /*sourceImage*/ fir::getBase(args[1]),
                             /*status*/ fir::getBase(args[2]),
                             /*errmsg*/ fir::getBase(args[3]));
}

// CO_MAX
void IntrinsicLibrary::genCoMax(llvm::ArrayRef<fir::ExtendedValue> args) {
  converter->checkCoarrayEnabled();
  assert(args.size() == 4);
  mif::CoMaxOp::create(builder, loc, fir::getBase(args[0]),
                       /*resultImage*/ fir::getBase(args[1]),
                       /*status*/ fir::getBase(args[2]),
                       /*errmsg*/ fir::getBase(args[3]));
}

// CO_MIN
void IntrinsicLibrary::genCoMin(llvm::ArrayRef<fir::ExtendedValue> args) {
  converter->checkCoarrayEnabled();
  assert(args.size() == 4);
  mif::CoMinOp::create(builder, loc, fir::getBase(args[0]),
                       /*resultImage*/ fir::getBase(args[1]),
                       /*status*/ fir::getBase(args[2]),
                       /*errmsg*/ fir::getBase(args[3]));
}

// CO_SUM
void IntrinsicLibrary::genCoSum(llvm::ArrayRef<fir::ExtendedValue> args) {
  converter->checkCoarrayEnabled();
  assert(args.size() == 4);
  mif::CoSumOp::create(builder, loc, fir::getBase(args[0]),
                       /*resultImage*/ fir::getBase(args[1]),
                       /*status*/ fir::getBase(args[2]),
                       /*errmsg*/ fir::getBase(args[3]));
}

// COMMAND_ARGUMENT_COUNT
fir::ExtendedValue IntrinsicLibrary::genCommandArgumentCount(
    aiir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 0);
  assert(resultType == builder.getDefaultIntegerType() &&
         "result type is not default integer kind type");
  return builder.createConvert(
      loc, resultType, fir::runtime::genCommandArgumentCount(builder, loc));
  ;
}

// CONJG
aiir::Value IntrinsicLibrary::genConjg(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);
  if (resultType != args[0].getType())
    llvm_unreachable("argument type mismatch");

  aiir::Value cplx = args[0];
  auto imag = fir::factory::Complex{builder, loc}.extractComplexPart(
      cplx, /*isImagPart=*/true);
  auto negImag = aiir::arith::NegFOp::create(builder, loc, imag);
  return fir::factory::Complex{builder, loc}.insertComplexPart(
      cplx, negImag, /*isImagPart=*/true);
}

// COSD
aiir::Value IntrinsicLibrary::genCosd(aiir::Type resultType,
                                      llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);
  aiir::AIIRContext *context = builder.getContext();
  aiir::FunctionType ftype =
      aiir::FunctionType::get(context, {resultType}, {args[0].getType()});
  const llvm::fltSemantics &fltSem =
      llvm::cast<aiir::FloatType>(resultType).getFloatSemantics();
  llvm::APFloat pi = llvm::APFloat(fltSem, llvm::numbers::pis);
  aiir::Value factor = builder.createRealConstant(
      loc, resultType, pi / llvm::APFloat(fltSem, "180.0"));
  aiir::Value arg = aiir::arith::MulFOp::create(builder, loc, args[0], factor);
  return getRuntimeCallGenerator("cos", ftype)(builder, loc, {arg});
}

// COSPI
aiir::Value IntrinsicLibrary::genCospi(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);
  aiir::AIIRContext *context = builder.getContext();
  aiir::FunctionType ftype =
      aiir::FunctionType::get(context, {resultType}, {args[0].getType()});
  llvm::APFloat pi =
      llvm::APFloat(llvm::cast<aiir::FloatType>(resultType).getFloatSemantics(),
                    llvm::numbers::pis);
  aiir::Value factor = builder.createRealConstant(loc, resultType, pi);
  aiir::Value arg = aiir::arith::MulFOp::create(builder, loc, args[0], factor);
  return getRuntimeCallGenerator("cos", ftype)(builder, loc, {arg});
}

// COUNT
fir::ExtendedValue
IntrinsicLibrary::genCount(aiir::Type resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);

  // Handle mask argument
  fir::BoxValue mask = builder.createBox(loc, args[0]);
  unsigned maskRank = mask.rank();

  assert(maskRank > 0);

  // Handle optional dim argument
  bool absentDim = isStaticallyAbsent(args[1]);
  aiir::Value dim =
      absentDim ? builder.createIntegerConstant(loc, builder.getIndexType(), 0)
                : fir::getBase(args[1]);

  if (absentDim || maskRank == 1) {
    // Result is scalar if no dim argument or mask is rank 1.
    // So, call specialized Count runtime routine.
    return builder.createConvert(
        loc, resultType,
        fir::runtime::genCount(builder, loc, fir::getBase(mask), dim));
  }

  // Call general CountDim runtime routine.

  // Handle optional kind argument
  bool absentKind = isStaticallyAbsent(args[2]);
  aiir::Value kind = absentKind ? builder.createIntegerConstant(
                                      loc, builder.getIndexType(),
                                      builder.getKindMap().defaultIntegerKind())
                                : fir::getBase(args[2]);

  // Create mutable fir.box to be passed to the runtime for the result.
  aiir::Type type = builder.getVarLenSeqTy(resultType, maskRank - 1);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, type);

  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genCountDim(builder, loc, resultIrBox, fir::getBase(mask), dim,
                            kind);
  // Handle cleanup of allocatable result descriptor and return
  return readAndAddCleanUp(resultMutableBox, resultType, "COUNT");
}

// CPU_TIME
void IntrinsicLibrary::genCpuTime(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  const aiir::Value *arg = args[0].getUnboxed();
  assert(arg && "nonscalar cpu_time argument");
  aiir::Value res1 = fir::runtime::genCpuTime(builder, loc);
  aiir::Value res2 =
      builder.createConvert(loc, fir::dyn_cast_ptrEleTy(arg->getType()), res1);
  fir::StoreOp::create(builder, loc, res2, *arg);
}

// CSHIFT
fir::ExtendedValue
IntrinsicLibrary::genCshift(aiir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);

  // Handle required ARRAY argument
  fir::BoxValue arrayBox = builder.createBox(loc, args[0]);
  aiir::Value array = fir::getBase(arrayBox);
  unsigned arrayRank = arrayBox.rank();

  // Create mutable fir.box to be passed to the runtime for the result.
  aiir::Type resultArrayType = builder.getVarLenSeqTy(resultType, arrayRank);
  fir::MutableBoxValue resultMutableBox = fir::factory::createTempMutableBox(
      builder, loc, resultArrayType, {},
      fir::isPolymorphicType(array.getType()) ? array : aiir::Value{});
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  if (arrayRank == 1) {
    // Vector case
    // Handle required SHIFT argument as a scalar
    const aiir::Value *shiftAddr = args[1].getUnboxed();
    assert(shiftAddr && "nonscalar CSHIFT argument");
    auto shift = fir::LoadOp::create(builder, loc, *shiftAddr);

    fir::runtime::genCshiftVector(builder, loc, resultIrBox, array, shift);
  } else {
    // Non-vector case
    // Handle required SHIFT argument as an array
    aiir::Value shift = builder.createBox(loc, args[1]);

    // Handle optional DIM argument
    aiir::Value dim =
        isStaticallyAbsent(args[2])
            ? builder.createIntegerConstant(loc, builder.getIndexType(), 1)
            : fir::getBase(args[2]);
    fir::runtime::genCshift(builder, loc, resultIrBox, array, shift, dim);
  }
  return readAndAddCleanUp(resultMutableBox, resultType, "CSHIFT");
}

// DATE_AND_TIME
void IntrinsicLibrary::genDateAndTime(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4 && "date_and_time has 4 args");
  llvm::SmallVector<std::optional<fir::CharBoxValue>> charArgs(3);
  for (unsigned i = 0; i < 3; ++i)
    if (const fir::CharBoxValue *charBox = args[i].getCharBox())
      charArgs[i] = *charBox;

  aiir::Value values = fir::getBase(args[3]);
  if (!values)
    values = fir::AbsentOp::create(builder, loc,
                                   fir::BoxType::get(builder.getNoneType()));

  fir::runtime::genDateAndTime(builder, loc, charArgs[0], charArgs[1],
                               charArgs[2], values);
}

// DIM
aiir::Value IntrinsicLibrary::genDim(aiir::Type resultType,
                                     llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  if (aiir::isa<aiir::IntegerType>(resultType)) {
    aiir::Value zero = builder.createIntegerConstant(loc, resultType, 0);
    auto diff = aiir::arith::SubIOp::create(builder, loc, args[0], args[1]);
    auto cmp = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::sgt, diff, zero);
    return aiir::arith::SelectOp::create(builder, loc, cmp, diff, zero);
  }
  assert(fir::isa_real(resultType) && "Only expects real and integer in DIM");
  aiir::Value zero = builder.createRealZeroConstant(loc, resultType);
  auto diff = aiir::arith::SubFOp::create(builder, loc, args[0], args[1]);
  auto cmp = aiir::arith::CmpFOp::create(
      builder, loc, aiir::arith::CmpFPredicate::OGT, diff, zero);
  return aiir::arith::SelectOp::create(builder, loc, cmp, diff, zero);
}

// DOT_PRODUCT
fir::ExtendedValue
IntrinsicLibrary::genDotProduct(aiir::Type resultType,
                                llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);

  // Handle required vector arguments
  aiir::Value vectorA = fir::getBase(args[0]);
  aiir::Value vectorB = fir::getBase(args[1]);
  // Result type is used for picking appropriate runtime function.
  aiir::Type eleTy = resultType;

  if (fir::isa_complex(eleTy)) {
    aiir::Value result = builder.createTemporary(loc, eleTy);
    fir::runtime::genDotProduct(builder, loc, vectorA, vectorB, result);
    return fir::LoadOp::create(builder, loc, result);
  }

  // This operation is only used to pass the result type
  // information to the DotProduct generator.
  auto resultBox =
      fir::AbsentOp::create(builder, loc, fir::BoxType::get(eleTy));
  return fir::runtime::genDotProduct(builder, loc, vectorA, vectorB, resultBox);
}

// DPROD
aiir::Value IntrinsicLibrary::genDprod(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  assert(fir::isa_real(resultType) &&
         "Result must be double precision in DPROD");
  aiir::Value a = builder.createConvert(loc, resultType, args[0]);
  aiir::Value b = builder.createConvert(loc, resultType, args[1]);
  return aiir::arith::MulFOp::create(builder, loc, a, b);
}

// DSECNDS
// Double precision variant of SECNDS (PGI extension)
fir::ExtendedValue
IntrinsicLibrary::genDsecnds(aiir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1 && "DSECNDS expects one argument");

  aiir::Value refTime = fir::getBase(args[0]);

  if (!refTime)
    fir::emitFatalError(loc, "expected REFERENCE TIME parameter");

  aiir::Value result = fir::runtime::genDsecnds(builder, loc, refTime);

  return builder.createConvert(loc, resultType, result);
}

// DSHIFTL
aiir::Value IntrinsicLibrary::genDshiftl(aiir::Type resultType,
                                         llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 3);

  aiir::Value i = args[0];
  aiir::Value j = args[1];
  int bits = resultType.getIntOrFloatBitWidth();
  aiir::Type signlessType =
      aiir::IntegerType::get(builder.getContext(), bits,
                             aiir::IntegerType::SignednessSemantics::Signless);
  if (resultType.isUnsignedInteger()) {
    i = builder.createConvert(loc, signlessType, i);
    j = builder.createConvert(loc, signlessType, j);
  }
  aiir::Value shift = builder.createConvert(loc, signlessType, args[2]);
  aiir::Value bitSize = builder.createIntegerConstant(loc, signlessType, bits);

  // Per the standard, the value of DSHIFTL(I, J, SHIFT) is equal to
  // IOR (SHIFTL(I, SHIFT), SHIFTR(J, BIT_SIZE(J) - SHIFT))
  aiir::Value diff = aiir::arith::SubIOp::create(builder, loc, bitSize, shift);

  aiir::Value lArgs[2]{i, shift};
  aiir::Value lft = genShift<aiir::arith::ShLIOp>(signlessType, lArgs);

  aiir::Value rArgs[2]{j, diff};
  aiir::Value rgt = genShift<aiir::arith::ShRUIOp>(signlessType, rArgs);
  aiir::Value result = aiir::arith::OrIOp::create(builder, loc, lft, rgt);
  if (resultType.isUnsignedInteger())
    return builder.createConvert(loc, resultType, result);
  return result;
}

// DSHIFTR
aiir::Value IntrinsicLibrary::genDshiftr(aiir::Type resultType,
                                         llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 3);

  aiir::Value i = args[0];
  aiir::Value j = args[1];
  int bits = resultType.getIntOrFloatBitWidth();
  aiir::Type signlessType =
      aiir::IntegerType::get(builder.getContext(), bits,
                             aiir::IntegerType::SignednessSemantics::Signless);
  if (resultType.isUnsignedInteger()) {
    i = builder.createConvert(loc, signlessType, i);
    j = builder.createConvert(loc, signlessType, j);
  }
  aiir::Value shift = builder.createConvert(loc, signlessType, args[2]);
  aiir::Value bitSize = builder.createIntegerConstant(loc, signlessType, bits);

  // Per the standard, the value of DSHIFTR(I, J, SHIFT) is equal to
  // IOR (SHIFTL(I, BIT_SIZE(I) - SHIFT), SHIFTR(J, SHIFT))
  aiir::Value diff = aiir::arith::SubIOp::create(builder, loc, bitSize, shift);

  aiir::Value lArgs[2]{i, diff};
  aiir::Value lft = genShift<aiir::arith::ShLIOp>(signlessType, lArgs);

  aiir::Value rArgs[2]{j, shift};
  aiir::Value rgt = genShift<aiir::arith::ShRUIOp>(signlessType, rArgs);
  aiir::Value result = aiir::arith::OrIOp::create(builder, loc, lft, rgt);
  if (resultType.isUnsignedInteger())
    return builder.createConvert(loc, resultType, result);
  return result;
}

// EOSHIFT
fir::ExtendedValue
IntrinsicLibrary::genEoshift(aiir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);

  // Handle required ARRAY argument
  fir::BoxValue arrayBox = builder.createBox(loc, args[0]);
  aiir::Value array = fir::getBase(arrayBox);
  unsigned arrayRank = arrayBox.rank();

  // Create mutable fir.box to be passed to the runtime for the result.
  aiir::Type resultArrayType = builder.getVarLenSeqTy(resultType, arrayRank);
  fir::MutableBoxValue resultMutableBox = fir::factory::createTempMutableBox(
      builder, loc, resultArrayType, {},
      fir::isPolymorphicType(array.getType()) ? array : aiir::Value{});
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  // Handle optional BOUNDARY argument
  aiir::Value boundary =
      isStaticallyAbsent(args[2])
          ? fir::AbsentOp::create(builder, loc,
                                  fir::BoxType::get(builder.getNoneType()))
          : builder.createBox(loc, args[2]);

  if (arrayRank == 1) {
    // Vector case
    // Handle required SHIFT argument as a scalar
    const aiir::Value *shiftAddr = args[1].getUnboxed();
    assert(shiftAddr && "nonscalar EOSHIFT SHIFT argument");
    auto shift = fir::LoadOp::create(builder, loc, *shiftAddr);
    fir::runtime::genEoshiftVector(builder, loc, resultIrBox, array, shift,
                                   boundary);
  } else {
    // Non-vector case
    // Handle required SHIFT argument as an array
    aiir::Value shift = builder.createBox(loc, args[1]);

    // Handle optional DIM argument
    aiir::Value dim =
        isStaticallyAbsent(args[3])
            ? builder.createIntegerConstant(loc, builder.getIndexType(), 1)
            : fir::getBase(args[3]);
    fir::runtime::genEoshift(builder, loc, resultIrBox, array, shift, boundary,
                             dim);
  }
  return readAndAddCleanUp(resultMutableBox, resultType, "EOSHIFT");
}

// EXECUTE_COMMAND_LINE
void IntrinsicLibrary::genExecuteCommandLine(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 5);

  aiir::Value command = fir::getBase(args[0]);
  // Optional arguments: wait, exitstat, cmdstat, cmdmsg.
  const fir::ExtendedValue &wait = args[1];
  const fir::ExtendedValue &exitstat = args[2];
  const fir::ExtendedValue &cmdstat = args[3];
  const fir::ExtendedValue &cmdmsg = args[4];

  if (!command)
    fir::emitFatalError(loc, "expected COMMAND parameter");

  aiir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());

  aiir::Value waitBool;
  if (isStaticallyAbsent(wait)) {
    waitBool = builder.createBool(loc, true);
  } else {
    aiir::Type i1Ty = builder.getI1Type();
    aiir::Value waitAddr = fir::getBase(wait);
    aiir::Value waitIsPresentAtRuntime =
        builder.genIsNotNullAddr(loc, waitAddr);
    waitBool =
        builder
            .genIfOp(loc, {i1Ty}, waitIsPresentAtRuntime,
                     /*withElseRegion=*/true)
            .genThen([&]() {
              auto waitLoad = fir::LoadOp::create(builder, loc, waitAddr);
              aiir::Value cast = builder.createConvert(loc, i1Ty, waitLoad);
              fir::ResultOp::create(builder, loc, cast);
            })
            .genElse([&]() {
              aiir::Value trueVal = builder.createBool(loc, true);
              fir::ResultOp::create(builder, loc, trueVal);
            })
            .getResults()[0];
  }

  aiir::Value exitstatBox =
      isStaticallyPresent(exitstat)
          ? fir::getBase(exitstat)
          : fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();
  aiir::Value cmdstatBox =
      isStaticallyPresent(cmdstat)
          ? fir::getBase(cmdstat)
          : fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();
  aiir::Value cmdmsgBox =
      isStaticallyPresent(cmdmsg)
          ? fir::getBase(cmdmsg)
          : fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();
  fir::runtime::genExecuteCommandLine(builder, loc, command, waitBool,
                                      exitstatBox, cmdstatBox, cmdmsgBox);
}

// ETIME
fir::ExtendedValue
IntrinsicLibrary::genEtime(std::optional<aiir::Type> resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  assert((args.size() == 2 && !resultType.has_value()) ||
         (args.size() == 1 && resultType.has_value()));

  aiir::Value values = fir::getBase(args[0]);
  if (resultType.has_value()) {
    // function form
    if (!values)
      fir::emitFatalError(loc, "expected VALUES parameter");

    auto timeAddr = builder.createTemporary(loc, *resultType);
    auto timeBox = builder.createBox(loc, timeAddr);
    fir::runtime::genEtime(builder, loc, values, timeBox);
    return fir::LoadOp::create(builder, loc, timeAddr);
  } else {
    // subroutine form
    aiir::Value time = fir::getBase(args[1]);
    if (!values)
      fir::emitFatalError(loc, "expected VALUES parameter");
    if (!time)
      fir::emitFatalError(loc, "expected TIME parameter");

    fir::runtime::genEtime(builder, loc, values, time);
    return {};
  }
  return {};
}

// EXIT
void IntrinsicLibrary::genExit(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);

  aiir::Value status =
      isStaticallyAbsent(args[0])
          ? builder.createIntegerConstant(loc, builder.getDefaultIntegerType(),
                                          EXIT_SUCCESS)
          : fir::getBase(args[0]);

  fir::runtime::genExit(builder, loc, status);
}

// EXPONENT
aiir::Value IntrinsicLibrary::genExponent(aiir::Type resultType,
                                          llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genExponent(builder, loc, resultType,
                                fir::getBase(args[0])));
}

// EXTENDS_TYPE_OF
fir::ExtendedValue
IntrinsicLibrary::genExtendsTypeOf(aiir::Type resultType,
                                   llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genExtendsTypeOf(builder, loc, fir::getBase(args[0]),
                                     fir::getBase(args[1])));
}

// F_C_STRING
fir::ExtendedValue
IntrinsicLibrary::genFCString(aiir::Type resultType,
                              llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() >= 1 && args.size() <= 2);

  aiir::Value string = builder.createBox(loc, args[0]);

  // Handle optional ASIS argument
  aiir::Value asis = isStaticallyAbsent(args, 1)
                         ? builder.createBool(loc, false)
                         : fir::getBase(args[1]);

  // Create mutable fir.box to be passed to the runtime for the result.
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genFCString(builder, loc, resultIrBox, string, asis);

  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  return readAndAddCleanUp(resultMutableBox, resultType, "F_C_STRING");
}

// FINDLOC
fir::ExtendedValue
IntrinsicLibrary::genFindloc(aiir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 6);

  // Handle required array argument
  aiir::Value array = builder.createBox(loc, args[0]);
  unsigned rank = fir::BoxValue(array).rank();
  assert(rank >= 1);

  // Handle required value argument
  aiir::Value val = builder.createBox(loc, args[1]);

  // Check if dim argument is present
  bool absentDim = isStaticallyAbsent(args[2]);

  // Handle optional mask argument
  auto mask = isStaticallyAbsent(args[3])
                  ? fir::AbsentOp::create(
                        builder, loc, fir::BoxType::get(builder.getI1Type()))
                  : builder.createBox(loc, args[3]);

  // Handle optional kind argument
  auto kind = isStaticallyAbsent(args[4])
                  ? builder.createIntegerConstant(
                        loc, builder.getIndexType(),
                        builder.getKindMap().defaultIntegerKind())
                  : fir::getBase(args[4]);

  // Handle optional back argument
  auto back = isStaticallyAbsent(args[5]) ? builder.createBool(loc, false)
                                          : fir::getBase(args[5]);

  if (!absentDim && rank == 1) {
    // If dim argument is present and the array is rank 1, then the result is
    // a scalar (since the the result is rank-1 or 0).
    // Therefore, we use a scalar result descriptor with FindlocDim().
    // Create mutable fir.box to be passed to the runtime for the result.
    fir::MutableBoxValue resultMutableBox =
        fir::factory::createTempMutableBox(builder, loc, resultType);
    aiir::Value resultIrBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
    aiir::Value dim = fir::getBase(args[2]);

    fir::runtime::genFindlocDim(builder, loc, resultIrBox, array, val, dim,
                                mask, kind, back);
    // Handle cleanup of allocatable result descriptor and return
    return readAndAddCleanUp(resultMutableBox, resultType, "FINDLOC");
  }

  // The result will be an array. Create mutable fir.box to be passed to the
  // runtime for the result.
  aiir::Type resultArrayType =
      builder.getVarLenSeqTy(resultType, absentDim ? 1 : rank - 1);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  if (absentDim) {
    fir::runtime::genFindloc(builder, loc, resultIrBox, array, val, mask, kind,
                             back);
  } else {
    aiir::Value dim = fir::getBase(args[2]);
    fir::runtime::genFindlocDim(builder, loc, resultIrBox, array, val, dim,
                                mask, kind, back);
  }
  return readAndAddCleanUp(resultMutableBox, resultType, "FINDLOC");
}

// FLOOR
aiir::Value IntrinsicLibrary::genFloor(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  // Optional KIND argument.
  assert(args.size() >= 1);
  aiir::Value arg = args[0];
  // Use LLVM floor that returns real.
  aiir::Value floor = genRuntimeCall("floor", arg.getType(), {arg});
  return builder.createConvert(loc, resultType, floor);
}

// FLUSH
void IntrinsicLibrary::genFlush(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);

  aiir::Value unit;
  if (isStaticallyAbsent(args[0]))
    // Give a sentinal value of `-1` on the `()` case.
    unit = builder.createIntegerConstant(loc, builder.getI32Type(), -1);
  else {
    unit = fir::getBase(args[0]);
    if (isOptional(unit)) {
      aiir::Value isPresent =
          fir::IsPresentOp::create(builder, loc, builder.getI1Type(), unit);
      unit = builder
                 .genIfOp(loc, builder.getI32Type(), isPresent,
                          /*withElseRegion=*/true)
                 .genThen([&]() {
                   aiir::Value loaded = fir::LoadOp::create(builder, loc, unit);
                   fir::ResultOp::create(builder, loc, loaded);
                 })
                 .genElse([&]() {
                   aiir::Value negOne = builder.createIntegerConstant(
                       loc, builder.getI32Type(), -1);
                   fir::ResultOp::create(builder, loc, negOne);
                 })
                 .getResults()[0];
    } else {
      unit = fir::LoadOp::create(builder, loc, unit);
    }
  }

  fir::runtime::genFlush(builder, loc, unit);
}

// FRACTION
aiir::Value IntrinsicLibrary::genFraction(aiir::Type resultType,
                                          llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genFraction(builder, loc, fir::getBase(args[0])));
}

void IntrinsicLibrary::genFree(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);

  fir::runtime::genFree(builder, loc, fir::getBase(args[0]));
}

// FSEEK
fir::ExtendedValue
IntrinsicLibrary::genFseek(std::optional<aiir::Type> resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  assert((args.size() == 4 && !resultType.has_value()) ||
         (args.size() == 3 && resultType.has_value()));
  aiir::Value unit = fir::getBase(args[0]);
  aiir::Value offset = fir::getBase(args[1]);
  aiir::Value whence = fir::getBase(args[2]);
  if (!unit)
    fir::emitFatalError(loc, "expected UNIT argument");
  if (!offset)
    fir::emitFatalError(loc, "expected OFFSET argument");
  if (!whence)
    fir::emitFatalError(loc, "expected WHENCE argument");
  aiir::Value statusValue =
      fir::runtime::genFseek(builder, loc, unit, offset, whence);
  if (resultType.has_value()) { // function
    return builder.createConvert(loc, *resultType, statusValue);
  } else { // subroutine
    const fir::ExtendedValue &statusVar = args[3];
    if (!isStaticallyAbsent(statusVar)) {
      aiir::Value statusAddr = fir::getBase(statusVar);
      aiir::Value statusIsPresentAtRuntime =
          builder.genIsNotNullAddr(loc, statusAddr);
      builder.genIfThen(loc, statusIsPresentAtRuntime)
          .genThen([&]() {
            builder.createStoreWithConvert(loc, statusValue, statusAddr);
          })
          .end();
    }
    return {};
  }
}

// FTELL
fir::ExtendedValue
IntrinsicLibrary::genFtell(std::optional<aiir::Type> resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  assert((args.size() == 2 && !resultType.has_value()) ||
         (args.size() == 1 && resultType.has_value()));
  aiir::Value unit = fir::getBase(args[0]);
  if (!unit)
    fir::emitFatalError(loc, "expected UNIT argument");
  aiir::Value offsetValue = fir::runtime::genFtell(builder, loc, unit);
  if (resultType.has_value()) { // function
    return offsetValue;
  } else { // subroutine
    const fir::ExtendedValue &offsetVar = args[1];
    if (!isStaticallyAbsent(offsetVar)) {
      aiir::Value offsetAddr = fir::getBase(offsetVar);
      aiir::Value offsetIsPresentAtRuntime =
          builder.genIsNotNullAddr(loc, offsetAddr);
      builder.genIfThen(loc, offsetIsPresentAtRuntime)
          .genThen([&]() {
            builder.createStoreWithConvert(loc, offsetValue, offsetAddr);
          })
          .end();
    }
    return {};
  }
}

// GET_TEAM
aiir::Value IntrinsicLibrary::genGetTeam(aiir::Type resultType,
                                         llvm::ArrayRef<aiir::Value> args) {
  converter->checkCoarrayEnabled();
  assert(args.size() == 1);
  return mif::GetTeamOp::create(builder, loc, fir::BoxType::get(resultType),
                                /*level*/ args[0]);
}

// GETCWD
fir::ExtendedValue
IntrinsicLibrary::genGetCwd(std::optional<aiir::Type> resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert((args.size() == 1 && resultType.has_value()) ||
         (args.size() >= 1 && !resultType.has_value()));

  aiir::Value cwd = fir::getBase(args[0]);
  aiir::Value statusValue = fir::runtime::genGetCwd(builder, loc, cwd);

  if (resultType.has_value()) {
    // Function form, return status.
    return statusValue;
  } else {
    // Subroutine form, store status and return none.
    const fir::ExtendedValue &status = args[1];
    if (!isStaticallyAbsent(status)) {
      aiir::Value statusAddr = fir::getBase(status);
      aiir::Value statusIsPresentAtRuntime =
          builder.genIsNotNullAddr(loc, statusAddr);
      builder.genIfThen(loc, statusIsPresentAtRuntime)
          .genThen([&]() {
            builder.createStoreWithConvert(loc, statusValue, statusAddr);
          })
          .end();
    }
  }

  return {};
}

// GET_COMMAND
void IntrinsicLibrary::genGetCommand(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  const fir::ExtendedValue &command = args[0];
  const fir::ExtendedValue &length = args[1];
  const fir::ExtendedValue &status = args[2];
  const fir::ExtendedValue &errmsg = args[3];

  // If none of the optional parameters are present, do nothing.
  if (!isStaticallyPresent(command) && !isStaticallyPresent(length) &&
      !isStaticallyPresent(status) && !isStaticallyPresent(errmsg))
    return;

  aiir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
  aiir::Value commandBox =
      isStaticallyPresent(command)
          ? fir::getBase(command)
          : fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();
  aiir::Value lenBox =
      isStaticallyPresent(length)
          ? fir::getBase(length)
          : fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();
  aiir::Value errBox =
      isStaticallyPresent(errmsg)
          ? fir::getBase(errmsg)
          : fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();
  aiir::Value stat =
      fir::runtime::genGetCommand(builder, loc, commandBox, lenBox, errBox);
  if (isStaticallyPresent(status)) {
    aiir::Value statAddr = fir::getBase(status);
    aiir::Value statIsPresentAtRuntime =
        builder.genIsNotNullAddr(loc, statAddr);
    builder.genIfThen(loc, statIsPresentAtRuntime)
        .genThen([&]() { builder.createStoreWithConvert(loc, stat, statAddr); })
        .end();
  }
}

// GETGID
aiir::Value IntrinsicLibrary::genGetGID(aiir::Type resultType,
                                        llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 0 && "getgid takes no input");
  return builder.createConvert(loc, resultType,
                               fir::runtime::genGetGID(builder, loc));
}

// GETPID
aiir::Value IntrinsicLibrary::genGetPID(aiir::Type resultType,
                                        llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 0 && "getpid takes no input");
  return builder.createConvert(loc, resultType,
                               fir::runtime::genGetPID(builder, loc));
}

// GETUID
aiir::Value IntrinsicLibrary::genGetUID(aiir::Type resultType,
                                        llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 0 && "getgid takes no input");
  return builder.createConvert(loc, resultType,
                               fir::runtime::genGetUID(builder, loc));
}

// GET_COMMAND_ARGUMENT
void IntrinsicLibrary::genGetCommandArgument(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 5);
  aiir::Value number = fir::getBase(args[0]);
  const fir::ExtendedValue &value = args[1];
  const fir::ExtendedValue &length = args[2];
  const fir::ExtendedValue &status = args[3];
  const fir::ExtendedValue &errmsg = args[4];

  if (!number)
    fir::emitFatalError(loc, "expected NUMBER parameter");

  // If none of the optional parameters are present, do nothing.
  if (!isStaticallyPresent(value) && !isStaticallyPresent(length) &&
      !isStaticallyPresent(status) && !isStaticallyPresent(errmsg))
    return;

  aiir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
  aiir::Value valBox =
      isStaticallyPresent(value)
          ? fir::getBase(value)
          : fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();
  aiir::Value lenBox =
      isStaticallyPresent(length)
          ? fir::getBase(length)
          : fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();
  aiir::Value errBox =
      isStaticallyPresent(errmsg)
          ? fir::getBase(errmsg)
          : fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();
  aiir::Value stat = fir::runtime::genGetCommandArgument(
      builder, loc, number, valBox, lenBox, errBox);
  if (isStaticallyPresent(status)) {
    aiir::Value statAddr = fir::getBase(status);
    aiir::Value statIsPresentAtRuntime =
        builder.genIsNotNullAddr(loc, statAddr);
    builder.genIfThen(loc, statIsPresentAtRuntime)
        .genThen([&]() { builder.createStoreWithConvert(loc, stat, statAddr); })
        .end();
  }
}

// GET_ENVIRONMENT_VARIABLE
void IntrinsicLibrary::genGetEnvironmentVariable(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 6);
  aiir::Value name = fir::getBase(args[0]);
  const fir::ExtendedValue &value = args[1];
  const fir::ExtendedValue &length = args[2];
  const fir::ExtendedValue &status = args[3];
  const fir::ExtendedValue &trimName = args[4];
  const fir::ExtendedValue &errmsg = args[5];

  if (!name)
    fir::emitFatalError(loc, "expected NAME parameter");

  // If none of the optional parameters are present, do nothing.
  if (!isStaticallyPresent(value) && !isStaticallyPresent(length) &&
      !isStaticallyPresent(status) && !isStaticallyPresent(errmsg))
    return;

  // Handle optional TRIM_NAME argument
  aiir::Value trim;
  if (isStaticallyAbsent(trimName)) {
    trim = builder.createBool(loc, true);
  } else {
    aiir::Type i1Ty = builder.getI1Type();
    aiir::Value trimNameAddr = fir::getBase(trimName);
    aiir::Value trimNameIsPresentAtRuntime =
        builder.genIsNotNullAddr(loc, trimNameAddr);
    trim = builder
               .genIfOp(loc, {i1Ty}, trimNameIsPresentAtRuntime,
                        /*withElseRegion=*/true)
               .genThen([&]() {
                 auto trimLoad =
                     fir::LoadOp::create(builder, loc, trimNameAddr);
                 aiir::Value cast = builder.createConvert(loc, i1Ty, trimLoad);
                 fir::ResultOp::create(builder, loc, cast);
               })
               .genElse([&]() {
                 aiir::Value trueVal = builder.createBool(loc, true);
                 fir::ResultOp::create(builder, loc, trueVal);
               })
               .getResults()[0];
  }

  aiir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
  aiir::Value valBox =
      isStaticallyPresent(value)
          ? fir::getBase(value)
          : fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();
  aiir::Value lenBox =
      isStaticallyPresent(length)
          ? fir::getBase(length)
          : fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();
  aiir::Value errBox =
      isStaticallyPresent(errmsg)
          ? fir::getBase(errmsg)
          : fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();
  aiir::Value stat = fir::runtime::genGetEnvVariable(builder, loc, name, valBox,
                                                     lenBox, trim, errBox);
  if (isStaticallyPresent(status)) {
    aiir::Value statAddr = fir::getBase(status);
    aiir::Value statIsPresentAtRuntime =
        builder.genIsNotNullAddr(loc, statAddr);
    builder.genIfThen(loc, statIsPresentAtRuntime)
        .genThen([&]() { builder.createStoreWithConvert(loc, stat, statAddr); })
        .end();
  }
}

// HOSTNM
fir::ExtendedValue
IntrinsicLibrary::genHostnm(std::optional<aiir::Type> resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert((args.size() == 1 && resultType.has_value()) ||
         (args.size() >= 1 && !resultType.has_value()));

  aiir::Value res = fir::getBase(args[0]);
  aiir::Value statusValue = fir::runtime::genHostnm(builder, loc, res);

  if (resultType.has_value()) {
    // Function form, return status.
    return builder.createConvert(loc, *resultType, statusValue);
  }

  // Subroutine form, store status and return none.
  const fir::ExtendedValue &status = args[1];
  if (!isStaticallyAbsent(status)) {
    aiir::Value statusAddr = fir::getBase(status);
    aiir::Value statusIsPresentAtRuntime =
        builder.genIsNotNullAddr(loc, statusAddr);
    builder.genIfThen(loc, statusIsPresentAtRuntime)
        .genThen([&]() {
          builder.createStoreWithConvert(loc, statusValue, statusAddr);
        })
        .end();
  }

  return {};
}

/// Process calls to Maxval, Minval, Product, Sum intrinsic functions that
/// take a DIM argument.
template <typename FD>
static fir::MutableBoxValue
genFuncDim(FD funcDim, aiir::Type resultType, fir::FirOpBuilder &builder,
           aiir::Location loc, aiir::Value array, fir::ExtendedValue dimArg,
           aiir::Value mask, int rank) {

  // Create mutable fir.box to be passed to the runtime for the result.
  aiir::Type resultArrayType = builder.getVarLenSeqTy(resultType, rank - 1);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  aiir::Value dim =
      isStaticallyAbsent(dimArg)
          ? builder.createIntegerConstant(loc, builder.getIndexType(), 0)
          : fir::getBase(dimArg);
  funcDim(builder, loc, resultIrBox, array, dim, mask);

  return resultMutableBox;
}

/// Process calls to Product, Sum, IAll, IAny, IParity intrinsic functions
template <typename FN, typename FD>
fir::ExtendedValue
IntrinsicLibrary::genReduction(FN func, FD funcDim, llvm::StringRef errMsg,
                               aiir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 3);

  // Handle required array argument
  fir::BoxValue arryTmp = builder.createBox(loc, args[0]);
  aiir::Value array = fir::getBase(arryTmp);
  int rank = arryTmp.rank();
  assert(rank >= 1);

  // Handle optional mask argument
  auto mask = isStaticallyAbsent(args[2])
                  ? fir::AbsentOp::create(
                        builder, loc, fir::BoxType::get(builder.getI1Type()))
                  : builder.createBox(loc, args[2]);

  bool absentDim = isStaticallyAbsent(args[1]);

  // We call the type specific versions because the result is scalar
  // in the case below.
  if (absentDim || rank == 1) {
    aiir::Type ty = array.getType();
    aiir::Type arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
    auto eleTy = aiir::cast<fir::SequenceType>(arrTy).getElementType();
    if (fir::isa_complex(eleTy)) {
      aiir::Value result = builder.createTemporary(loc, eleTy);
      func(builder, loc, array, mask, result);
      return fir::LoadOp::create(builder, loc, result);
    }
    auto resultBox = fir::AbsentOp::create(
        builder, loc, fir::BoxType::get(builder.getI1Type()));
    return func(builder, loc, array, mask, resultBox);
  }
  // Handle Product/Sum cases that have an array result.
  auto resultMutableBox =
      genFuncDim(funcDim, resultType, builder, loc, array, args[1], mask, rank);
  return readAndAddCleanUp(resultMutableBox, resultType, errMsg);
}

// IALL
fir::ExtendedValue
IntrinsicLibrary::genIall(aiir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  return genReduction(fir::runtime::genIAll, fir::runtime::genIAllDim, "IALL",
                      resultType, args);
}

// IAND
aiir::Value IntrinsicLibrary::genIand(aiir::Type resultType,
                                      llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  return builder.createUnsigned<aiir::arith::AndIOp>(loc, resultType, args[0],
                                                     args[1]);
}

// IANY
fir::ExtendedValue
IntrinsicLibrary::genIany(aiir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  return genReduction(fir::runtime::genIAny, fir::runtime::genIAnyDim, "IANY",
                      resultType, args);
}

// IBCLR
aiir::Value IntrinsicLibrary::genIbclr(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  // A conformant IBCLR(I,POS) call satisfies:
  //     POS >= 0
  //     POS < BIT_SIZE(I)
  // Return:  I & (!(1 << POS))
  assert(args.size() == 2);
  aiir::Type signlessType = aiir::IntegerType::get(
      builder.getContext(), resultType.getIntOrFloatBitWidth(),
      aiir::IntegerType::SignednessSemantics::Signless);
  aiir::Value one = builder.createIntegerConstant(loc, signlessType, 1);
  aiir::Value ones = builder.createAllOnesInteger(loc, signlessType);
  aiir::Value pos = builder.createConvert(loc, signlessType, args[1]);
  aiir::Value bit = aiir::arith::ShLIOp::create(builder, loc, one, pos);
  aiir::Value mask = aiir::arith::XOrIOp::create(builder, loc, ones, bit);
  return builder.createUnsigned<aiir::arith::AndIOp>(loc, resultType, args[0],
                                                     mask);
}

// IBITS
aiir::Value IntrinsicLibrary::genIbits(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  // A conformant IBITS(I,POS,LEN) call satisfies:
  //     POS >= 0
  //     LEN >= 0
  //     POS + LEN <= BIT_SIZE(I)
  // Return:  LEN == 0 ? 0 : (I >> POS) & (-1 >> (BIT_SIZE(I) - LEN))
  // For a conformant call, implementing (I >> POS) with a signed or an
  // unsigned shift produces the same result.  For a nonconformant call,
  // the two choices may produce different results.
  assert(args.size() == 3);
  aiir::Type signlessType = aiir::IntegerType::get(
      builder.getContext(), resultType.getIntOrFloatBitWidth(),
      aiir::IntegerType::SignednessSemantics::Signless);
  aiir::Value word = args[0];
  if (word.getType().isUnsignedInteger())
    word = builder.createConvert(loc, signlessType, word);
  aiir::Value pos = builder.createConvert(loc, signlessType, args[1]);
  aiir::Value len = builder.createConvert(loc, signlessType, args[2]);
  aiir::Value bitSize = builder.createIntegerConstant(
      loc, signlessType, aiir::cast<aiir::IntegerType>(resultType).getWidth());
  aiir::Value shiftCount =
      aiir::arith::SubIOp::create(builder, loc, bitSize, len);
  aiir::Value zero = builder.createIntegerConstant(loc, signlessType, 0);
  aiir::Value ones = builder.createAllOnesInteger(loc, signlessType);
  aiir::Value mask =
      aiir::arith::ShRUIOp::create(builder, loc, ones, shiftCount);
  aiir::Value res1 = builder.createUnsigned<aiir::arith::ShRSIOp>(
      loc, signlessType, word, pos);
  aiir::Value res2 = aiir::arith::AndIOp::create(builder, loc, res1, mask);
  aiir::Value lenIsZero = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::eq, len, zero);
  aiir::Value result =
      aiir::arith::SelectOp::create(builder, loc, lenIsZero, zero, res2);
  if (resultType.isUnsignedInteger())
    return builder.createConvert(loc, resultType, result);
  return result;
}

// IBSET
aiir::Value IntrinsicLibrary::genIbset(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  // A conformant IBSET(I,POS) call satisfies:
  //     POS >= 0
  //     POS < BIT_SIZE(I)
  // Return:  I | (1 << POS)
  assert(args.size() == 2);
  aiir::Type signlessType = aiir::IntegerType::get(
      builder.getContext(), resultType.getIntOrFloatBitWidth(),
      aiir::IntegerType::SignednessSemantics::Signless);
  aiir::Value one = builder.createIntegerConstant(loc, signlessType, 1);
  aiir::Value pos = builder.createConvert(loc, signlessType, args[1]);
  aiir::Value mask = aiir::arith::ShLIOp::create(builder, loc, one, pos);
  return builder.createUnsigned<aiir::arith::OrIOp>(loc, resultType, args[0],
                                                    mask);
}

// ICHAR
fir::ExtendedValue
IntrinsicLibrary::genIchar(aiir::Type resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  // There can be an optional kind in second argument.
  assert(args.size() == 2);
  const fir::CharBoxValue *charBox = args[0].getCharBox();
  if (!charBox)
    llvm::report_fatal_error("expected character scalar");

  fir::factory::CharacterExprHelper helper{builder, loc};
  aiir::Value buffer = charBox->getBuffer();
  aiir::Type bufferTy = buffer.getType();
  aiir::Value charVal;
  if (auto charTy = aiir::dyn_cast<fir::CharacterType>(bufferTy)) {
    assert(charTy.singleton());
    charVal = buffer;
  } else {
    // Character is in memory, cast to fir.ref<char> and load.
    aiir::Type ty = fir::dyn_cast_ptrEleTy(bufferTy);
    if (!ty)
      llvm::report_fatal_error("expected memory type");
    // The length of in the character type may be unknown. Casting
    // to a singleton ref is required before loading.
    fir::CharacterType eleType = helper.getCharacterType(ty);
    fir::CharacterType charType =
        fir::CharacterType::get(builder.getContext(), eleType.getFKind(), 1);
    aiir::Type toTy = builder.getRefType(charType);
    aiir::Value cast = builder.createConvert(loc, toTy, buffer);
    charVal = fir::LoadOp::create(builder, loc, cast);
  }
  LLVM_DEBUG(llvm::dbgs() << "ichar(" << charVal << ")\n");
  auto code = helper.extractCodeFromSingleton(charVal);
  if (code.getType() == resultType)
    return code;
  return aiir::arith::ExtUIOp::create(builder, loc, resultType, code);
}

// llvm floating point class intrinsic test values
//   0   Signaling NaN
//   1   Quiet NaN
//   2   Negative infinity
//   3   Negative normal
//   4   Negative subnormal
//   5   Negative zero
//   6   Positive zero
//   7   Positive subnormal
//   8   Positive normal
//   9   Positive infinity
static constexpr int finiteTest = 0b0111111000;
static constexpr int infiniteTest = 0b1000000100;
static constexpr int nanTest = 0b0000000011;
static constexpr int negativeTest = 0b0000111100;
static constexpr int normalTest = 0b0101101000;
static constexpr int positiveTest = 0b1111000000;
static constexpr int snanTest = 0b0000000001;
static constexpr int subnormalTest = 0b0010010000;
static constexpr int zeroTest = 0b0001100000;

aiir::Value IntrinsicLibrary::genIsFPClass(aiir::Type resultType,
                                           llvm::ArrayRef<aiir::Value> args,
                                           int fpclass) {
  assert(args.size() == 1);
  aiir::Type i1Ty = builder.getI1Type();
  aiir::Value isfpclass =
      aiir::LLVM::IsFPClass::create(builder, loc, i1Ty, args[0], fpclass);
  return builder.createConvert(loc, resultType, isfpclass);
}

// Generate a quiet NaN of a given floating point type.
aiir::Value IntrinsicLibrary::genQNan(aiir::Type resultType) {
  return genIeeeValue(resultType, builder.createIntegerConstant(
                                      loc, builder.getIntegerType(8),
                                      _FORTRAN_RUNTIME_IEEE_QUIET_NAN));
}

// Generate code to raise \p excepts if \p cond is absent, or present and true.
void IntrinsicLibrary::genRaiseExcept(int excepts, aiir::Value cond) {
  fir::IfOp ifOp;
  if (cond) {
    ifOp = fir::IfOp::create(builder, loc, cond, /*withElseRegion=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  }
  aiir::Type i32Ty = builder.getIntegerType(32);
  fir::runtime::genFeraiseexcept(
      builder, loc,
      fir::runtime::genMapExcept(
          builder, loc, builder.createIntegerConstant(loc, i32Ty, excepts)));
  if (cond)
    builder.setInsertionPointAfter(ifOp);
}

// Return a reference to the contents of a derived type with one field.
// Also return the field type.
static std::pair<aiir::Value, aiir::Type>
getFieldRef(fir::FirOpBuilder &builder, aiir::Location loc, aiir::Value rec,
            unsigned index = 0) {
  auto recType =
      aiir::dyn_cast<fir::RecordType>(fir::unwrapPassByRefType(rec.getType()));
  assert(index < recType.getTypeList().size() && "not enough components");
  auto [fieldName, fieldTy] = recType.getTypeList()[index];
  aiir::Value field = fir::FieldIndexOp::create(
      builder, loc, fir::FieldType::get(recType.getContext()), fieldName,
      recType, fir::getTypeParams(rec));
  return {fir::CoordinateOp::create(builder, loc, builder.getRefType(fieldTy),
                                    rec, field),
          fieldTy};
}

// IEEE_CLASS_TYPE OPERATOR(==), OPERATOR(/=)
// IEEE_ROUND_TYPE OPERATOR(==), OPERATOR(/=)
template <aiir::arith::CmpIPredicate pred>
aiir::Value
IntrinsicLibrary::genIeeeTypeCompare(aiir::Type resultType,
                                     llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  auto [leftRef, fieldTy] = getFieldRef(builder, loc, args[0]);
  auto [rightRef, ignore] = getFieldRef(builder, loc, args[1]);
  aiir::Value left = fir::LoadOp::create(builder, loc, fieldTy, leftRef);
  aiir::Value right = fir::LoadOp::create(builder, loc, fieldTy, rightRef);
  return aiir::arith::CmpIOp::create(builder, loc, pred, left, right);
}

// IEEE_CLASS
aiir::Value IntrinsicLibrary::genIeeeClass(aiir::Type resultType,
                                           llvm::ArrayRef<aiir::Value> args) {
  // Classify REAL argument X as one of 11 IEEE_CLASS_TYPE values via
  // a table lookup on an index built from 5 values derived from X.
  // In indexing order, the values are:
  //
  //   [s] sign bit
  //   [e] exponent != 0
  //   [m] exponent == 1..1 (max exponent)
  //   [l] low-order significand != 0
  //   [h] high-order significand (kind=10: 2 bits; other kinds: 1 bit)
  //
  // kind=10 values have an explicit high-order integer significand bit,
  // whereas this bit is implicit for other kinds. This requires using a 6-bit
  // index into a 64-slot table for kind=10 argument classification queries
  // vs. a 5-bit index into a 32-slot table for other argument kind queries.
  // The instruction sequence is the same for the two cases.
  //
  // Placing the [l] and [h] significand bits in "swapped" order rather than
  // "natural" order enables more efficient generated code.

  assert(args.size() == 1);
  aiir::Value realVal = args[0];
  aiir::FloatType realType = aiir::dyn_cast<aiir::FloatType>(realVal.getType());
  const unsigned intWidth = realType.getWidth();
  aiir::Type intType = builder.getIntegerType(intWidth);
  aiir::Value intVal =
      aiir::arith::BitcastOp::create(builder, loc, intType, realVal);
  llvm::StringRef tableName = RTNAME_STRING(IeeeClassTable);
  uint64_t highSignificandSize = (realType.getWidth() == 80) + 1;

  // Get masks and shift counts.
  aiir::Value signShift, highSignificandShift, exponentMask, lowSignificandMask;
  auto createIntegerConstant = [&](uint64_t k) {
    return builder.createIntegerConstant(loc, intType, k);
  };
  auto createIntegerConstantAPI = [&](const llvm::APInt &apInt) {
    return aiir::arith::ConstantOp::create(
        builder, loc, intType, builder.getIntegerAttr(intType, apInt));
  };
  auto getMasksAndShifts = [&](uint64_t totalSize, uint64_t exponentSize,
                               uint64_t significandSize,
                               bool hasExplicitBit = false) {
    assert(1 + exponentSize + significandSize == totalSize &&
           "invalid floating point fields");
    uint64_t lowSignificandSize = significandSize - hasExplicitBit - 1;
    signShift = createIntegerConstant(totalSize - 1 - hasExplicitBit - 4);
    highSignificandShift = createIntegerConstant(lowSignificandSize);
    llvm::APInt exponentMaskAPI =
        llvm::APInt::getBitsSet(intWidth, /*lo=*/significandSize,
                                /*hi=*/significandSize + exponentSize);
    exponentMask = createIntegerConstantAPI(exponentMaskAPI);
    llvm::APInt lowSignificandMaskAPI =
        llvm::APInt::getLowBitsSet(intWidth, lowSignificandSize);
    lowSignificandMask = createIntegerConstantAPI(lowSignificandMaskAPI);
  };
  switch (realType.getWidth()) {
  case 16:
    if (realType.isF16()) {
      // kind=2: 1 sign bit, 5 exponent bits, 10 significand bits
      getMasksAndShifts(16, 5, 10);
    } else {
      // kind=3: 1 sign bit, 8 exponent bits, 7 significand bits
      getMasksAndShifts(16, 8, 7);
    }
    break;
  case 32: // kind=4: 1 sign bit, 8 exponent bits, 23 significand bits
    getMasksAndShifts(32, 8, 23);
    break;
  case 64: // kind=8: 1 sign bit, 11 exponent bits, 52 significand bits
    getMasksAndShifts(64, 11, 52);
    break;
  case 80: // kind=10: 1 sign bit, 15 exponent bits, 1+63 significand bits
    getMasksAndShifts(80, 15, 64, /*hasExplicitBit=*/true);
    tableName = RTNAME_STRING(IeeeClassTable_10);
    break;
  case 128: // kind=16: 1 sign bit, 15 exponent bits, 112 significand bits
    getMasksAndShifts(128, 15, 112);
    break;
  default:
    llvm_unreachable("unknown real type");
  }

  // [s] sign bit
  int pos = 3 + highSignificandSize;
  aiir::Value index = aiir::arith::AndIOp::create(
      builder, loc,
      aiir::arith::ShRUIOp::create(builder, loc, intVal, signShift),
      createIntegerConstant(1ULL << pos));

  // [e] exponent != 0
  aiir::Value exponent =
      aiir::arith::AndIOp::create(builder, loc, intVal, exponentMask);
  aiir::Value zero = createIntegerConstant(0);
  index = aiir::arith::OrIOp::create(
      builder, loc, index,
      aiir::arith::SelectOp::create(
          builder, loc,
          aiir::arith::CmpIOp::create(
              builder, loc, aiir::arith::CmpIPredicate::ne, exponent, zero),
          createIntegerConstant(1ULL << --pos), zero));

  // [m] exponent == 1..1 (max exponent)
  index = aiir::arith::OrIOp::create(
      builder, loc, index,
      aiir::arith::SelectOp::create(
          builder, loc,
          aiir::arith::CmpIOp::create(builder, loc,
                                      aiir::arith::CmpIPredicate::eq, exponent,
                                      exponentMask),
          createIntegerConstant(1ULL << --pos), zero));

  // [l] low-order significand != 0
  index = aiir::arith::OrIOp::create(
      builder, loc, index,
      aiir::arith::SelectOp::create(
          builder, loc,
          aiir::arith::CmpIOp::create(
              builder, loc, aiir::arith::CmpIPredicate::ne,
              aiir::arith::AndIOp::create(builder, loc, intVal,
                                          lowSignificandMask),
              zero),
          createIntegerConstant(1ULL << --pos), zero));

  // [h] high-order significand (1 or 2 bits)
  index = aiir::arith::OrIOp::create(
      builder, loc, index,
      aiir::arith::AndIOp::create(
          builder, loc,
          aiir::arith::ShRUIOp::create(builder, loc, intVal,
                                       highSignificandShift),
          createIntegerConstant((1 << highSignificandSize) - 1)));

  int tableSize = 1 << (4 + highSignificandSize);
  aiir::Type int8Ty = builder.getIntegerType(8);
  aiir::Type tableTy = fir::SequenceType::get(tableSize, int8Ty);
  if (!builder.getNamedGlobal(tableName)) {
    llvm::SmallVector<aiir::Attribute, 64> values;
    auto insert = [&](std::int8_t which) {
      values.push_back(builder.getIntegerAttr(int8Ty, which));
    };
    // If indexing value [e] is 0, value [m] can't be 1. (If the exponent is 0,
    // it can't be the max exponent). Use IEEE_OTHER_VALUE for impossible
    // combinations.
    constexpr std::int8_t impossible = _FORTRAN_RUNTIME_IEEE_OTHER_VALUE;
    if (tableSize == 32) {
      //   s   e m   l h     kinds 2,3,4,8,16
      //   ===================================================================
      /*   0   0 0   0 0  */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_ZERO);
      /*   0   0 0   0 1  */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_SUBNORMAL);
      /*   0   0 0   1 0  */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_SUBNORMAL);
      /*   0   0 0   1 1  */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_SUBNORMAL);
      /*   0   0 1   0 0  */ insert(impossible);
      /*   0   0 1   0 1  */ insert(impossible);
      /*   0   0 1   1 0  */ insert(impossible);
      /*   0   0 1   1 1  */ insert(impossible);
      /*   0   1 0   0 0  */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_NORMAL);
      /*   0   1 0   0 1  */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_NORMAL);
      /*   0   1 0   1 0  */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_NORMAL);
      /*   0   1 0   1 1  */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_NORMAL);
      /*   0   1 1   0 0  */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_INF);
      /*   0   1 1   0 1  */ insert(_FORTRAN_RUNTIME_IEEE_QUIET_NAN);
      /*   0   1 1   1 0  */ insert(_FORTRAN_RUNTIME_IEEE_SIGNALING_NAN);
      /*   0   1 1   1 1  */ insert(_FORTRAN_RUNTIME_IEEE_QUIET_NAN);
      /*   1   0 0   0 0  */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_ZERO);
      /*   1   0 0   0 1  */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_SUBNORMAL);
      /*   1   0 0   1 0  */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_SUBNORMAL);
      /*   1   0 0   1 1  */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_SUBNORMAL);
      /*   1   0 1   0 0  */ insert(impossible);
      /*   1   0 1   0 1  */ insert(impossible);
      /*   1   0 1   1 0  */ insert(impossible);
      /*   1   0 1   1 1  */ insert(impossible);
      /*   1   1 0   0 0  */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_NORMAL);
      /*   1   1 0   0 1  */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_NORMAL);
      /*   1   1 0   1 0  */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_NORMAL);
      /*   1   1 0   1 1  */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_NORMAL);
      /*   1   1 1   0 0  */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_INF);
      /*   1   1 1   0 1  */ insert(_FORTRAN_RUNTIME_IEEE_QUIET_NAN);
      /*   1   1 1   1 0  */ insert(_FORTRAN_RUNTIME_IEEE_SIGNALING_NAN);
      /*   1   1 1   1 1  */ insert(_FORTRAN_RUNTIME_IEEE_QUIET_NAN);
    } else {
      // Unlike values of other kinds, kind=10 values can be "invalid", and
      // can appear in code. Use IEEE_OTHER_VALUE for invalid bit patterns.
      // Runtime IO may print an invalid value as a NaN.
      constexpr std::int8_t invalid = _FORTRAN_RUNTIME_IEEE_OTHER_VALUE;
      //   s   e m   l  h    kind 10
      //   ===================================================================
      /*   0   0 0   0 00 */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_ZERO);
      /*   0   0 0   0 01 */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_SUBNORMAL);
      /*   0   0 0   0 10 */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_SUBNORMAL);
      /*   0   0 0   0 11 */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_SUBNORMAL);
      /*   0   0 0   1 00 */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_SUBNORMAL);
      /*   0   0 0   1 01 */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_SUBNORMAL);
      /*   0   0 0   1 10 */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_SUBNORMAL);
      /*   0   0 0   1 11 */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_SUBNORMAL);
      /*   0   0 1   0 00 */ insert(impossible);
      /*   0   0 1   0 01 */ insert(impossible);
      /*   0   0 1   0 10 */ insert(impossible);
      /*   0   0 1   0 11 */ insert(impossible);
      /*   0   0 1   1 00 */ insert(impossible);
      /*   0   0 1   1 01 */ insert(impossible);
      /*   0   0 1   1 10 */ insert(impossible);
      /*   0   0 1   1 11 */ insert(impossible);
      /*   0   1 0   0 00 */ insert(invalid);
      /*   0   1 0   0 01 */ insert(invalid);
      /*   0   1 0   0 10 */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_NORMAL);
      /*   0   1 0   0 11 */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_NORMAL);
      /*   0   1 0   1 00 */ insert(invalid);
      /*   0   1 0   1 01 */ insert(invalid);
      /*   0   1 0   1 10 */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_NORMAL);
      /*   0   1 0   1 11 */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_NORMAL);
      /*   0   1 1   0 00 */ insert(invalid);
      /*   0   1 1   0 01 */ insert(invalid);
      /*   0   1 1   0 10 */ insert(_FORTRAN_RUNTIME_IEEE_POSITIVE_INF);
      /*   0   1 1   0 11 */ insert(_FORTRAN_RUNTIME_IEEE_QUIET_NAN);
      /*   0   1 1   1 00 */ insert(invalid);
      /*   0   1 1   1 01 */ insert(invalid);
      /*   0   1 1   1 10 */ insert(_FORTRAN_RUNTIME_IEEE_SIGNALING_NAN);
      /*   0   1 1   1 11 */ insert(_FORTRAN_RUNTIME_IEEE_QUIET_NAN);
      /*   1   0 0   0 00 */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_ZERO);
      /*   1   0 0   0 01 */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_SUBNORMAL);
      /*   1   0 0   0 10 */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_SUBNORMAL);
      /*   1   0 0   0 11 */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_SUBNORMAL);
      /*   1   0 0   1 00 */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_SUBNORMAL);
      /*   1   0 0   1 01 */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_SUBNORMAL);
      /*   1   0 0   1 10 */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_SUBNORMAL);
      /*   1   0 0   1 11 */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_SUBNORMAL);
      /*   1   0 1   0 00 */ insert(impossible);
      /*   1   0 1   0 01 */ insert(impossible);
      /*   1   0 1   0 10 */ insert(impossible);
      /*   1   0 1   0 11 */ insert(impossible);
      /*   1   0 1   1 00 */ insert(impossible);
      /*   1   0 1   1 01 */ insert(impossible);
      /*   1   0 1   1 10 */ insert(impossible);
      /*   1   0 1   1 11 */ insert(impossible);
      /*   1   1 0   0 00 */ insert(invalid);
      /*   1   1 0   0 01 */ insert(invalid);
      /*   1   1 0   0 10 */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_NORMAL);
      /*   1   1 0   0 11 */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_NORMAL);
      /*   1   1 0   1 00 */ insert(invalid);
      /*   1   1 0   1 01 */ insert(invalid);
      /*   1   1 0   1 10 */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_NORMAL);
      /*   1   1 0   1 11 */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_NORMAL);
      /*   1   1 1   0 00 */ insert(invalid);
      /*   1   1 1   0 01 */ insert(invalid);
      /*   1   1 1   0 10 */ insert(_FORTRAN_RUNTIME_IEEE_NEGATIVE_INF);
      /*   1   1 1   0 11 */ insert(_FORTRAN_RUNTIME_IEEE_QUIET_NAN);
      /*   1   1 1   1 00 */ insert(invalid);
      /*   1   1 1   1 01 */ insert(invalid);
      /*   1   1 1   1 10 */ insert(_FORTRAN_RUNTIME_IEEE_SIGNALING_NAN);
      /*   1   1 1   1 11 */ insert(_FORTRAN_RUNTIME_IEEE_QUIET_NAN);
    }
    builder.createGlobalConstant(
        loc, tableTy, tableName, builder.createLinkOnceLinkage(),
        aiir::DenseElementsAttr::get(
            aiir::RankedTensorType::get(tableSize, int8Ty), values));
  }

  return fir::CoordinateOp::create(
      builder, loc, builder.getRefType(resultType),
      fir::AddrOfOp::create(builder, loc, builder.getRefType(tableTy),
                            builder.getSymbolRefAttr(tableName)),
      index);
}

// IEEE_COPY_SIGN
aiir::Value
IntrinsicLibrary::genIeeeCopySign(aiir::Type resultType,
                                  llvm::ArrayRef<aiir::Value> args) {
  // Copy the sign of REAL arg Y to REAL arg X.
  assert(args.size() == 2);
  aiir::Value xRealVal = args[0];
  aiir::Value yRealVal = args[1];
  aiir::FloatType xRealType =
      aiir::dyn_cast<aiir::FloatType>(xRealVal.getType());
  aiir::FloatType yRealType =
      aiir::dyn_cast<aiir::FloatType>(yRealVal.getType());

  if (yRealType == aiir::BFloat16Type::get(builder.getContext())) {
    // Workaround: CopySignOp and BitcastOp don't work for kind 3 arg Y.
    // This conversion should always preserve the sign bit.
    yRealVal = builder.createConvert(
        loc, aiir::Float32Type::get(builder.getContext()), yRealVal);
    yRealType = aiir::Float32Type::get(builder.getContext());
  }

  // Args have the same type.
  if (xRealType == yRealType)
    return aiir::math::CopySignOp::create(builder, loc, xRealVal, yRealVal);

  // Args have different types.
  aiir::Type xIntType = builder.getIntegerType(xRealType.getWidth());
  aiir::Type yIntType = builder.getIntegerType(yRealType.getWidth());
  aiir::Value xIntVal =
      aiir::arith::BitcastOp::create(builder, loc, xIntType, xRealVal);
  aiir::Value yIntVal =
      aiir::arith::BitcastOp::create(builder, loc, yIntType, yRealVal);
  aiir::Value xZero = builder.createIntegerConstant(loc, xIntType, 0);
  aiir::Value yZero = builder.createIntegerConstant(loc, yIntType, 0);
  aiir::Value xOne = builder.createIntegerConstant(loc, xIntType, 1);
  aiir::Value ySign = aiir::arith::ShRUIOp::create(
      builder, loc, yIntVal,
      builder.createIntegerConstant(loc, yIntType, yRealType.getWidth() - 1));
  aiir::Value xAbs = aiir::arith::ShRUIOp::create(
      builder, loc, aiir::arith::ShLIOp::create(builder, loc, xIntVal, xOne),
      xOne);
  aiir::Value xSign = aiir::arith::SelectOp::create(
      builder, loc,
      aiir::arith::CmpIOp::create(builder, loc, aiir::arith::CmpIPredicate::eq,
                                  ySign, yZero),
      xZero,
      aiir::arith::ShLIOp::create(
          builder, loc, xOne,
          builder.createIntegerConstant(loc, xIntType,
                                        xRealType.getWidth() - 1)));
  return aiir::arith::BitcastOp::create(
      builder, loc, xRealType,
      aiir::arith::OrIOp::create(builder, loc, xAbs, xSign));
}

// IEEE_GET_FLAG
void IntrinsicLibrary::genIeeeGetFlag(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  // Set FLAG_VALUE=.TRUE. if the exception specified by FLAG is signaling.
  aiir::Value flag = fir::getBase(args[0]);
  aiir::Value flagValue = fir::getBase(args[1]);
  aiir::Type resultTy =
      aiir::dyn_cast<fir::ReferenceType>(flagValue.getType()).getEleTy();
  aiir::Type i32Ty = builder.getIntegerType(32);
  aiir::Value zero = builder.createIntegerConstant(loc, i32Ty, 0);
  auto [fieldRef, ignore] = getFieldRef(builder, loc, flag);
  aiir::Value field = fir::LoadOp::create(builder, loc, fieldRef);
  aiir::Value excepts = fir::runtime::genFetestexcept(
      builder, loc,
      fir::runtime::genMapExcept(
          builder, loc, fir::ConvertOp::create(builder, loc, i32Ty, field)));
  aiir::Value logicalResult = fir::ConvertOp::create(
      builder, loc, resultTy,
      aiir::arith::CmpIOp::create(builder, loc, aiir::arith::CmpIPredicate::ne,
                                  excepts, zero));
  fir::StoreOp::create(builder, loc, logicalResult, flagValue);
}

// IEEE_GET_HALTING_MODE
void IntrinsicLibrary::genIeeeGetHaltingMode(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  // Set HALTING=.TRUE. if the exception specified by FLAG will cause halting.
  assert(args.size() == 2);
  aiir::Value flag = fir::getBase(args[0]);
  aiir::Value halting = fir::getBase(args[1]);
  aiir::Type resultTy =
      aiir::dyn_cast<fir::ReferenceType>(halting.getType()).getEleTy();
  aiir::Type i32Ty = builder.getIntegerType(32);
  aiir::Value zero = builder.createIntegerConstant(loc, i32Ty, 0);
  auto [fieldRef, ignore] = getFieldRef(builder, loc, flag);
  aiir::Value field = fir::LoadOp::create(builder, loc, fieldRef);
  aiir::Value haltSet = fir::runtime::genFegetexcept(builder, loc);
  aiir::Value intResult = aiir::arith::AndIOp::create(
      builder, loc, haltSet,
      fir::runtime::genMapExcept(
          builder, loc, fir::ConvertOp::create(builder, loc, i32Ty, field)));
  aiir::Value logicalResult = fir::ConvertOp::create(
      builder, loc, resultTy,
      aiir::arith::CmpIOp::create(builder, loc, aiir::arith::CmpIPredicate::ne,
                                  intResult, zero));
  fir::StoreOp::create(builder, loc, logicalResult, halting);
}

// IEEE_GET_MODES, IEEE_SET_MODES
// IEEE_GET_STATUS, IEEE_SET_STATUS
template <bool isGet, bool isModes>
void IntrinsicLibrary::genIeeeGetOrSetModesOrStatus(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
#ifndef __GLIBC_USE_IEC_60559_BFP_EXT // only use of "#include <cfenv>"
  // No definitions of fegetmode, fesetmode
  llvm::StringRef func = isModes
                             ? (isGet ? "ieee_get_modes" : "ieee_set_modes")
                             : (isGet ? "ieee_get_status" : "ieee_set_status");
  TODO(loc, "intrinsic module procedure: " + func);
#else
  aiir::Type i32Ty = builder.getIntegerType(32);
  aiir::Type i64Ty = builder.getIntegerType(64);
  aiir::Type ptrTy = builder.getRefType(i32Ty);
  aiir::Value addr;
  if (fir::getTargetTriple(builder.getModule()).isSPARC()) {
    // Floating point environment data is larger than the __data field
    // allotment. Allocate data space from the heap.
    auto [fieldRef, fieldTy] =
        getFieldRef(builder, loc, fir::getBase(args[0]), 1);
    addr = fir::BoxAddrOp::create(builder, loc,
                                  fir::LoadOp::create(builder, loc, fieldRef));
    aiir::Type heapTy = addr.getType();
    aiir::Value allocated = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::ne,
        builder.createConvert(loc, i64Ty, addr),
        builder.createIntegerConstant(loc, i64Ty, 0));
    auto ifOp = fir::IfOp::create(builder, loc, heapTy, allocated,
                                  /*withElseRegion=*/true);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    fir::ResultOp::create(builder, loc, addr);
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    aiir::Value byteSize =
        isModes ? fir::runtime::genGetModesTypeSize(builder, loc)
                : fir::runtime::genGetStatusTypeSize(builder, loc);
    byteSize = builder.createConvert(loc, builder.getIndexType(), byteSize);
    addr = fir::AllocMemOp::create(builder, loc, extractSequenceType(heapTy),
                                   /*typeparams=*/aiir::ValueRange(), byteSize);
    aiir::Value shape = fir::ShapeOp::create(builder, loc, byteSize);
    fir::StoreOp::create(
        builder, loc, fir::EmboxOp::create(builder, loc, fieldTy, addr, shape),
        fieldRef);
    fir::ResultOp::create(builder, loc, addr);
    builder.setInsertionPointAfter(ifOp);
    addr = fir::ConvertOp::create(builder, loc, ptrTy, ifOp.getResult(0));
  } else {
    // Place floating point environment data in __data storage.
    addr = fir::ConvertOp::create(builder, loc, ptrTy, getBase(args[0]));
  }
  llvm::StringRef func = isModes ? (isGet ? "fegetmode" : "fesetmode")
                                 : (isGet ? "fegetenv" : "fesetenv");
  genRuntimeCall(func, i32Ty, addr);
#endif
}

// Check that an explicit ieee_[get|set]_rounding_mode call radix value is 2.
static void checkRadix(fir::FirOpBuilder &builder, aiir::Location loc,
                       aiir::Value radix, std::string procName) {
  aiir::Value notTwo = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::ne, radix,
      builder.createIntegerConstant(loc, radix.getType(), 2));
  auto ifOp = fir::IfOp::create(builder, loc, notTwo,
                                /*withElseRegion=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  fir::runtime::genReportFatalUserError(builder, loc,
                                        procName + " radix argument must be 2");
  builder.setInsertionPointAfter(ifOp);
}

// IEEE_GET_ROUNDING_MODE
void IntrinsicLibrary::genIeeeGetRoundingMode(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  // Set arg ROUNDING_VALUE to the current floating point rounding mode.
  // Values are chosen to match the llvm.get.rounding encoding.
  // Generate an error if the value of optional arg RADIX is not 2.
  assert(args.size() == 1 || args.size() == 2);
  if (args.size() == 2)
    checkRadix(builder, loc, fir::getBase(args[1]), "ieee_get_rounding_mode");
  auto [fieldRef, fieldTy] = getFieldRef(builder, loc, fir::getBase(args[0]));
  aiir::func::FuncOp getRound = fir::factory::getLlvmGetRounding(builder);
  aiir::Value mode = fir::CallOp::create(builder, loc, getRound).getResult(0);
  mode = builder.createConvert(loc, fieldTy, mode);
  fir::StoreOp::create(builder, loc, mode, fieldRef);
}

// IEEE_GET_UNDERFLOW_MODE
void IntrinsicLibrary::genIeeeGetUnderflowMode(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  aiir::Value flag = fir::runtime::genGetUnderflowMode(builder, loc);
  builder.createStoreWithConvert(loc, flag, fir::getBase(args[0]));
}

// IEEE_INT
aiir::Value IntrinsicLibrary::genIeeeInt(aiir::Type resultType,
                                         llvm::ArrayRef<aiir::Value> args) {
  // Convert real argument A to an integer, with rounding according to argument
  // ROUND. Signal IEEE_INVALID if A is a NaN, an infinity, or out of range,
  // and return either the largest or smallest integer result value (*).
  // For valid results (when IEEE_INVALID is not signaled), signal IEEE_INEXACT
  // if A is not an exact integral value (*). The (*) choices are processor
  // dependent implementation choices not mandated by the standard.
  // The primary result is generated with a call to IEEE_RINT.
  assert(args.size() == 3);
  aiir::FloatType realType = aiir::cast<aiir::FloatType>(args[0].getType());
  aiir::Value realResult = genIeeeRint(realType, {args[0], args[1]});
  int intWidth = aiir::cast<aiir::IntegerType>(resultType).getWidth();
  aiir::Value intLBound = aiir::arith::ConstantOp::create(
      builder, loc, resultType,
      builder.getIntegerAttr(resultType,
                             llvm::APInt::getBitsSet(intWidth,
                                                     /*lo=*/intWidth - 1,
                                                     /*hi=*/intWidth)));
  aiir::Value intUBound = aiir::arith::ConstantOp::create(
      builder, loc, resultType,
      builder.getIntegerAttr(resultType,
                             llvm::APInt::getBitsSet(intWidth, /*lo=*/0,
                                                     /*hi=*/intWidth - 1)));
  aiir::Value realLBound =
      fir::ConvertOp::create(builder, loc, realType, intLBound);
  aiir::Value realUBound =
      aiir::arith::NegFOp::create(builder, loc, realLBound);
  aiir::Value aGreaterThanLBound = aiir::arith::CmpFOp::create(
      builder, loc, aiir::arith::CmpFPredicate::OGE, realResult, realLBound);
  aiir::Value aLessThanUBound = aiir::arith::CmpFOp::create(
      builder, loc, aiir::arith::CmpFPredicate::OLT, realResult, realUBound);
  aiir::Value resultIsValid = aiir::arith::AndIOp::create(
      builder, loc, aGreaterThanLBound, aLessThanUBound);

  // Result is valid. It may be exact or inexact.
  aiir::Value result;
  fir::IfOp ifOp = fir::IfOp::create(builder, loc, resultType, resultIsValid,
                                     /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  aiir::Value inexact = aiir::arith::CmpFOp::create(
      builder, loc, aiir::arith::CmpFPredicate::ONE, args[0], realResult);
  genRaiseExcept(_FORTRAN_RUNTIME_IEEE_INEXACT, inexact);
  result = fir::ConvertOp::create(builder, loc, resultType, realResult);
  fir::ResultOp::create(builder, loc, result);

  // Result is invalid.
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  genRaiseExcept(_FORTRAN_RUNTIME_IEEE_INVALID);
  result = aiir::arith::SelectOp::create(builder, loc, aGreaterThanLBound,
                                         intUBound, intLBound);
  fir::ResultOp::create(builder, loc, result);
  builder.setInsertionPointAfter(ifOp);
  return ifOp.getResult(0);
}

// IEEE_IS_FINITE
aiir::Value
IntrinsicLibrary::genIeeeIsFinite(aiir::Type resultType,
                                  llvm::ArrayRef<aiir::Value> args) {
  // Check if arg X is a (negative or positive) (normal, denormal, or zero).
  assert(args.size() == 1);
  return genIsFPClass(resultType, args, finiteTest);
}

// IEEE_IS_NAN
aiir::Value IntrinsicLibrary::genIeeeIsNan(aiir::Type resultType,
                                           llvm::ArrayRef<aiir::Value> args) {
  // Check if arg X is a (signaling or quiet) NaN.
  assert(args.size() == 1);
  return genIsFPClass(resultType, args, nanTest);
}

// IEEE_IS_NEGATIVE
aiir::Value
IntrinsicLibrary::genIeeeIsNegative(aiir::Type resultType,
                                    llvm::ArrayRef<aiir::Value> args) {
  // Check if arg X is a negative (infinity, normal, denormal or zero).
  assert(args.size() == 1);
  return genIsFPClass(resultType, args, negativeTest);
}

// IEEE_IS_NORMAL
aiir::Value
IntrinsicLibrary::genIeeeIsNormal(aiir::Type resultType,
                                  llvm::ArrayRef<aiir::Value> args) {
  // Check if arg X is a (negative or positive) (normal or zero).
  assert(args.size() == 1);
  return genIsFPClass(resultType, args, normalTest);
}

// IEEE_LOGB
aiir::Value IntrinsicLibrary::genIeeeLogb(aiir::Type resultType,
                                          llvm::ArrayRef<aiir::Value> args) {
  // Exponent of X, with special case treatment for some input values.
  // Return: X == 0
  //             ? -infinity (and raise FE_DIVBYZERO)
  //             : ieee_is_finite(X)
  //                 ? exponent(X) - 1        // unbiased exponent of X
  //                 : ieee_copy_sign(X, 1.0) // +infinity or NaN
  assert(args.size() == 1);
  aiir::Value realVal = args[0];
  aiir::FloatType realType = aiir::dyn_cast<aiir::FloatType>(realVal.getType());
  int bitWidth = realType.getWidth();
  aiir::Type intType = builder.getIntegerType(realType.getWidth());
  aiir::Value intVal =
      aiir::arith::BitcastOp::create(builder, loc, intType, realVal);
  aiir::Type i1Ty = builder.getI1Type();

  int exponentBias, significandSize, nonSignificandSize;
  switch (bitWidth) {
  case 16:
    if (realType.isF16()) {
      // kind=2: 1 sign bit, 5 exponent bits, 10 significand bits
      exponentBias = (1 << (5 - 1)) - 1; // 15
      significandSize = 10;
      nonSignificandSize = 6;
      break;
    }
    assert(realType.isBF16() && "unknown 16-bit real type");
    // kind=3: 1 sign bit, 8 exponent bits, 7 significand bits
    exponentBias = (1 << (8 - 1)) - 1; // 127
    significandSize = 7;
    nonSignificandSize = 9;
    break;
  case 32:
    // kind=4: 1 sign bit, 8 exponent bits, 23 significand bits
    exponentBias = (1 << (8 - 1)) - 1; // 127
    significandSize = 23;
    nonSignificandSize = 9;
    break;
  case 64:
    // kind=8: 1 sign bit, 11 exponent bits, 52 significand bits
    exponentBias = (1 << (11 - 1)) - 1; // 1023
    significandSize = 52;
    nonSignificandSize = 12;
    break;
  case 80:
    // kind=10: 1 sign bit, 15 exponent bits, 1+63 significand bits
    exponentBias = (1 << (15 - 1)) - 1; // 16383
    significandSize = 64;
    nonSignificandSize = 16 + 1;
    break;
  case 128:
    // kind=16: 1 sign bit, 15 exponent bits, 112 significand bits
    exponentBias = (1 << (15 - 1)) - 1; // 16383
    significandSize = 112;
    nonSignificandSize = 16;
    break;
  default:
    llvm_unreachable("unknown real type");
  }

  aiir::Value isZero = aiir::arith::CmpFOp::create(
      builder, loc, aiir::arith::CmpFPredicate::OEQ, realVal,
      builder.createRealZeroConstant(loc, resultType));
  auto outerIfOp = fir::IfOp::create(builder, loc, resultType, isZero,
                                     /*withElseRegion=*/true);
  // X is zero -- result is -infinity
  builder.setInsertionPointToStart(&outerIfOp.getThenRegion().front());
  genRaiseExcept(_FORTRAN_RUNTIME_IEEE_DIVIDE_BY_ZERO);
  aiir::Value ones = builder.createAllOnesInteger(loc, intType);
  aiir::Value result = aiir::arith::ShLIOp::create(
      builder, loc, ones,
      builder.createIntegerConstant(loc, intType,
                                    // kind=10 high-order bit is explicit
                                    significandSize - (bitWidth == 80)));
  result = aiir::arith::BitcastOp::create(builder, loc, resultType, result);
  fir::ResultOp::create(builder, loc, result);

  builder.setInsertionPointToStart(&outerIfOp.getElseRegion().front());
  aiir::Value one = builder.createIntegerConstant(loc, intType, 1);
  aiir::Value shiftLeftOne =
      aiir::arith::ShLIOp::create(builder, loc, intVal, one);
  aiir::Value isFinite = genIsFPClass(i1Ty, args, finiteTest);
  auto innerIfOp = fir::IfOp::create(builder, loc, resultType, isFinite,
                                     /*withElseRegion=*/true);
  // X is non-zero finite -- result is unbiased exponent of X
  builder.setInsertionPointToStart(&innerIfOp.getThenRegion().front());
  aiir::Value isNormal = genIsFPClass(i1Ty, args, normalTest);
  auto normalIfOp = fir::IfOp::create(builder, loc, resultType, isNormal,
                                      /*withElseRegion=*/true);
  // X is normal
  builder.setInsertionPointToStart(&normalIfOp.getThenRegion().front());
  aiir::Value biasedExponent = aiir::arith::ShRUIOp::create(
      builder, loc, shiftLeftOne,
      builder.createIntegerConstant(loc, intType, significandSize + 1));
  result = aiir::arith::SubIOp::create(
      builder, loc, biasedExponent,
      builder.createIntegerConstant(loc, intType, exponentBias));
  result = fir::ConvertOp::create(builder, loc, resultType, result);
  fir::ResultOp::create(builder, loc, result);

  // X is denormal -- result is (-exponentBias - ctlz(significand))
  builder.setInsertionPointToStart(&normalIfOp.getElseRegion().front());
  aiir::Value significand = aiir::arith::ShLIOp::create(
      builder, loc, intVal,
      builder.createIntegerConstant(loc, intType, nonSignificandSize));
  aiir::Value ctlz =
      aiir::math::CountLeadingZerosOp::create(builder, loc, significand);
  aiir::Type i32Ty = builder.getI32Type();
  result = aiir::arith::SubIOp::create(
      builder, loc, builder.createIntegerConstant(loc, i32Ty, -exponentBias),
      fir::ConvertOp::create(builder, loc, i32Ty, ctlz));
  result = fir::ConvertOp::create(builder, loc, resultType, result);
  fir::ResultOp::create(builder, loc, result);

  builder.setInsertionPointToEnd(&innerIfOp.getThenRegion().front());
  fir::ResultOp::create(builder, loc, normalIfOp.getResult(0));

  // X is infinity or NaN -- result is +infinity or NaN
  builder.setInsertionPointToStart(&innerIfOp.getElseRegion().front());
  result = aiir::arith::ShRUIOp::create(builder, loc, shiftLeftOne, one);
  result = aiir::arith::BitcastOp::create(builder, loc, resultType, result);
  fir::ResultOp::create(builder, loc, result);

  // Unwind the if nest.
  builder.setInsertionPointToEnd(&outerIfOp.getElseRegion().front());
  fir::ResultOp::create(builder, loc, innerIfOp.getResult(0));
  builder.setInsertionPointAfter(outerIfOp);
  return outerIfOp.getResult(0);
}

// IEEE_MAX, IEEE_MAX_MAG, IEEE_MAX_NUM, IEEE_MAX_NUM_MAG
// IEEE_MIN, IEEE_MIN_MAG, IEEE_MIN_NUM, IEEE_MIN_NUM_MAG
template <bool isMax, bool isNum, bool isMag>
aiir::Value IntrinsicLibrary::genIeeeMaxMin(aiir::Type resultType,
                                            llvm::ArrayRef<aiir::Value> args) {
  // Maximum/minimum of X and Y with special case treatment of NaN operands.
  // The f18 definitions of these procedures (where applicable) are incomplete.
  // And f18 results involving NaNs are different from and incompatible with
  // f23 results. This code implements the f23 procedures.
  // For IEEE_MAX_MAG and IEEE_MAX_NUM_MAG:
  //   if (ABS(X) > ABS(Y))
  //     return X
  //   else if (ABS(Y) > ABS(X))
  //     return Y
  //   else if (ABS(X) == ABS(Y))
  //     return IEEE_SIGNBIT(Y) ? X : Y
  //   // X or Y or both are NaNs
  //   if (X is an sNaN or Y is an sNaN) raise FE_INVALID
  //   if (IEEE_MAX_NUM_MAG and X is not a NaN) return X
  //   if (IEEE_MAX_NUM_MAG and Y is not a NaN) return Y
  //   return a qNaN
  // For IEEE_MAX, IEEE_MAX_NUM: compare X vs. Y rather than ABS(X) vs. ABS(Y)
  // IEEE_MIN, IEEE_MIN_MAG, IEEE_MIN_NUM, IEEE_MIN_NUM_MAG: invert comparisons
  assert(args.size() == 2);
  aiir::Value x = args[0];
  aiir::Value y = args[1];
  aiir::Value x1, y1; // X or ABS(X), Y or ABS(Y)
  if constexpr (isMag) {
    aiir::Value zero = builder.createRealZeroConstant(loc, resultType);
    x1 = aiir::math::CopySignOp::create(builder, loc, x, zero);
    y1 = aiir::math::CopySignOp::create(builder, loc, y, zero);
  } else {
    x1 = x;
    y1 = y;
  }
  aiir::Type i1Ty = builder.getI1Type();
  aiir::arith::CmpFPredicate pred;
  aiir::Value cmp, result, resultIsX, resultIsY;

  // X1 < Y1 -- MAX result is Y; MIN result is X.
  pred = aiir::arith::CmpFPredicate::OLT;
  cmp = aiir::arith::CmpFOp::create(builder, loc, pred, x1, y1);
  auto ifOp1 = fir::IfOp::create(builder, loc, resultType, cmp, true);
  builder.setInsertionPointToStart(&ifOp1.getThenRegion().front());
  result = isMax ? y : x;
  fir::ResultOp::create(builder, loc, result);

  // X1 > Y1 -- MAX result is X; MIN result is Y.
  builder.setInsertionPointToStart(&ifOp1.getElseRegion().front());
  pred = aiir::arith::CmpFPredicate::OGT;
  cmp = aiir::arith::CmpFOp::create(builder, loc, pred, x1, y1);
  auto ifOp2 = fir::IfOp::create(builder, loc, resultType, cmp, true);
  builder.setInsertionPointToStart(&ifOp2.getThenRegion().front());
  result = isMax ? x : y;
  fir::ResultOp::create(builder, loc, result);

  // X1 == Y1 -- MAX favors a positive result; MIN favors a negative result.
  builder.setInsertionPointToStart(&ifOp2.getElseRegion().front());
  pred = aiir::arith::CmpFPredicate::OEQ;
  cmp = aiir::arith::CmpFOp::create(builder, loc, pred, x1, y1);
  auto ifOp3 = fir::IfOp::create(builder, loc, resultType, cmp, true);
  builder.setInsertionPointToStart(&ifOp3.getThenRegion().front());
  resultIsX = isMax ? genIsFPClass(i1Ty, x, positiveTest)
                    : genIsFPClass(i1Ty, x, negativeTest);
  result = aiir::arith::SelectOp::create(builder, loc, resultIsX, x, y);
  fir::ResultOp::create(builder, loc, result);

  // X or Y or both are NaNs -- result may be X, Y, or a qNaN
  builder.setInsertionPointToStart(&ifOp3.getElseRegion().front());
  if constexpr (isNum) {
    pred = aiir::arith::CmpFPredicate::ORD; // check for a non-NaN
    resultIsX = aiir::arith::CmpFOp::create(builder, loc, pred, x, x);
    resultIsY = aiir::arith::CmpFOp::create(builder, loc, pred, y, y);
  } else {
    resultIsX = resultIsY = builder.createBool(loc, false);
  }
  result = aiir::arith::SelectOp::create(
      builder, loc, resultIsX, x,
      aiir::arith::SelectOp::create(builder, loc, resultIsY, y,
                                    genQNan(resultType)));
  aiir::Value hasSNaNOp = aiir::arith::OrIOp::create(
      builder, loc, genIsFPClass(builder.getI1Type(), args[0], snanTest),
      genIsFPClass(builder.getI1Type(), args[1], snanTest));
  genRaiseExcept(_FORTRAN_RUNTIME_IEEE_INVALID, hasSNaNOp);
  fir::ResultOp::create(builder, loc, result);

  // Unwind the if nest.
  builder.setInsertionPointAfter(ifOp3);
  fir::ResultOp::create(builder, loc, ifOp3.getResult(0));
  builder.setInsertionPointAfter(ifOp2);
  fir::ResultOp::create(builder, loc, ifOp2.getResult(0));
  builder.setInsertionPointAfter(ifOp1);
  return ifOp1.getResult(0);
}

// IEEE_QUIET_EQ, IEEE_QUIET_GE, IEEE_QUIET_GT,
// IEEE_QUIET_LE, IEEE_QUIET_LT, IEEE_QUIET_NE
template <aiir::arith::CmpFPredicate pred>
aiir::Value
IntrinsicLibrary::genIeeeQuietCompare(aiir::Type resultType,
                                      llvm::ArrayRef<aiir::Value> args) {
  // Compare X and Y with special case treatment of NaN operands.
  assert(args.size() == 2);
  aiir::Value hasSNaNOp = aiir::arith::OrIOp::create(
      builder, loc, genIsFPClass(builder.getI1Type(), args[0], snanTest),
      genIsFPClass(builder.getI1Type(), args[1], snanTest));
  aiir::Value res =
      aiir::arith::CmpFOp::create(builder, loc, pred, args[0], args[1]);
  genRaiseExcept(_FORTRAN_RUNTIME_IEEE_INVALID, hasSNaNOp);
  return fir::ConvertOp::create(builder, loc, resultType, res);
}

// IEEE_REAL
aiir::Value IntrinsicLibrary::genIeeeReal(aiir::Type resultType,
                                          llvm::ArrayRef<aiir::Value> args) {
  // Convert integer or real argument A to a real of a specified kind.
  // Round according to the current rounding mode.
  // Signal IEEE_INVALID if A is an sNaN, and return a qNaN.
  // Signal IEEE_UNDERFLOW for an inexact subnormal or zero result.
  // Signal IEEE_OVERFLOW if A is finite and the result is infinite.
  // Signal IEEE_INEXACT for an inexact result.
  //
  // if (type(a) == resultType) {
  //   // Conversion to the same type is a nop except for sNaN processing.
  //   result = a
  // } else {
  //   result = r = real(a, kind(result))
  //   // Conversion to a larger type is exact.
  //   if (c_sizeof(a) >= c_sizeof(r)) {
  //     b = (a is integer) ? int(r, kind(a)) : real(r, kind(a))
  //     if (a == b || isNaN(a)) {
  //       // a is {-0, +0, -inf, +inf, NaN} or exact; result is r
  //     } else {
  //       // odd(r) is true if the low bit of significand(r) is 1
  //       // rounding mode ieee_other is an alias for mode ieee_nearest
  //       if (a < b) {
  //         if (mode == ieee_nearest && odd(r)) result = ieee_next_down(r)
  //         if (mode == ieee_other   && odd(r)) result = ieee_next_down(r)
  //         if (mode == ieee_to_zero && a > 0)  result = ieee_next_down(r)
  //         if (mode == ieee_away    && a < 0)  result = ieee_next_down(r)
  //         if (mode == ieee_down)              result = ieee_next_down(r)
  //       } else { // a > b
  //         if (mode == ieee_nearest && odd(r)) result = ieee_next_up(r)
  //         if (mode == ieee_other   && odd(r)) result = ieee_next_up(r)
  //         if (mode == ieee_to_zero && a < 0)  result = ieee_next_up(r)
  //         if (mode == ieee_away    && a > 0)  result = ieee_next_up(r)
  //         if (mode == ieee_up)                result = ieee_next_up(r)
  //       }
  //     }
  //   }
  // }

  assert(args.size() == 2);
  aiir::Type i1Ty = builder.getI1Type();
  aiir::Type f32Ty = aiir::Float32Type::get(builder.getContext());
  aiir::Value a = args[0];
  aiir::Type aType = a.getType();

  // If the argument is an sNaN, raise an invalid exception and return a qNaN.
  // Otherwise return the argument.
  auto processSnan = [&](aiir::Value x) {
    fir::IfOp ifOp = fir::IfOp::create(builder, loc, resultType,
                                       genIsFPClass(i1Ty, x, snanTest),
                                       /*withElseRegion=*/true);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    genRaiseExcept(_FORTRAN_RUNTIME_IEEE_INVALID);
    fir::ResultOp::create(builder, loc, genQNan(resultType));
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    fir::ResultOp::create(builder, loc, x);
    builder.setInsertionPointAfter(ifOp);
    return ifOp.getResult(0);
  };

  // Conversion is a nop, except that A may be an sNaN.
  if (resultType == aType)
    return processSnan(a);

  // Can't directly convert between kind=2 and kind=3.
  aiir::Value r, r1;
  if ((aType.isBF16() && resultType.isF16()) ||
      (aType.isF16() && resultType.isBF16())) {
    a = builder.createConvert(loc, f32Ty, a);
    aType = f32Ty;
  }
  r = fir::ConvertOp::create(builder, loc, resultType, a);

  aiir::IntegerType aIntType = aiir::dyn_cast<aiir::IntegerType>(aType);
  aiir::FloatType aFloatType = aiir::dyn_cast<aiir::FloatType>(aType);
  aiir::FloatType resultFloatType = aiir::dyn_cast<aiir::FloatType>(resultType);

  // Conversion from a smaller type to a larger type is exact.
  if ((aIntType ? aIntType.getWidth() : aFloatType.getWidth()) <
      resultFloatType.getWidth())
    return aIntType ? r : processSnan(r);

  // A possibly inexact conversion result may need to be rounded up or down.
  aiir::Value b = fir::ConvertOp::create(builder, loc, aType, r);
  aiir::Value aEqB;
  if (aIntType)
    aEqB = aiir::arith::CmpIOp::create(builder, loc,
                                       aiir::arith::CmpIPredicate::eq, a, b);
  else
    aEqB = aiir::arith::CmpFOp::create(builder, loc,
                                       aiir::arith::CmpFPredicate::UEQ, a, b);

  // [a == b] a is a NaN or r is exact (a may be -0, +0, -inf, +inf) -- return r
  fir::IfOp ifOp1 = fir::IfOp::create(builder, loc, resultType, aEqB,
                                      /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&ifOp1.getThenRegion().front());
  fir::ResultOp::create(builder, loc, aIntType ? r : processSnan(r));

  // Code common to (a < b) and (a > b) branches.
  builder.setInsertionPointToStart(&ifOp1.getElseRegion().front());
  aiir::func::FuncOp getRound = fir::factory::getLlvmGetRounding(builder);
  aiir::Value mode = fir::CallOp::create(builder, loc, getRound).getResult(0);
  aiir::Value aIsNegative, aIsPositive;
  if (aIntType) {
    aiir::Value zero = builder.createIntegerConstant(loc, aIntType, 0);
    aIsNegative = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::slt, a, zero);
    aIsPositive = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::sgt, a, zero);
  } else {
    aiir::Value zero = builder.createRealZeroConstant(loc, aFloatType);
    aIsNegative = aiir::arith::CmpFOp::create(
        builder, loc, aiir::arith::CmpFPredicate::OLT, a, zero);
    aIsPositive = aiir::arith::CmpFOp::create(
        builder, loc, aiir::arith::CmpFPredicate::OGT, a, zero);
  }
  aiir::Type resultIntType = builder.getIntegerType(resultFloatType.getWidth());
  aiir::Value resultCast =
      aiir::arith::BitcastOp::create(builder, loc, resultIntType, r);
  aiir::Value one = builder.createIntegerConstant(loc, resultIntType, 1);
  aiir::Value rIsOdd = fir::ConvertOp::create(
      builder, loc, i1Ty,
      aiir::arith::AndIOp::create(builder, loc, resultCast, one));
  // Check for a rounding mode match.
  auto match = [&](int m) {
    return aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::eq, mode,
        builder.createIntegerConstant(loc, mode.getType(), m));
  };
  aiir::Value roundToNearestBit = aiir::arith::OrIOp::create(
      builder, loc,
      // IEEE_OTHER is an alias for IEEE_NEAREST.
      match(_FORTRAN_RUNTIME_IEEE_NEAREST), match(_FORTRAN_RUNTIME_IEEE_OTHER));
  aiir::Value roundToNearest =
      aiir::arith::AndIOp::create(builder, loc, roundToNearestBit, rIsOdd);
  aiir::Value roundToZeroBit = match(_FORTRAN_RUNTIME_IEEE_TO_ZERO);
  aiir::Value roundAwayBit = match(_FORTRAN_RUNTIME_IEEE_AWAY);
  aiir::Value roundToZero, roundAway, mustAdjust;
  fir::IfOp adjustIfOp;
  aiir::Value aLtB;
  if (aIntType)
    aLtB = aiir::arith::CmpIOp::create(builder, loc,
                                       aiir::arith::CmpIPredicate::slt, a, b);
  else
    aLtB = aiir::arith::CmpFOp::create(builder, loc,
                                       aiir::arith::CmpFPredicate::OLT, a, b);
  aiir::Value upResult =
      aiir::arith::AddIOp::create(builder, loc, resultCast, one);
  aiir::Value downResult =
      aiir::arith::SubIOp::create(builder, loc, resultCast, one);

  // (a < b): r is inexact -- return r or ieee_next_down(r)
  fir::IfOp ifOp2 = fir::IfOp::create(builder, loc, resultType, aLtB,
                                      /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&ifOp2.getThenRegion().front());
  roundToZero =
      aiir::arith::AndIOp::create(builder, loc, roundToZeroBit, aIsPositive);
  roundAway =
      aiir::arith::AndIOp::create(builder, loc, roundAwayBit, aIsNegative);
  aiir::Value roundDown = match(_FORTRAN_RUNTIME_IEEE_DOWN);
  mustAdjust =
      aiir::arith::OrIOp::create(builder, loc, roundToNearest, roundToZero);
  mustAdjust = aiir::arith::OrIOp::create(builder, loc, mustAdjust, roundAway);
  mustAdjust = aiir::arith::OrIOp::create(builder, loc, mustAdjust, roundDown);
  adjustIfOp = fir::IfOp::create(builder, loc, resultType, mustAdjust,
                                 /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&adjustIfOp.getThenRegion().front());
  if (resultType.isF80())
    r1 = fir::runtime::genNearest(builder, loc, r,
                                  builder.createBool(loc, false));
  else
    r1 = aiir::arith::BitcastOp::create(
        builder, loc, resultType,
        aiir::arith::SelectOp::create(builder, loc, aIsNegative, upResult,
                                      downResult));
  fir::ResultOp::create(builder, loc, r1);
  builder.setInsertionPointToStart(&adjustIfOp.getElseRegion().front());
  fir::ResultOp::create(builder, loc, r);
  builder.setInsertionPointAfter(adjustIfOp);
  fir::ResultOp::create(builder, loc, adjustIfOp.getResult(0));

  // (a > b): r is inexact -- return r or ieee_next_up(r)
  builder.setInsertionPointToStart(&ifOp2.getElseRegion().front());
  roundToZero =
      aiir::arith::AndIOp::create(builder, loc, roundToZeroBit, aIsNegative);
  roundAway =
      aiir::arith::AndIOp::create(builder, loc, roundAwayBit, aIsPositive);
  aiir::Value roundUp = match(_FORTRAN_RUNTIME_IEEE_UP);
  mustAdjust =
      aiir::arith::OrIOp::create(builder, loc, roundToNearest, roundToZero);
  mustAdjust = aiir::arith::OrIOp::create(builder, loc, mustAdjust, roundAway);
  mustAdjust = aiir::arith::OrIOp::create(builder, loc, mustAdjust, roundUp);
  adjustIfOp = fir::IfOp::create(builder, loc, resultType, mustAdjust,
                                 /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&adjustIfOp.getThenRegion().front());
  if (resultType.isF80())
    r1 = fir::runtime::genNearest(builder, loc, r,
                                  builder.createBool(loc, true));
  else
    r1 = aiir::arith::BitcastOp::create(
        builder, loc, resultType,
        aiir::arith::SelectOp::create(builder, loc, aIsPositive, upResult,
                                      downResult));
  fir::ResultOp::create(builder, loc, r1);
  builder.setInsertionPointToStart(&adjustIfOp.getElseRegion().front());
  fir::ResultOp::create(builder, loc, r);
  builder.setInsertionPointAfter(adjustIfOp);
  fir::ResultOp::create(builder, loc, adjustIfOp.getResult(0));

  // Generate exceptions for (a < b) and (a > b) branches.
  builder.setInsertionPointAfter(ifOp2);
  r = ifOp2.getResult(0);
  fir::IfOp exceptIfOp1 =
      fir::IfOp::create(builder, loc, genIsFPClass(i1Ty, r, infiniteTest),
                        /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&exceptIfOp1.getThenRegion().front());
  genRaiseExcept(_FORTRAN_RUNTIME_IEEE_OVERFLOW |
                 _FORTRAN_RUNTIME_IEEE_INEXACT);
  builder.setInsertionPointToStart(&exceptIfOp1.getElseRegion().front());
  fir::IfOp exceptIfOp2 = fir::IfOp::create(
      builder, loc, genIsFPClass(i1Ty, r, subnormalTest | zeroTest),
      /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&exceptIfOp2.getThenRegion().front());
  genRaiseExcept(_FORTRAN_RUNTIME_IEEE_UNDERFLOW |
                 _FORTRAN_RUNTIME_IEEE_INEXACT);
  builder.setInsertionPointToStart(&exceptIfOp2.getElseRegion().front());
  genRaiseExcept(_FORTRAN_RUNTIME_IEEE_INEXACT);
  builder.setInsertionPointAfter(exceptIfOp1);
  fir::ResultOp::create(builder, loc, ifOp2.getResult(0));
  builder.setInsertionPointAfter(ifOp1);
  return ifOp1.getResult(0);
}

// IEEE_REM
aiir::Value IntrinsicLibrary::genIeeeRem(aiir::Type resultType,
                                         llvm::ArrayRef<aiir::Value> args) {
  // Return the remainder of X divided by Y.
  // Signal IEEE_UNDERFLOW if X is subnormal and Y is infinite.
  // Signal IEEE_INVALID if X is infinite or Y is zero and neither is a NaN.
  assert(args.size() == 2);
  aiir::Value x = args[0];
  aiir::Value y = args[1];
  if (aiir::dyn_cast<aiir::FloatType>(resultType).getWidth() < 32) {
    aiir::Type f32Ty = aiir::Float32Type::get(builder.getContext());
    x = fir::ConvertOp::create(builder, loc, f32Ty, x);
    y = fir::ConvertOp::create(builder, loc, f32Ty, y);
  } else {
    x = fir::ConvertOp::create(builder, loc, resultType, x);
    y = fir::ConvertOp::create(builder, loc, resultType, y);
  }
  // remainder calls do not signal IEEE_UNDERFLOW.
  aiir::Value underflow = aiir::arith::AndIOp::create(
      builder, loc, genIsFPClass(builder.getI1Type(), x, subnormalTest),
      genIsFPClass(builder.getI1Type(), y, infiniteTest));
  aiir::Value result = genRuntimeCall("remainder", x.getType(), {x, y});
  genRaiseExcept(_FORTRAN_RUNTIME_IEEE_UNDERFLOW, underflow);
  return fir::ConvertOp::create(builder, loc, resultType, result);
}

// IEEE_RINT
aiir::Value IntrinsicLibrary::genIeeeRint(aiir::Type resultType,
                                          llvm::ArrayRef<aiir::Value> args) {
  // Return the value of real argument A rounded to an integer value according
  // to argument ROUND if present, otherwise according to the current rounding
  // mode. If ROUND is not present, signal IEEE_INEXACT if A is not an exact
  // integral value.
  assert(args.size() == 2);
  aiir::Value a = args[0];
  aiir::func::FuncOp getRound = fir::factory::getLlvmGetRounding(builder);
  aiir::func::FuncOp setRound = fir::factory::getLlvmSetRounding(builder);
  aiir::Value mode;
  if (isStaticallyPresent(args[1])) {
    mode = fir::CallOp::create(builder, loc, getRound).getResult(0);
    genIeeeSetRoundingMode({args[1]});
  }
  if (aiir::cast<aiir::FloatType>(resultType).getWidth() == 16)
    a = fir::ConvertOp::create(builder, loc,
                               aiir::Float32Type::get(builder.getContext()), a);
  aiir::Value result = fir::ConvertOp::create(
      builder, loc, resultType, genRuntimeCall("nearbyint", a.getType(), a));
  if (isStaticallyPresent(args[1])) {
    fir::CallOp::create(builder, loc, setRound, mode);
  } else {
    aiir::Value inexact = aiir::arith::CmpFOp::create(
        builder, loc, aiir::arith::CmpFPredicate::ONE, args[0], result);
    genRaiseExcept(_FORTRAN_RUNTIME_IEEE_INEXACT, inexact);
  }
  return result;
}

// IEEE_SET_FLAG, IEEE_SET_HALTING_MODE
template <bool isFlag>
void IntrinsicLibrary::genIeeeSetFlagOrHaltingMode(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  // IEEE_SET_FLAG: Set an exception FLAG to a FLAG_VALUE.
  // IEEE_SET_HALTING: Set an exception halting mode FLAG to a HALTING value.
  assert(args.size() == 2);
  aiir::Type i1Ty = builder.getI1Type();
  aiir::Type i32Ty = builder.getIntegerType(32);
  auto [fieldRef, ignore] = getFieldRef(builder, loc, getBase(args[0]));
  aiir::Value field = fir::LoadOp::create(builder, loc, fieldRef);
  aiir::Value except = fir::runtime::genMapExcept(
      builder, loc, fir::ConvertOp::create(builder, loc, i32Ty, field));
  auto ifOp = fir::IfOp::create(
      builder, loc,
      fir::ConvertOp::create(builder, loc, i1Ty, getBase(args[1])),
      /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  (isFlag ? fir::runtime::genFeraiseexcept : fir::runtime::genFeenableexcept)(
      builder, loc, fir::ConvertOp::create(builder, loc, i32Ty, except));
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  (isFlag ? fir::runtime::genFeclearexcept : fir::runtime::genFedisableexcept)(
      builder, loc, fir::ConvertOp::create(builder, loc, i32Ty, except));
  builder.setInsertionPointAfter(ifOp);
}

// IEEE_SET_ROUNDING_MODE
void IntrinsicLibrary::genIeeeSetRoundingMode(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  // Set the current floating point rounding mode to the value of arg
  // ROUNDING_VALUE. Values are llvm.get.rounding encoding values.
  // Modes ieee_to_zero, ieee_nearest, ieee_up, and ieee_down are supported.
  // Modes ieee_away and ieee_other are not supported, and are treated as
  // ieee_nearest. Generate an error if the optional RADIX arg is not 2.
  assert(args.size() == 1 || args.size() == 2);
  if (args.size() == 2)
    checkRadix(builder, loc, fir::getBase(args[1]), "ieee_set_rounding_mode");
  auto [fieldRef, fieldTy] = getFieldRef(builder, loc, fir::getBase(args[0]));
  aiir::func::FuncOp setRound = fir::factory::getLlvmSetRounding(builder);
  aiir::Value mode = fir::LoadOp::create(builder, loc, fieldRef);
  static_assert(
      _FORTRAN_RUNTIME_IEEE_TO_ZERO >= 0 &&
      _FORTRAN_RUNTIME_IEEE_TO_ZERO <= 3 &&
      _FORTRAN_RUNTIME_IEEE_NEAREST >= 0 &&
      _FORTRAN_RUNTIME_IEEE_NEAREST <= 3 && _FORTRAN_RUNTIME_IEEE_UP >= 0 &&
      _FORTRAN_RUNTIME_IEEE_UP <= 3 && _FORTRAN_RUNTIME_IEEE_DOWN >= 0 &&
      _FORTRAN_RUNTIME_IEEE_DOWN <= 3 && "unexpected rounding mode mapping");
  aiir::Value mask = aiir::arith::ShLIOp::create(
      builder, loc, builder.createAllOnesInteger(loc, fieldTy),
      builder.createIntegerConstant(loc, fieldTy, 2));
  aiir::Value modeIsSupported = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::eq,
      aiir::arith::AndIOp::create(builder, loc, mode, mask),
      builder.createIntegerConstant(loc, fieldTy, 0));
  aiir::Value nearest = builder.createIntegerConstant(
      loc, fieldTy, _FORTRAN_RUNTIME_IEEE_NEAREST);
  mode = aiir::arith::SelectOp::create(builder, loc, modeIsSupported, mode,
                                       nearest);
  mode = fir::ConvertOp::create(builder, loc,
                                setRound.getFunctionType().getInput(0), mode);
  fir::CallOp::create(builder, loc, setRound, mode);
}

// IEEE_SET_UNDERFLOW_MODE
void IntrinsicLibrary::genIeeeSetUnderflowMode(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  aiir::Value gradual = fir::ConvertOp::create(
      builder, loc, builder.getI1Type(), getBase(args[0]));
  fir::runtime::genSetUnderflowMode(builder, loc, {gradual});
}

// IEEE_SIGNALING_EQ, IEEE_SIGNALING_GE, IEEE_SIGNALING_GT,
// IEEE_SIGNALING_LE, IEEE_SIGNALING_LT, IEEE_SIGNALING_NE
template <aiir::arith::CmpFPredicate pred>
aiir::Value
IntrinsicLibrary::genIeeeSignalingCompare(aiir::Type resultType,
                                          llvm::ArrayRef<aiir::Value> args) {
  // Compare X and Y with special case treatment of NaN operands.
  assert(args.size() == 2);
  aiir::Value hasNaNOp = genIeeeUnordered(aiir::Type{}, args);
  aiir::Value res =
      aiir::arith::CmpFOp::create(builder, loc, pred, args[0], args[1]);
  genRaiseExcept(_FORTRAN_RUNTIME_IEEE_INVALID, hasNaNOp);
  return fir::ConvertOp::create(builder, loc, resultType, res);
}

// IEEE_SIGNBIT
aiir::Value IntrinsicLibrary::genIeeeSignbit(aiir::Type resultType,
                                             llvm::ArrayRef<aiir::Value> args) {
  // Check if the sign bit of arg X is set.
  assert(args.size() == 1);
  aiir::Value realVal = args[0];
  aiir::FloatType realType = aiir::dyn_cast<aiir::FloatType>(realVal.getType());
  int bitWidth = realType.getWidth();
  if (realType == aiir::BFloat16Type::get(builder.getContext())) {
    // Workaround: can't bitcast or convert real(3) to integer(2) or real(2).
    realVal = builder.createConvert(
        loc, aiir::Float32Type::get(builder.getContext()), realVal);
    bitWidth = 32;
  }
  aiir::Type intType = builder.getIntegerType(bitWidth);
  aiir::Value intVal =
      aiir::arith::BitcastOp::create(builder, loc, intType, realVal);
  aiir::Value shift = builder.createIntegerConstant(loc, intType, bitWidth - 1);
  aiir::Value sign = aiir::arith::ShRUIOp::create(builder, loc, intVal, shift);
  return builder.createConvert(loc, resultType, sign);
}

// IEEE_SUPPORT_FLAG
fir::ExtendedValue
IntrinsicLibrary::genIeeeSupportFlag(aiir::Type resultType,
                                     llvm::ArrayRef<fir::ExtendedValue> args) {
  // Check if a floating point exception flag is supported.
  assert(args.size() == 1 || args.size() == 2);
  aiir::Type i1Ty = builder.getI1Type();
  aiir::Type i32Ty = builder.getIntegerType(32);
  auto [fieldRef, fieldTy] = getFieldRef(builder, loc, getBase(args[0]));
  aiir::Value flag = fir::LoadOp::create(builder, loc, fieldRef);
  aiir::Value standardFlagMask = builder.createIntegerConstant(
      loc, fieldTy,
      _FORTRAN_RUNTIME_IEEE_INVALID | _FORTRAN_RUNTIME_IEEE_DIVIDE_BY_ZERO |
          _FORTRAN_RUNTIME_IEEE_OVERFLOW | _FORTRAN_RUNTIME_IEEE_UNDERFLOW |
          _FORTRAN_RUNTIME_IEEE_INEXACT);
  aiir::Value isStandardFlag = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::ne,
      aiir::arith::AndIOp::create(builder, loc, flag, standardFlagMask),
      builder.createIntegerConstant(loc, fieldTy, 0));
  fir::IfOp ifOp = fir::IfOp::create(builder, loc, i1Ty, isStandardFlag,
                                     /*withElseRegion=*/true);
  // Standard flags are supported.
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  fir::ResultOp::create(builder, loc, builder.createBool(loc, true));

  // TargetCharacteristics information for the nonstandard ieee_denorm flag
  // is not available here. So use a runtime check restricted to possibly
  // supported kinds.
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  bool mayBeSupported = false;
  if (aiir::Value arg1 = getBase(args[1])) {
    aiir::Type arg1Ty = arg1.getType();
    if (auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(arg1.getType()))
      arg1Ty = eleTy;
    if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(arg1Ty))
      arg1Ty = seqTy.getEleTy();
    switch (aiir::dyn_cast<aiir::FloatType>(arg1Ty).getWidth()) {
    case 16:
      mayBeSupported = arg1Ty.isBF16(); // kind=3
      break;
    case 32: // kind=4
    case 64: // kind=8
      mayBeSupported = true;
      break;
    }
  }
  if (mayBeSupported) {
    aiir::Value isDenorm = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::eq, flag,
        builder.createIntegerConstant(loc, fieldTy,
                                      _FORTRAN_RUNTIME_IEEE_DENORM));
    aiir::Value result = aiir::arith::AndIOp::create(
        builder, loc, isDenorm,
        fir::runtime::genSupportHalting(
            builder, loc, fir::ConvertOp::create(builder, loc, i32Ty, flag)));
    fir::ResultOp::create(builder, loc, result);
  } else {
    fir::ResultOp::create(builder, loc, builder.createBool(loc, false));
  }
  builder.setInsertionPointAfter(ifOp);
  return builder.createConvert(loc, resultType, ifOp.getResult(0));
}

// IEEE_SUPPORT_HALTING
fir::ExtendedValue IntrinsicLibrary::genIeeeSupportHalting(
    aiir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args) {
  // Check if halting is supported for a floating point exception flag.
  // Standard flags are all supported. The nonstandard DENORM extension is
  // not supported, at least for now.
  assert(args.size() == 1);
  aiir::Type i32Ty = builder.getIntegerType(32);
  auto [fieldRef, ignore] = getFieldRef(builder, loc, getBase(args[0]));
  aiir::Value field = fir::LoadOp::create(builder, loc, fieldRef);
  return builder.createConvert(
      loc, resultType,
      fir::runtime::genSupportHalting(
          builder, loc, fir::ConvertOp::create(builder, loc, i32Ty, field)));
}

// IEEE_SUPPORT_ROUNDING
fir::ExtendedValue IntrinsicLibrary::genIeeeSupportRounding(
    aiir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args) {
  // Check if floating point rounding mode ROUND_VALUE is supported.
  // Rounding is supported either for all type kinds or none.
  // An optional X kind argument is therefore ignored.
  // Values are chosen to match the llvm.get.rounding encoding:
  //  0 - toward zero [supported]
  //  1 - to nearest, ties to even [supported] - default
  //  2 - toward positive infinity [supported]
  //  3 - toward negative infinity [supported]
  //  4 - to nearest, ties away from zero [not supported]
  assert(args.size() == 1 || args.size() == 2);
  auto [fieldRef, fieldTy] = getFieldRef(builder, loc, getBase(args[0]));
  aiir::Value mode = fir::LoadOp::create(builder, loc, fieldRef);
  aiir::Value lbOk = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::sge, mode,
      builder.createIntegerConstant(loc, fieldTy,
                                    _FORTRAN_RUNTIME_IEEE_TO_ZERO));
  aiir::Value ubOk = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::sle, mode,
      builder.createIntegerConstant(loc, fieldTy, _FORTRAN_RUNTIME_IEEE_DOWN));
  return builder.createConvert(
      loc, resultType, aiir::arith::AndIOp::create(builder, loc, lbOk, ubOk));
}

// IEEE_SUPPORT_STANDARD
fir::ExtendedValue IntrinsicLibrary::genIeeeSupportStandard(
    aiir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args) {
  // Check if IEEE standard support is available, which reduces to checking
  // if halting control is supported, as that is the only support component
  // that may not be available.
  assert(args.size() <= 1);
  aiir::Value overflow = builder.createIntegerConstant(
      loc, builder.getIntegerType(32), _FORTRAN_RUNTIME_IEEE_OVERFLOW);
  return builder.createConvert(
      loc, resultType, fir::runtime::genSupportHalting(builder, loc, overflow));
}

// IEEE_UNORDERED
aiir::Value
IntrinsicLibrary::genIeeeUnordered(aiir::Type resultType,
                                   llvm::ArrayRef<aiir::Value> args) {
  // Check if REAL args X or Y or both are (signaling or quiet) NaNs.
  // If there is no result type return an i1 result.
  assert(args.size() == 2);
  if (args[0].getType() == args[1].getType()) {
    aiir::Value res = aiir::arith::CmpFOp::create(
        builder, loc, aiir::arith::CmpFPredicate::UNO, args[0], args[1]);
    return resultType ? builder.createConvert(loc, resultType, res) : res;
  }
  assert(resultType && "expecting a (mixed arg type) unordered result type");
  aiir::Type i1Ty = builder.getI1Type();
  aiir::Value xIsNan = genIsFPClass(i1Ty, args[0], nanTest);
  aiir::Value yIsNan = genIsFPClass(i1Ty, args[1], nanTest);
  aiir::Value res = aiir::arith::OrIOp::create(builder, loc, xIsNan, yIsNan);
  return builder.createConvert(loc, resultType, res);
}

// IEEE_VALUE
aiir::Value IntrinsicLibrary::genIeeeValue(aiir::Type resultType,
                                           llvm::ArrayRef<aiir::Value> args) {
  // Return a KIND(X) REAL number of IEEE_CLASS_TYPE CLASS.
  // A user call has two arguments:
  //  - arg[0] is X (ignored, since the resultType is provided)
  //  - arg[1] is CLASS, an IEEE_CLASS_TYPE CLASS argument containing an index
  // A compiler generated call has one argument:
  //  - arg[0] is an index constant
  assert(args.size() == 1 || args.size() == 2);
  aiir::FloatType realType = aiir::dyn_cast<aiir::FloatType>(resultType);
  int bitWidth = realType.getWidth();
  aiir::Type intType = builder.getIntegerType(bitWidth);
  aiir::Type valueTy = bitWidth <= 64 ? intType : builder.getIntegerType(64);
  constexpr int tableSize = _FORTRAN_RUNTIME_IEEE_OTHER_VALUE + 1;
  aiir::Type tableTy = fir::SequenceType::get(tableSize, valueTy);
  std::string tableName = RTNAME_STRING(IeeeValueTable_) +
                          std::to_string(realType.isBF16() ? 3 : bitWidth >> 3);
  if (!builder.getNamedGlobal(tableName)) {
    llvm::SmallVector<aiir::Attribute, tableSize> values;
    auto insert = [&](std::int64_t v) {
      values.push_back(builder.getIntegerAttr(valueTy, v));
    };
    insert(0); // placeholder
    switch (bitWidth) {
    case 16:
      if (realType.isF16()) {
        // kind=2: 1 sign bit, 5 exponent bits, 10 significand bits
        /* IEEE_SIGNALING_NAN      */ insert(0x7d00);
        /* IEEE_QUIET_NAN          */ insert(0x7e00);
        /* IEEE_NEGATIVE_INF       */ insert(0xfc00);
        /* IEEE_NEGATIVE_NORMAL    */ insert(0xbc00);
        /* IEEE_NEGATIVE_SUBNORMAL */ insert(0x8200);
        /* IEEE_NEGATIVE_ZERO      */ insert(0x8000);
        /* IEEE_POSITIVE_ZERO      */ insert(0x0000);
        /* IEEE_POSITIVE_SUBNORMAL */ insert(0x0200);
        /* IEEE_POSITIVE_NORMAL    */ insert(0x3c00); // 1.0
        /* IEEE_POSITIVE_INF       */ insert(0x7c00);
        break;
      }
      assert(realType.isBF16() && "unknown 16-bit real type");
      // kind=3: 1 sign bit, 8 exponent bits, 7 significand bits
      /* IEEE_SIGNALING_NAN      */ insert(0x7fa0);
      /* IEEE_QUIET_NAN          */ insert(0x7fc0);
      /* IEEE_NEGATIVE_INF       */ insert(0xff80);
      /* IEEE_NEGATIVE_NORMAL    */ insert(0xbf80);
      /* IEEE_NEGATIVE_SUBNORMAL */ insert(0x8040);
      /* IEEE_NEGATIVE_ZERO      */ insert(0x8000);
      /* IEEE_POSITIVE_ZERO      */ insert(0x0000);
      /* IEEE_POSITIVE_SUBNORMAL */ insert(0x0040);
      /* IEEE_POSITIVE_NORMAL    */ insert(0x3f80); // 1.0
      /* IEEE_POSITIVE_INF       */ insert(0x7f80);
      break;
    case 32:
      // kind=4: 1 sign bit, 8 exponent bits, 23 significand bits
      /* IEEE_SIGNALING_NAN      */ insert(0x7fa00000);
      /* IEEE_QUIET_NAN          */ insert(0x7fc00000);
      /* IEEE_NEGATIVE_INF       */ insert(0xff800000);
      /* IEEE_NEGATIVE_NORMAL    */ insert(0xbf800000);
      /* IEEE_NEGATIVE_SUBNORMAL */ insert(0x80400000);
      /* IEEE_NEGATIVE_ZERO      */ insert(0x80000000);
      /* IEEE_POSITIVE_ZERO      */ insert(0x00000000);
      /* IEEE_POSITIVE_SUBNORMAL */ insert(0x00400000);
      /* IEEE_POSITIVE_NORMAL    */ insert(0x3f800000); // 1.0
      /* IEEE_POSITIVE_INF       */ insert(0x7f800000);
      break;
    case 64:
      // kind=8: 1 sign bit, 11 exponent bits, 52 significand bits
      /* IEEE_SIGNALING_NAN      */ insert(0x7ff4000000000000);
      /* IEEE_QUIET_NAN          */ insert(0x7ff8000000000000);
      /* IEEE_NEGATIVE_INF       */ insert(0xfff0000000000000);
      /* IEEE_NEGATIVE_NORMAL    */ insert(0xbff0000000000000);
      /* IEEE_NEGATIVE_SUBNORMAL */ insert(0x8008000000000000);
      /* IEEE_NEGATIVE_ZERO      */ insert(0x8000000000000000);
      /* IEEE_POSITIVE_ZERO      */ insert(0x0000000000000000);
      /* IEEE_POSITIVE_SUBNORMAL */ insert(0x0008000000000000);
      /* IEEE_POSITIVE_NORMAL    */ insert(0x3ff0000000000000); // 1.0
      /* IEEE_POSITIVE_INF       */ insert(0x7ff0000000000000);
      break;
    case 80:
      // kind=10: 1 sign bit, 15 exponent bits, 1+63 significand bits
      // 64 high order bits; 16 low order bits are 0.
      /* IEEE_SIGNALING_NAN      */ insert(0x7fffa00000000000);
      /* IEEE_QUIET_NAN          */ insert(0x7fffc00000000000);
      /* IEEE_NEGATIVE_INF       */ insert(0xffff800000000000);
      /* IEEE_NEGATIVE_NORMAL    */ insert(0xbfff800000000000);
      /* IEEE_NEGATIVE_SUBNORMAL */ insert(0x8000400000000000);
      /* IEEE_NEGATIVE_ZERO      */ insert(0x8000000000000000);
      /* IEEE_POSITIVE_ZERO      */ insert(0x0000000000000000);
      /* IEEE_POSITIVE_SUBNORMAL */ insert(0x0000400000000000);
      /* IEEE_POSITIVE_NORMAL    */ insert(0x3fff800000000000); // 1.0
      /* IEEE_POSITIVE_INF       */ insert(0x7fff800000000000);
      break;
    case 128:
      // kind=16: 1 sign bit, 15 exponent bits, 112 significand bits
      // 64 high order bits; 64 low order bits are 0.
      /* IEEE_SIGNALING_NAN      */ insert(0x7fff400000000000);
      /* IEEE_QUIET_NAN          */ insert(0x7fff800000000000);
      /* IEEE_NEGATIVE_INF       */ insert(0xffff000000000000);
      /* IEEE_NEGATIVE_NORMAL    */ insert(0xbfff000000000000);
      /* IEEE_NEGATIVE_SUBNORMAL */ insert(0x8000200000000000);
      /* IEEE_NEGATIVE_ZERO      */ insert(0x8000000000000000);
      /* IEEE_POSITIVE_ZERO      */ insert(0x0000000000000000);
      /* IEEE_POSITIVE_SUBNORMAL */ insert(0x0000200000000000);
      /* IEEE_POSITIVE_NORMAL    */ insert(0x3fff000000000000); // 1.0
      /* IEEE_POSITIVE_INF       */ insert(0x7fff000000000000);
      break;
    default:
      llvm_unreachable("unknown real type");
    }
    insert(0); // IEEE_OTHER_VALUE
    assert(values.size() == tableSize && "ieee value mismatch");
    builder.createGlobalConstant(
        loc, tableTy, tableName, builder.createLinkOnceLinkage(),
        aiir::DenseElementsAttr::get(
            aiir::RankedTensorType::get(tableSize, valueTy), values));
  }

  aiir::Value which;
  if (args.size() == 2) { // user call
    auto [index, ignore] = getFieldRef(builder, loc, args[1]);
    which = fir::LoadOp::create(builder, loc, index);
  } else { // compiler generated call
    which = args[0];
  }
  aiir::Value bits = fir::LoadOp::create(
      builder, loc,
      fir::CoordinateOp::create(
          builder, loc, builder.getRefType(valueTy),
          fir::AddrOfOp::create(builder, loc, builder.getRefType(tableTy),
                                builder.getSymbolRefAttr(tableName)),
          which));
  if (bitWidth > 64)
    bits = aiir::arith::ShLIOp::create(
        builder, loc, builder.createConvert(loc, intType, bits),
        builder.createIntegerConstant(loc, intType, bitWidth - 64));
  return aiir::arith::BitcastOp::create(builder, loc, realType, bits);
}

// IEOR
aiir::Value IntrinsicLibrary::genIeor(aiir::Type resultType,
                                      llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  return builder.createUnsigned<aiir::arith::XOrIOp>(loc, resultType, args[0],
                                                     args[1]);
}

// INDEX
fir::ExtendedValue
IntrinsicLibrary::genIndex(aiir::Type resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() >= 2 && args.size() <= 4);

  aiir::Value stringBase = fir::getBase(args[0]);
  fir::KindTy kind =
      fir::factory::CharacterExprHelper{builder, loc}.getCharacterKind(
          stringBase.getType());
  aiir::Value stringLen = fir::getLen(args[0]);
  aiir::Value substringBase = fir::getBase(args[1]);
  aiir::Value substringLen = fir::getLen(args[1]);
  aiir::Value back =
      isStaticallyAbsent(args, 2)
          ? builder.createIntegerConstant(loc, builder.getI1Type(), 0)
          : fir::getBase(args[2]);
  if (isStaticallyAbsent(args, 3))
    return builder.createConvert(
        loc, resultType,
        fir::runtime::genIndex(builder, loc, kind, stringBase, stringLen,
                               substringBase, substringLen, back));

  // Call the descriptor-based Index implementation
  aiir::Value string = builder.createBox(loc, args[0]);
  aiir::Value substring = builder.createBox(loc, args[1]);
  auto makeRefThenEmbox = [&](aiir::Value b) {
    fir::LogicalType logTy = fir::LogicalType::get(
        builder.getContext(), builder.getKindMap().defaultLogicalKind());
    aiir::Value temp = builder.createTemporary(loc, logTy);
    aiir::Value castb = builder.createConvert(loc, logTy, b);
    fir::StoreOp::create(builder, loc, castb, temp);
    return builder.createBox(loc, temp);
  };
  aiir::Value backOpt =
      isStaticallyAbsent(args, 2)
          ? fir::AbsentOp::create(builder, loc,
                                  fir::BoxType::get(builder.getI1Type()))
          : makeRefThenEmbox(fir::getBase(args[2]));
  aiir::Value kindVal = isStaticallyAbsent(args, 3)
                            ? builder.createIntegerConstant(
                                  loc, builder.getIndexType(),
                                  builder.getKindMap().defaultIntegerKind())
                            : fir::getBase(args[3]);
  // Create mutable fir.box to be passed to the runtime for the result.
  fir::MutableBoxValue mutBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  aiir::Value resBox = fir::factory::getMutableIRBox(builder, loc, mutBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genIndexDescriptor(builder, loc, resBox, string, substring,
                                   backOpt, kindVal);
  // Read back the result from the mutable box.
  return readAndAddCleanUp(mutBox, resultType, "INDEX");
}

// IOR
aiir::Value IntrinsicLibrary::genIor(aiir::Type resultType,
                                     llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  return builder.createUnsigned<aiir::arith::OrIOp>(loc, resultType, args[0],
                                                    args[1]);
}

// IPARITY
fir::ExtendedValue
IntrinsicLibrary::genIparity(aiir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  return genReduction(fir::runtime::genIParity, fir::runtime::genIParityDim,
                      "IPARITY", resultType, args);
}

// IRAND
fir::ExtendedValue
IntrinsicLibrary::genIrand(aiir::Type resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  aiir::Value i =
      isStaticallyPresent(args[0])
          ? fir::getBase(args[0])
          : fir::AbsentOp::create(builder, loc,
                                  builder.getRefType(builder.getI32Type()))
                .getResult();
  return fir::runtime::genIrand(builder, loc, i);
}

// IS_CONTIGUOUS
fir::ExtendedValue
IntrinsicLibrary::genIsContiguous(aiir::Type resultType,
                                  llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  return builder.createConvert(
      loc, resultType,
      fir::runtime::genIsContiguous(builder, loc, fir::getBase(args[0])));
}

// IS_IOSTAT_END, IS_IOSTAT_EOR
template <Fortran::runtime::io::Iostat value>
aiir::Value
IntrinsicLibrary::genIsIostatValue(aiir::Type resultType,
                                   llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);
  return aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::eq, args[0],
      builder.createIntegerConstant(loc, args[0].getType(), value));
}

// ISHFT
aiir::Value IntrinsicLibrary::genIshft(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  // A conformant ISHFT(I,SHIFT) call satisfies:
  //     abs(SHIFT) <= BIT_SIZE(I)
  // Return:  abs(SHIFT) >= BIT_SIZE(I)
  //              ? 0
  //              : SHIFT < 0
  //                    ? I >> abs(SHIFT)
  //                    : I << abs(SHIFT)
  assert(args.size() == 2);
  int intWidth = resultType.getIntOrFloatBitWidth();
  aiir::Type signlessType =
      aiir::IntegerType::get(builder.getContext(), intWidth,
                             aiir::IntegerType::SignednessSemantics::Signless);
  aiir::Value bitSize =
      builder.createIntegerConstant(loc, signlessType, intWidth);
  aiir::Value zero = builder.createIntegerConstant(loc, signlessType, 0);
  aiir::Value shift = builder.createConvert(loc, signlessType, args[1]);
  aiir::Value absShift = genAbs(signlessType, {shift});
  aiir::Value word = args[0];
  if (word.getType().isUnsignedInteger())
    word = builder.createConvert(loc, signlessType, word);
  auto left = aiir::arith::ShLIOp::create(builder, loc, word, absShift);
  auto right = aiir::arith::ShRUIOp::create(builder, loc, word, absShift);
  auto shiftIsLarge = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::sge, absShift, bitSize);
  auto shiftIsNegative = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::slt, shift, zero);
  auto sel =
      aiir::arith::SelectOp::create(builder, loc, shiftIsNegative, right, left);
  aiir::Value result =
      aiir::arith::SelectOp::create(builder, loc, shiftIsLarge, zero, sel);
  if (resultType.isUnsignedInteger())
    return builder.createConvert(loc, resultType, result);
  return result;
}

// ISHFTC
aiir::Value IntrinsicLibrary::genIshftc(aiir::Type resultType,
                                        llvm::ArrayRef<aiir::Value> args) {
  // A conformant ISHFTC(I,SHIFT,SIZE) call satisfies:
  //     SIZE > 0
  //     SIZE <= BIT_SIZE(I)
  //     abs(SHIFT) <= SIZE
  // if SHIFT > 0
  //     leftSize = abs(SHIFT)
  //     rightSize = SIZE - abs(SHIFT)
  // else [if SHIFT < 0]
  //     leftSize = SIZE - abs(SHIFT)
  //     rightSize = abs(SHIFT)
  // unchanged = SIZE == BIT_SIZE(I) ? 0 : (I >> SIZE) << SIZE
  // leftMaskShift = BIT_SIZE(I) - leftSize
  // rightMaskShift = BIT_SIZE(I) - rightSize
  // left = (I >> rightSize) & (-1 >> leftMaskShift)
  // right = (I & (-1 >> rightMaskShift)) << leftSize
  // Return:  SHIFT == 0 || SIZE == abs(SHIFT) ? I : (unchanged | left | right)
  assert(args.size() == 3);
  int intWidth = resultType.getIntOrFloatBitWidth();
  aiir::Type signlessType =
      aiir::IntegerType::get(builder.getContext(), intWidth,
                             aiir::IntegerType::SignednessSemantics::Signless);
  aiir::Value bitSize =
      builder.createIntegerConstant(loc, signlessType, intWidth);
  aiir::Value word = args[0];
  if (word.getType().isUnsignedInteger())
    word = builder.createConvert(loc, signlessType, word);
  aiir::Value shift = builder.createConvert(loc, signlessType, args[1]);
  aiir::Value size =
      args[2] ? builder.createConvert(loc, signlessType, args[2]) : bitSize;
  aiir::Value zero = builder.createIntegerConstant(loc, signlessType, 0);
  aiir::Value ones = builder.createAllOnesInteger(loc, signlessType);
  aiir::Value absShift = genAbs(signlessType, {shift});
  auto elseSize = aiir::arith::SubIOp::create(builder, loc, size, absShift);
  auto shiftIsZero = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::eq, shift, zero);
  auto shiftEqualsSize = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::eq, absShift, size);
  auto shiftIsNop =
      aiir::arith::OrIOp::create(builder, loc, shiftIsZero, shiftEqualsSize);
  auto shiftIsPositive = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::sgt, shift, zero);
  auto leftSize = aiir::arith::SelectOp::create(builder, loc, shiftIsPositive,
                                                absShift, elseSize);
  auto rightSize = aiir::arith::SelectOp::create(builder, loc, shiftIsPositive,
                                                 elseSize, absShift);
  auto hasUnchanged = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::ne, size, bitSize);
  auto unchangedTmp1 = aiir::arith::ShRUIOp::create(builder, loc, word, size);
  auto unchangedTmp2 =
      aiir::arith::ShLIOp::create(builder, loc, unchangedTmp1, size);
  auto unchanged = aiir::arith::SelectOp::create(builder, loc, hasUnchanged,
                                                 unchangedTmp2, zero);
  auto leftMaskShift =
      aiir::arith::SubIOp::create(builder, loc, bitSize, leftSize);
  auto leftMask =
      aiir::arith::ShRUIOp::create(builder, loc, ones, leftMaskShift);
  auto leftTmp = aiir::arith::ShRUIOp::create(builder, loc, word, rightSize);
  auto left = aiir::arith::AndIOp::create(builder, loc, leftTmp, leftMask);
  auto rightMaskShift =
      aiir::arith::SubIOp::create(builder, loc, bitSize, rightSize);
  auto rightMask =
      aiir::arith::ShRUIOp::create(builder, loc, ones, rightMaskShift);
  auto rightTmp = aiir::arith::AndIOp::create(builder, loc, word, rightMask);
  auto right = aiir::arith::ShLIOp::create(builder, loc, rightTmp, leftSize);
  auto resTmp = aiir::arith::OrIOp::create(builder, loc, unchanged, left);
  auto res = aiir::arith::OrIOp::create(builder, loc, resTmp, right);
  aiir::Value result =
      aiir::arith::SelectOp::create(builder, loc, shiftIsNop, word, res);
  if (resultType.isUnsignedInteger())
    return builder.createConvert(loc, resultType, result);
  return result;
}

// LEADZ
aiir::Value IntrinsicLibrary::genLeadz(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);

  aiir::Value result =
      aiir::math::CountLeadingZerosOp::create(builder, loc, args);

  return builder.createConvert(loc, resultType, result);
}

// LEN
// Note that this is only used for an unrestricted intrinsic LEN call.
// Other uses of LEN are rewritten as descriptor inquiries by the front-end.
fir::ExtendedValue
IntrinsicLibrary::genLen(aiir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {
  // Optional KIND argument reflected in result type and otherwise ignored.
  assert(args.size() == 1 || args.size() == 2);
  aiir::Value len = fir::factory::readCharLen(builder, loc, args[0]);
  return builder.createConvert(loc, resultType, len);
}

// LEN_TRIM
fir::ExtendedValue
IntrinsicLibrary::genLenTrim(aiir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  // Optional KIND argument reflected in result type and otherwise ignored.
  assert(args.size() == 1 || args.size() == 2);
  const fir::CharBoxValue *charBox = args[0].getCharBox();
  if (!charBox)
    TODO(loc, "intrinsic: len_trim for character array");
  auto len =
      fir::factory::CharacterExprHelper(builder, loc).createLenTrim(*charBox);
  return builder.createConvert(loc, resultType, len);
}

// LGE, LGT, LLE, LLT
template <aiir::arith::CmpIPredicate pred>
fir::ExtendedValue
IntrinsicLibrary::genCharacterCompare(aiir::Type resultType,
                                      llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  return fir::runtime::genCharCompare(
      builder, loc, pred, fir::getBase(args[0]), fir::getLen(args[0]),
      fir::getBase(args[1]), fir::getLen(args[1]));
}

// LOC
fir::ExtendedValue
IntrinsicLibrary::genLoc(aiir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  aiir::Value box = fir::getBase(args[0]);
  assert(fir::isa_box_type(box.getType()) &&
         "argument must have been lowered to box type");
  bool isFunc = aiir::isa<fir::BoxProcType>(box.getType());
  if (!isOptional(box)) {
    aiir::Value argAddr = getAddrFromBox(builder, loc, args[0], isFunc);
    return builder.createConvert(loc, resultType, argAddr);
  }
  // Optional assumed shape case.  Although this is not specified in this GNU
  // intrinsic extension, LOC accepts absent optional and returns zero in that
  // case.
  // Note that the other OPTIONAL cases do not fall here since `box` was
  // created when preparing the argument cases, but the box can be safely be
  // used for all those cases and the address will be null if absent.
  aiir::Value isPresent =
      fir::IsPresentOp::create(builder, loc, builder.getI1Type(), box);
  return builder
      .genIfOp(loc, {resultType}, isPresent,
               /*withElseRegion=*/true)
      .genThen([&]() {
        aiir::Value argAddr = getAddrFromBox(builder, loc, args[0], isFunc);
        aiir::Value cast = builder.createConvert(loc, resultType, argAddr);
        fir::ResultOp::create(builder, loc, cast);
      })
      .genElse([&]() {
        aiir::Value zero = builder.createIntegerConstant(loc, resultType, 0);
        fir::ResultOp::create(builder, loc, zero);
      })
      .getResults()[0];
}

aiir::Value IntrinsicLibrary::genMalloc(aiir::Type resultType,
                                        llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);
  return builder.createConvert(loc, resultType,
                               fir::runtime::genMalloc(builder, loc, args[0]));
}

// MASKL, MASKR, UMASKL, UMASKR
template <typename Shift>
aiir::Value IntrinsicLibrary::genMask(aiir::Type resultType,
                                      llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);

  int bits = resultType.getIntOrFloatBitWidth();
  aiir::Type signlessType =
      aiir::IntegerType::get(builder.getContext(), bits,
                             aiir::IntegerType::SignednessSemantics::Signless);
  aiir::Value zero = builder.createIntegerConstant(loc, signlessType, 0);
  aiir::Value ones = builder.createAllOnesInteger(loc, signlessType);
  aiir::Value bitSize = builder.createIntegerConstant(loc, signlessType, bits);
  aiir::Value bitsToSet = builder.createConvert(loc, signlessType, args[0]);

  // The standard does not specify what to return if the number of bits to be
  // set, I < 0 or I >= BIT_SIZE(KIND). The shift instruction used below will
  // produce a poison value which may return a possibly platform-specific and/or
  // non-deterministic result. Other compilers don't produce a consistent result
  // in this case either, so we choose the most efficient implementation.
  aiir::Value shift =
      aiir::arith::SubIOp::create(builder, loc, bitSize, bitsToSet);
  aiir::Value shifted = Shift::create(builder, loc, ones, shift);
  aiir::Value isZero = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::eq, bitsToSet, zero);
  aiir::Value result =
      aiir::arith::SelectOp::create(builder, loc, isZero, zero, shifted);
  if (resultType.isUnsignedInteger())
    return builder.createConvert(loc, resultType, result);
  return result;
}

// MATMUL
fir::ExtendedValue
IntrinsicLibrary::genMatmul(aiir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);

  // Handle required matmul arguments
  fir::BoxValue matrixTmpA = builder.createBox(loc, args[0]);
  aiir::Value matrixA = fir::getBase(matrixTmpA);
  fir::BoxValue matrixTmpB = builder.createBox(loc, args[1]);
  aiir::Value matrixB = fir::getBase(matrixTmpB);
  unsigned resultRank =
      (matrixTmpA.rank() == 1 || matrixTmpB.rank() == 1) ? 1 : 2;

  // Create mutable fir.box to be passed to the runtime for the result.
  aiir::Type resultArrayType = builder.getVarLenSeqTy(resultType, resultRank);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genMatmul(builder, loc, resultIrBox, matrixA, matrixB);
  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  return readAndAddCleanUp(resultMutableBox, resultType, "MATMUL");
}

// MATMUL_TRANSPOSE
fir::ExtendedValue
IntrinsicLibrary::genMatmulTranspose(aiir::Type resultType,
                                     llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);

  // Handle required matmul_transpose arguments
  fir::BoxValue matrixTmpA = builder.createBox(loc, args[0]);
  aiir::Value matrixA = fir::getBase(matrixTmpA);
  fir::BoxValue matrixTmpB = builder.createBox(loc, args[1]);
  aiir::Value matrixB = fir::getBase(matrixTmpB);
  unsigned resultRank =
      (matrixTmpA.rank() == 1 || matrixTmpB.rank() == 1) ? 1 : 2;

  // Create mutable fir.box to be passed to the runtime for the result.
  aiir::Type resultArrayType = builder.getVarLenSeqTy(resultType, resultRank);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genMatmulTranspose(builder, loc, resultIrBox, matrixA, matrixB);
  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  return readAndAddCleanUp(resultMutableBox, resultType, "MATMUL_TRANSPOSE");
}

// MERGE
fir::ExtendedValue
IntrinsicLibrary::genMerge(aiir::Type,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  aiir::Value tsource = fir::getBase(args[0]);
  aiir::Value fsource = fir::getBase(args[1]);
  aiir::Value rawMask = fir::getBase(args[2]);
  aiir::Type type0 = fir::unwrapRefType(tsource.getType());
  bool isCharRslt = fir::isa_char(type0); // result is same as first argument
  aiir::Value mask = builder.createConvert(loc, builder.getI1Type(), rawMask);

  // The result is polymorphic if and only if both TSOURCE and FSOURCE are
  // polymorphic. TSOURCE and FSOURCE are required to have the same type
  // (for both declared and dynamic types) so a simple convert op can be
  // used.
  aiir::Value tsourceCast = tsource;
  aiir::Value fsourceCast = fsource;
  auto convertToStaticType = [&](aiir::Value polymorphic,
                                 aiir::Value other) -> aiir::Value {
    aiir::Type otherType = other.getType();
    if (aiir::isa<fir::BaseBoxType>(otherType))
      return fir::ReboxOp::create(builder, loc, otherType, polymorphic,
                                  /*shape*/ aiir::Value{},
                                  /*slice=*/aiir::Value{});
    return fir::BoxAddrOp::create(builder, loc, otherType, polymorphic);
  };
  if (fir::isPolymorphicType(tsource.getType()) &&
      !fir::isPolymorphicType(fsource.getType())) {
    tsourceCast = convertToStaticType(tsource, fsource);
  } else if (!fir::isPolymorphicType(tsource.getType()) &&
             fir::isPolymorphicType(fsource.getType())) {
    fsourceCast = convertToStaticType(fsource, tsource);
  } else {
    // FSOURCE and TSOURCE are not polymorphic.
    // FSOURCE has the same type as TSOURCE, but they may not have the same AIIR
    // types (one can have dynamic length while the other has constant lengths,
    // or one may be a fir.logical<> while the other is an i1). Insert a cast to
    // fulfill aiir::SelectOp constraint that the AIIR types must be the same.
    fsourceCast = builder.createConvert(loc, tsource.getType(), fsource);
  }
  auto rslt = aiir::arith::SelectOp::create(builder, loc, mask, tsourceCast,
                                            fsourceCast);
  if (isCharRslt) {
    // Need a CharBoxValue for character results
    const fir::CharBoxValue *charBox = args[0].getCharBox();
    fir::CharBoxValue charRslt(rslt, charBox->getLen());
    return charRslt;
  }
  return rslt;
}

// MERGE_BITS
aiir::Value IntrinsicLibrary::genMergeBits(aiir::Type resultType,
                                           llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 3);

  aiir::Type signlessType = aiir::IntegerType::get(
      builder.getContext(), resultType.getIntOrFloatBitWidth(),
      aiir::IntegerType::SignednessSemantics::Signless);
  // MERGE_BITS(I, J, MASK) = IOR(IAND(I, MASK), IAND(J, NOT(MASK)))
  aiir::Value ones = builder.createAllOnesInteger(loc, signlessType);
  aiir::Value notMask = builder.createUnsigned<aiir::arith::XOrIOp>(
      loc, resultType, args[2], ones);
  aiir::Value lft = builder.createUnsigned<aiir::arith::AndIOp>(
      loc, resultType, args[0], args[2]);
  aiir::Value rgt = builder.createUnsigned<aiir::arith::AndIOp>(
      loc, resultType, args[1], notMask);
  return builder.createUnsigned<aiir::arith::OrIOp>(loc, resultType, lft, rgt);
}

// MOD
static aiir::Value genFastMod(fir::FirOpBuilder &builder, aiir::Location loc,
                              aiir::Value a, aiir::Value p) {
  auto fastmathFlags = aiir::arith::FastMathFlags::contract;
  auto fastmathAttr =
      aiir::arith::FastMathFlagsAttr::get(builder.getContext(), fastmathFlags);
  aiir::Value divResult =
      aiir::arith::DivFOp::create(builder, loc, a, p, fastmathAttr);
  aiir::Type intType = builder.getIntegerType(
      a.getType().getIntOrFloatBitWidth(), /*signed=*/true);
  aiir::Value intResult = builder.createConvert(loc, intType, divResult);
  aiir::Value cnvResult = builder.createConvert(loc, a.getType(), intResult);
  aiir::Value mulResult =
      aiir::arith::MulFOp::create(builder, loc, cnvResult, p, fastmathAttr);
  aiir::Value subResult =
      aiir::arith::SubFOp::create(builder, loc, a, mulResult, fastmathAttr);
  return subResult;
}

aiir::Value IntrinsicLibrary::genMod(aiir::Type resultType,
                                     llvm::ArrayRef<aiir::Value> args) {
  auto mod = builder.getModule();
  bool useFastRealMod = false;
  if (auto attr = mod->getAttrOfType<aiir::BoolAttr>("fir.fast_real_mod"))
    useFastRealMod = attr.getValue();

  assert(args.size() == 2);
  if (resultType.isUnsignedInteger()) {
    aiir::Type signlessType = aiir::IntegerType::get(
        builder.getContext(), resultType.getIntOrFloatBitWidth(),
        aiir::IntegerType::SignednessSemantics::Signless);
    return builder.createUnsigned<aiir::arith::RemUIOp>(loc, signlessType,
                                                        args[0], args[1]);
  }
  if (aiir::isa<aiir::IntegerType>(resultType))
    return aiir::arith::RemSIOp::create(builder, loc, args[0], args[1]);

  if (resultType.isFloat() && useFastRealMod) {
    // Treat MOD as an approximate function and code-gen inline code
    // instead of calling into the Fortran runtime library.
    return builder.createConvert(loc, resultType,
                                 genFastMod(builder, loc, args[0], args[1]));
  } else {
    // Use runtime.
    return builder.createConvert(
        loc, resultType, fir::runtime::genMod(builder, loc, args[0], args[1]));
  }
}

// MODULO
aiir::Value IntrinsicLibrary::genModulo(aiir::Type resultType,
                                        llvm::ArrayRef<aiir::Value> args) {
  // TODO: we'd better generate a runtime call here, when runtime error
  // checking is needed (to detect 0 divisor) or when precise math is requested.
  assert(args.size() == 2);
  // No floored modulo op in LLVM/AIIR yet. TODO: add one to AIIR.
  // In the meantime, use a simple inlined implementation based on truncated
  // modulo (MOD(A, P) implemented by RemIOp, RemFOp). This avoids making manual
  // division and multiplication from MODULO formula.
  //  - If A/P > 0 or MOD(A,P)=0, then INT(A/P) = FLOOR(A/P), and MODULO = MOD.
  //  - Otherwise, when A/P < 0 and MOD(A,P) !=0, then MODULO(A, P) =
  //    A-FLOOR(A/P)*P = A-(INT(A/P)-1)*P = A-INT(A/P)*P+P = MOD(A,P)+P
  // Note that A/P < 0 if and only if A and P signs are different.
  if (resultType.isUnsignedInteger()) {
    aiir::Type signlessType = aiir::IntegerType::get(
        builder.getContext(), resultType.getIntOrFloatBitWidth(),
        aiir::IntegerType::SignednessSemantics::Signless);
    return builder.createUnsigned<aiir::arith::RemUIOp>(loc, signlessType,
                                                        args[0], args[1]);
  }
  if (aiir::isa<aiir::IntegerType>(resultType)) {
    auto remainder =
        aiir::arith::RemSIOp::create(builder, loc, args[0], args[1]);
    auto argXor = aiir::arith::XOrIOp::create(builder, loc, args[0], args[1]);
    aiir::Value zero = builder.createIntegerConstant(loc, argXor.getType(), 0);
    auto argSignDifferent = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::slt, argXor, zero);
    auto remainderIsNotZero = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::ne, remainder, zero);
    auto mustAddP = aiir::arith::AndIOp::create(
        builder, loc, remainderIsNotZero, argSignDifferent);
    auto remPlusP =
        aiir::arith::AddIOp::create(builder, loc, remainder, args[1]);
    return aiir::arith::SelectOp::create(builder, loc, mustAddP, remPlusP,
                                         remainder);
  }

  auto fastMathFlags = builder.getFastMathFlags();
  // F128 arith::RemFOp may be lowered to a runtime call that may be unsupported
  // on the target, so generate a call to Fortran Runtime's ModuloReal16.
  if (resultType == aiir::Float128Type::get(builder.getContext()) ||
      (fastMathFlags & aiir::arith::FastMathFlags::ninf) ==
          aiir::arith::FastMathFlags::none)
    return builder.createConvert(
        loc, resultType,
        fir::runtime::genModulo(builder, loc, args[0], args[1]));

  auto remainder = aiir::arith::RemFOp::create(builder, loc, args[0], args[1]);
  aiir::Value zero = builder.createRealZeroConstant(loc, remainder.getType());
  auto remainderIsNotZero = aiir::arith::CmpFOp::create(
      builder, loc, aiir::arith::CmpFPredicate::UNE, remainder, zero);
  auto aLessThanZero = aiir::arith::CmpFOp::create(
      builder, loc, aiir::arith::CmpFPredicate::OLT, args[0], zero);
  auto pLessThanZero = aiir::arith::CmpFOp::create(
      builder, loc, aiir::arith::CmpFPredicate::OLT, args[1], zero);
  auto argSignDifferent =
      aiir::arith::XOrIOp::create(builder, loc, aLessThanZero, pLessThanZero);
  auto mustAddP = aiir::arith::AndIOp::create(builder, loc, remainderIsNotZero,
                                              argSignDifferent);
  auto remPlusP = aiir::arith::AddFOp::create(builder, loc, remainder, args[1]);
  return aiir::arith::SelectOp::create(builder, loc, mustAddP, remPlusP,
                                       remainder);
}

void IntrinsicLibrary::genMoveAlloc(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);

  const fir::ExtendedValue &from = args[0];
  const fir::ExtendedValue &to = args[1];
  const fir::ExtendedValue &status = args[2];
  const fir::ExtendedValue &errMsg = args[3];

  aiir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
  aiir::Value errBox =
      isStaticallyPresent(errMsg)
          ? fir::getBase(errMsg)
          : fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();

  const fir::MutableBoxValue *fromBox = from.getBoxOf<fir::MutableBoxValue>();
  const fir::MutableBoxValue *toBox = to.getBoxOf<fir::MutableBoxValue>();

  assert(fromBox && toBox && "move_alloc parameters must be mutable arrays");

  aiir::Value fromAddr = fir::factory::getMutableIRBox(builder, loc, *fromBox);
  aiir::Value toAddr = fir::factory::getMutableIRBox(builder, loc, *toBox);

  aiir::Value hasStat = builder.createBool(loc, isStaticallyPresent(status));

  aiir::Value stat = fir::runtime::genMoveAlloc(builder, loc, toAddr, fromAddr,
                                                hasStat, errBox);

  fir::factory::syncMutableBoxFromIRBox(builder, loc, *fromBox);
  fir::factory::syncMutableBoxFromIRBox(builder, loc, *toBox);

  if (isStaticallyPresent(status)) {
    aiir::Value statAddr = fir::getBase(status);
    aiir::Value statIsPresentAtRuntime =
        builder.genIsNotNullAddr(loc, statAddr);
    builder.genIfThen(loc, statIsPresentAtRuntime)
        .genThen([&]() { builder.createStoreWithConvert(loc, stat, statAddr); })
        .end();
  }
}

// MVBITS
void IntrinsicLibrary::genMvbits(llvm::ArrayRef<fir::ExtendedValue> args) {
  // A conformant MVBITS(FROM,FROMPOS,LEN,TO,TOPOS) call satisfies:
  //     FROMPOS >= 0
  //     LEN >= 0
  //     TOPOS >= 0
  //     FROMPOS + LEN <= BIT_SIZE(FROM)
  //     TOPOS + LEN <= BIT_SIZE(TO)
  // MASK = -1 >> (BIT_SIZE(FROM) - LEN)
  // TO = LEN == 0 ? TO : ((!(MASK << TOPOS)) & TO) |
  //                      (((FROM >> FROMPOS) & MASK) << TOPOS)
  assert(args.size() == 5);
  auto unbox = [&](fir::ExtendedValue exv) {
    const aiir::Value *arg = exv.getUnboxed();
    assert(arg && "nonscalar mvbits argument");
    return *arg;
  };
  aiir::Value from = unbox(args[0]);
  aiir::Type fromType = from.getType();
  aiir::Type signlessType = aiir::IntegerType::get(
      builder.getContext(), fromType.getIntOrFloatBitWidth(),
      aiir::IntegerType::SignednessSemantics::Signless);
  aiir::Value frompos =
      builder.createConvert(loc, signlessType, unbox(args[1]));
  aiir::Value len = builder.createConvert(loc, signlessType, unbox(args[2]));
  aiir::Value toAddr = unbox(args[3]);
  aiir::Type toType{fir::dyn_cast_ptrEleTy(toAddr.getType())};
  assert(toType.getIntOrFloatBitWidth() == fromType.getIntOrFloatBitWidth() &&
         "mismatched mvbits types");
  auto to = fir::LoadOp::create(builder, loc, signlessType, toAddr);
  aiir::Value topos = builder.createConvert(loc, signlessType, unbox(args[4]));
  aiir::Value zero = builder.createIntegerConstant(loc, signlessType, 0);
  aiir::Value ones = builder.createAllOnesInteger(loc, signlessType);
  aiir::Value bitSize = builder.createIntegerConstant(
      loc, signlessType,
      aiir::cast<aiir::IntegerType>(signlessType).getWidth());
  auto shiftCount = aiir::arith::SubIOp::create(builder, loc, bitSize, len);
  auto mask = aiir::arith::ShRUIOp::create(builder, loc, ones, shiftCount);
  auto unchangedTmp1 = aiir::arith::ShLIOp::create(builder, loc, mask, topos);
  auto unchangedTmp2 =
      aiir::arith::XOrIOp::create(builder, loc, unchangedTmp1, ones);
  auto unchanged = aiir::arith::AndIOp::create(builder, loc, unchangedTmp2, to);
  if (fromType.isUnsignedInteger())
    from = builder.createConvert(loc, signlessType, from);
  auto frombitsTmp1 = aiir::arith::ShRUIOp::create(builder, loc, from, frompos);
  auto frombitsTmp2 =
      aiir::arith::AndIOp::create(builder, loc, frombitsTmp1, mask);
  auto frombits =
      aiir::arith::ShLIOp::create(builder, loc, frombitsTmp2, topos);
  auto resTmp = aiir::arith::OrIOp::create(builder, loc, unchanged, frombits);
  auto lenIsZero = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::eq, len, zero);
  aiir::Value res =
      aiir::arith::SelectOp::create(builder, loc, lenIsZero, to, resTmp);
  if (toType.isUnsignedInteger())
    res = builder.createConvert(loc, toType, res);
  fir::StoreOp::create(builder, loc, res, toAddr);
}

// NEAREST, IEEE_NEXT_AFTER, IEEE_NEXT_DOWN, IEEE_NEXT_UP
template <I::NearestProc proc>
aiir::Value IntrinsicLibrary::genNearest(aiir::Type resultType,
                                         llvm::ArrayRef<aiir::Value> args) {
  // NEAREST
  //   Return the number adjacent to arg X in the direction of the infinity
  //   with the sign of arg S. Terminate with an error if arg S is zero.
  //   Generate exceptions as for IEEE_NEXT_AFTER.
  // IEEE_NEXT_AFTER
  //   Return isNan(Y) ? NaN : X==Y ? X : num adjacent to X in the dir of Y.
  //   Signal IEEE_OVERFLOW, IEEE_INEXACT for finite X and infinite result.
  //   Signal IEEE_UNDERFLOW, IEEE_INEXACT for subnormal result.
  // IEEE_NEXT_DOWN
  //   Return the number adjacent to X and less than X.
  //   Signal IEEE_INVALID when X is a signaling NaN.
  // IEEE_NEXT_UP
  //   Return the number adjacent to X and greater than X.
  //   Signal IEEE_INVALID when X is a signaling NaN.
  //
  // valueUp     -- true if a finite result must be larger than X.
  // magnitudeUp -- true if a finite abs(result) must be larger than abs(X).
  //
  // if (isNextAfter && isNan(Y)) X = NaN // result = NaN
  // if (isNan(X) || (isNextAfter && X == Y) || (isInfinite(X) && magnitudeUp))
  //   result = X
  // else if (isZero(X))
  //   result = valueUp ? minPositiveSubnormal : minNegativeSubnormal
  // else
  //   result = magUp ? (X + minPositiveSubnormal) : (X - minPositiveSubnormal)

  assert(args.size() == 1 || args.size() == 2);
  aiir::Value x = args[0];
  aiir::FloatType xType = aiir::dyn_cast<aiir::FloatType>(x.getType());
  const unsigned xBitWidth = xType.getWidth();
  aiir::Type i1Ty = builder.getI1Type();
  if constexpr (proc == NearestProc::NextAfter) {
    // If isNan(Y), set X to a qNaN that will propagate to the resultIsX result.
    aiir::Value qNan = genQNan(xType);
    aiir::Value isFPClass = genIsFPClass(i1Ty, args[1], nanTest);
    x = aiir::arith::SelectOp::create(builder, loc, isFPClass, qNan, x);
  }
  aiir::Value resultIsX = genIsFPClass(i1Ty, x, nanTest);
  aiir::Type intType = builder.getIntegerType(xBitWidth);
  aiir::Value one = builder.createIntegerConstant(loc, intType, 1);

  // Set valueUp to true if a finite result must be larger than arg X.
  aiir::Value valueUp;
  if constexpr (proc == NearestProc::Nearest) {
    // Arg S must not be zero.
    fir::IfOp ifOp =
        fir::IfOp::create(builder, loc, genIsFPClass(i1Ty, args[1], zeroTest),
                          /*withElseRegion=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    fir::runtime::genReportFatalUserError(
        builder, loc, "intrinsic nearest S argument is zero");
    builder.setInsertionPointAfter(ifOp);
    aiir::Value sSign = IntrinsicLibrary::genIeeeSignbit(intType, {args[1]});
    valueUp = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::ne, sSign, one);
  } else if constexpr (proc == NearestProc::NextAfter) {
    // Convert X and Y to a common type to allow comparison. Direct conversions
    // between kinds 2, 3, 10, and 16 are not all supported. These conversions
    // are implemented by converting kind=2,3 values to kind=4, possibly
    // followed with a conversion of that value to a larger type.
    aiir::Value x1 = x;
    aiir::Value y = args[1];
    aiir::FloatType yType = aiir::dyn_cast<aiir::FloatType>(args[1].getType());
    const unsigned yBitWidth = yType.getWidth();
    if (xType != yType) {
      aiir::Type f32Ty = aiir::Float32Type::get(builder.getContext());
      if (xBitWidth < 32)
        x1 = builder.createConvert(loc, f32Ty, x1);
      if (yBitWidth > 32 && yBitWidth > xBitWidth)
        x1 = builder.createConvert(loc, yType, x1);
      if (yBitWidth < 32)
        y = builder.createConvert(loc, f32Ty, y);
      if (xBitWidth > 32 && xBitWidth > yBitWidth)
        y = builder.createConvert(loc, xType, y);
    }
    resultIsX = aiir::arith::OrIOp::create(
        builder, loc, resultIsX,
        aiir::arith::CmpFOp::create(builder, loc,
                                    aiir::arith::CmpFPredicate::OEQ, x1, y));
    valueUp = aiir::arith::CmpFOp::create(
        builder, loc, aiir::arith::CmpFPredicate::OLT, x1, y);
  } else if constexpr (proc == NearestProc::NextDown) {
    valueUp = builder.createBool(loc, false);
  } else if constexpr (proc == NearestProc::NextUp) {
    valueUp = builder.createBool(loc, true);
  }
  aiir::Value magnitudeUp = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::ne, valueUp,
      IntrinsicLibrary::genIeeeSignbit(i1Ty, {args[0]}));
  resultIsX = aiir::arith::OrIOp::create(
      builder, loc, resultIsX,
      aiir::arith::AndIOp::create(
          builder, loc, genIsFPClass(i1Ty, x, infiniteTest), magnitudeUp));

  // Result is X. (For ieee_next_after with isNan(Y), X has been set to a NaN.)
  fir::IfOp outerIfOp = fir::IfOp::create(builder, loc, resultType, resultIsX,
                                          /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&outerIfOp.getThenRegion().front());
  if constexpr (proc == NearestProc::NextDown || proc == NearestProc::NextUp)
    genRaiseExcept(_FORTRAN_RUNTIME_IEEE_INVALID,
                   genIsFPClass(i1Ty, x, snanTest));
  fir::ResultOp::create(builder, loc, x);

  // Result is minPositiveSubnormal or minNegativeSubnormal. (X is zero.)
  builder.setInsertionPointToStart(&outerIfOp.getElseRegion().front());
  aiir::Value resultIsMinSubnormal = aiir::arith::CmpFOp::create(
      builder, loc, aiir::arith::CmpFPredicate::OEQ, x,
      builder.createRealZeroConstant(loc, xType));
  fir::IfOp innerIfOp =
      fir::IfOp::create(builder, loc, resultType, resultIsMinSubnormal,
                        /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&innerIfOp.getThenRegion().front());
  aiir::Value minPositiveSubnormal =
      aiir::arith::BitcastOp::create(builder, loc, resultType, one);
  aiir::Value minNegativeSubnormal = aiir::arith::BitcastOp::create(
      builder, loc, resultType,
      aiir::arith::ConstantOp::create(
          builder, loc, intType,
          builder.getIntegerAttr(
              intType, llvm::APInt::getBitsSetWithWrap(
                           xBitWidth, /*lo=*/xBitWidth - 1, /*hi=*/1))));
  aiir::Value result = aiir::arith::SelectOp::create(
      builder, loc, valueUp, minPositiveSubnormal, minNegativeSubnormal);
  if constexpr (proc == NearestProc::Nearest || proc == NearestProc::NextAfter)
    genRaiseExcept(_FORTRAN_RUNTIME_IEEE_UNDERFLOW |
                   _FORTRAN_RUNTIME_IEEE_INEXACT);
  fir::ResultOp::create(builder, loc, result);

  // Result is (X + minPositiveSubnormal) or (X - minPositiveSubnormal).
  builder.setInsertionPointToStart(&innerIfOp.getElseRegion().front());
  if (xBitWidth == 80) {
    // Kind 10. Call std::nextafter, which generates exceptions as required
    // for ieee_next_after and nearest. Override this exception processing
    // for ieee_next_down and ieee_next_up.
    constexpr bool overrideExceptionGeneration =
        proc == NearestProc::NextDown || proc == NearestProc::NextUp;
    [[maybe_unused]] aiir::Type i32Ty;
    [[maybe_unused]] aiir::Value allExcepts, excepts, mask;
    if constexpr (overrideExceptionGeneration) {
      i32Ty = builder.getIntegerType(32);
      allExcepts = fir::runtime::genMapExcept(
          builder, loc,
          builder.createIntegerConstant(loc, i32Ty, _FORTRAN_RUNTIME_IEEE_ALL));
      excepts = genRuntimeCall("fetestexcept", i32Ty, allExcepts);
      mask = genRuntimeCall("fedisableexcept", i32Ty, allExcepts);
    }
    result = fir::runtime::genNearest(builder, loc, x, valueUp);
    if constexpr (overrideExceptionGeneration) {
      genRuntimeCall("feclearexcept", i32Ty, allExcepts);
      genRuntimeCall("feraiseexcept", i32Ty, excepts);
      genRuntimeCall("feenableexcept", i32Ty, mask);
    }
    fir::ResultOp::create(builder, loc, result);
  } else {
    // Kind 2, 3, 4, 8, 16. Increment or decrement X cast to integer.
    aiir::Value intX = aiir::arith::BitcastOp::create(builder, loc, intType, x);
    aiir::Value add = aiir::arith::AddIOp::create(builder, loc, intX, one);
    aiir::Value sub = aiir::arith::SubIOp::create(builder, loc, intX, one);
    result = aiir::arith::BitcastOp::create(
        builder, loc, resultType,
        aiir::arith::SelectOp::create(builder, loc, magnitudeUp, add, sub));
    if constexpr (proc == NearestProc::Nearest ||
                  proc == NearestProc::NextAfter) {
      genRaiseExcept(_FORTRAN_RUNTIME_IEEE_OVERFLOW |
                         _FORTRAN_RUNTIME_IEEE_INEXACT,
                     genIsFPClass(i1Ty, result, infiniteTest));
      genRaiseExcept(_FORTRAN_RUNTIME_IEEE_UNDERFLOW |
                         _FORTRAN_RUNTIME_IEEE_INEXACT,
                     genIsFPClass(i1Ty, result, subnormalTest));
    }
    fir::ResultOp::create(builder, loc, result);
  }

  builder.setInsertionPointAfter(innerIfOp);
  fir::ResultOp::create(builder, loc, innerIfOp.getResult(0));
  builder.setInsertionPointAfter(outerIfOp);
  return outerIfOp.getResult(0);
}

// NINT
aiir::Value IntrinsicLibrary::genNint(aiir::Type resultType,
                                      llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() >= 1);
  // Skip optional kind argument to search the runtime; it is already reflected
  // in result type.
  return genRuntimeCall("nint", resultType, {args[0]});
}

// NORM2
fir::ExtendedValue
IntrinsicLibrary::genNorm2(aiir::Type resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);

  // Handle required array argument
  aiir::Value array = builder.createBox(loc, args[0]);
  unsigned rank = fir::BoxValue(array).rank();
  assert(rank >= 1);

  // Check if the dim argument is present
  bool absentDim = isStaticallyAbsent(args[1]);

  // If dim argument is absent or the array is rank 1, then the result is
  // a scalar (since the the result is rank-1 or 0). Otherwise, the result is
  // an array.
  if (absentDim || rank == 1) {
    return fir::runtime::genNorm2(builder, loc, array);
  } else {
    // Create mutable fir.box to be passed to the runtime for the result.
    aiir::Type resultArrayType = builder.getVarLenSeqTy(resultType, rank - 1);
    fir::MutableBoxValue resultMutableBox =
        fir::factory::createTempMutableBox(builder, loc, resultArrayType);
    aiir::Value resultIrBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    aiir::Value dim = fir::getBase(args[1]);
    fir::runtime::genNorm2Dim(builder, loc, resultIrBox, array, dim);
    // Handle cleanup of allocatable result descriptor and return
    return readAndAddCleanUp(resultMutableBox, resultType, "NORM2");
  }
}

// NOT
aiir::Value IntrinsicLibrary::genNot(aiir::Type resultType,
                                     llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);
  aiir::Type signlessType = aiir::IntegerType::get(
      builder.getContext(), resultType.getIntOrFloatBitWidth(),
      aiir::IntegerType::SignednessSemantics::Signless);
  aiir::Value allOnes = builder.createAllOnesInteger(loc, signlessType);
  return builder.createUnsigned<aiir::arith::XOrIOp>(loc, resultType, args[0],
                                                     allOnes);
}

// NULL
fir::ExtendedValue
IntrinsicLibrary::genNull(aiir::Type, llvm::ArrayRef<fir::ExtendedValue> args) {
  // NULL() without MOLD must be handled in the contexts where it can appear
  // (see table 16.5 of Fortran 2018 standard).
  assert(args.size() == 1 && isStaticallyPresent(args[0]) &&
         "MOLD argument required to lower NULL outside of any context");
  aiir::Type ptrTy = fir::getBase(args[0]).getType();
  if (ptrTy && fir::isBoxProcAddressType(ptrTy)) {
    auto boxProcType = aiir::cast<fir::BoxProcType>(fir::unwrapRefType(ptrTy));
    aiir::Value boxStorage = builder.createTemporary(loc, boxProcType);
    aiir::Value nullBoxProc =
        fir::factory::createNullBoxProc(builder, loc, boxProcType);
    builder.createStoreWithConvert(loc, nullBoxProc, boxStorage);
    return boxStorage;
  }
  const auto *mold = args[0].getBoxOf<fir::MutableBoxValue>();
  assert(mold && "MOLD must be a pointer or allocatable");
  fir::BaseBoxType boxType = mold->getBoxTy();
  aiir::Value boxStorage = builder.createTemporary(loc, boxType);
  aiir::Value box = fir::factory::createUnallocatedBox(
      builder, loc, boxType, mold->nonDeferredLenParams());
  fir::StoreOp::create(builder, loc, box, boxStorage);
  return fir::MutableBoxValue(boxStorage, mold->nonDeferredLenParams(), {});
}

// NUM_IMAGES
fir::ExtendedValue
IntrinsicLibrary::genNumImages(aiir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args) {
  converter->checkCoarrayEnabled();
  assert(args.size() == 0 || args.size() == 1);

  if (args.size())
    return mif::NumImagesOp::create(builder, loc, fir::getBase(args[0]))
        .getResult();
  return mif::NumImagesOp::create(builder, loc).getResult();
}

// PACK
fir::ExtendedValue
IntrinsicLibrary::genPack(aiir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  [[maybe_unused]] auto numArgs = args.size();
  assert(numArgs == 2 || numArgs == 3);

  // Handle required array argument
  aiir::Value array = builder.createBox(loc, args[0]);

  // Handle required mask argument
  aiir::Value mask = builder.createBox(loc, args[1]);

  // Handle optional vector argument
  aiir::Value vector =
      isStaticallyAbsent(args, 2)
          ? fir::AbsentOp::create(builder, loc,
                                  fir::BoxType::get(builder.getI1Type()))
          : builder.createBox(loc, args[2]);

  // Create mutable fir.box to be passed to the runtime for the result.
  aiir::Type resultArrayType = builder.getVarLenSeqTy(resultType, 1);
  fir::MutableBoxValue resultMutableBox = fir::factory::createTempMutableBox(
      builder, loc, resultArrayType, {},
      fir::isPolymorphicType(array.getType()) ? array : aiir::Value{});
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genPack(builder, loc, resultIrBox, array, mask, vector);

  return readAndAddCleanUp(resultMutableBox, resultType, "PACK");
}

// PARITY
fir::ExtendedValue
IntrinsicLibrary::genParity(aiir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 2);
  // Handle required mask argument
  aiir::Value mask = builder.createBox(loc, args[0]);

  fir::BoxValue maskArry = builder.createBox(loc, args[0]);
  int rank = maskArry.rank();
  assert(rank >= 1);

  // Handle optional dim argument
  bool absentDim = isStaticallyAbsent(args[1]);
  aiir::Value dim =
      absentDim ? builder.createIntegerConstant(loc, builder.getIndexType(), 1)
                : fir::getBase(args[1]);

  if (rank == 1 || absentDim)
    return builder.createConvert(
        loc, resultType, fir::runtime::genParity(builder, loc, mask, dim));

  // else use the result descriptor ParityDim() intrinsic

  // Create mutable fir.box to be passed to the runtime for the result.

  aiir::Type resultArrayType = builder.getVarLenSeqTy(resultType, rank - 1);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  // Call runtime. The runtime is allocating the result.
  fir::runtime::genParityDescriptor(builder, loc, resultIrBox, mask, dim);
  return readAndAddCleanUp(resultMutableBox, resultType, "PARITY");
}

// PERROR
void IntrinsicLibrary::genPerror(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);

  fir::ExtendedValue str = args[0];
  const auto *box = str.getBoxOf<fir::BoxValue>();
  aiir::Value addr =
      fir::BoxAddrOp::create(builder, loc, box->getMemTy(), fir::getBase(*box));
  fir::runtime::genPerror(builder, loc, addr);
}

// POPCNT
aiir::Value IntrinsicLibrary::genPopcnt(aiir::Type resultType,
                                        llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);

  aiir::Value count = aiir::math::CtPopOp::create(builder, loc, args);

  return builder.createConvert(loc, resultType, count);
}

// POPPAR
aiir::Value IntrinsicLibrary::genPoppar(aiir::Type resultType,
                                        llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);

  aiir::Value count = genPopcnt(resultType, args);
  aiir::Value one = builder.createIntegerConstant(loc, resultType, 1);

  return aiir::arith::AndIOp::create(builder, loc, count, one);
}

// PRESENT
fir::ExtendedValue
IntrinsicLibrary::genPresent(aiir::Type,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  return fir::IsPresentOp::create(builder, loc, builder.getI1Type(),
                                  fir::getBase(args[0]));
}

// PRODUCT
fir::ExtendedValue
IntrinsicLibrary::genProduct(aiir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  return genReduction(fir::runtime::genProduct, fir::runtime::genProductDim,
                      "PRODUCT", resultType, args);
}

// PUTENV
fir::ExtendedValue
IntrinsicLibrary::genPutenv(std::optional<aiir::Type> resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert((resultType.has_value() && args.size() == 1) ||
         (!resultType.has_value() && args.size() >= 1 && args.size() <= 2));

  aiir::Value str = fir::getBase(args[0]);
  aiir::Value strLength = fir::getLen(args[0]);
  aiir::Value statusValue =
      fir::runtime::genPutEnv(builder, loc, str, strLength);

  if (resultType.has_value()) {
    // Function form, return status.
    return builder.createConvert(loc, *resultType, statusValue);
  }

  // Subroutine form, store status and return none.
  const fir::ExtendedValue &status = args[1];
  if (!isStaticallyAbsent(status)) {
    aiir::Value statusAddr = fir::getBase(status);
    aiir::Value statusIsPresentAtRuntime =
        builder.genIsNotNullAddr(loc, statusAddr);
    builder.genIfThen(loc, statusIsPresentAtRuntime)
        .genThen([&]() {
          builder.createStoreWithConvert(loc, statusValue, statusAddr);
        })
        .end();
  }

  return {};
}

// RAND
fir::ExtendedValue
IntrinsicLibrary::genRand(aiir::Type, llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  aiir::Value i =
      isStaticallyPresent(args[0])
          ? fir::getBase(args[0])
          : fir::AbsentOp::create(builder, loc,
                                  builder.getRefType(builder.getI32Type()))
                .getResult();
  return fir::runtime::genRand(builder, loc, i);
}

// RANDOM_INIT
void IntrinsicLibrary::genRandomInit(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  fir::runtime::genRandomInit(builder, loc, fir::getBase(args[0]),
                              fir::getBase(args[1]));
}

// RANDOM_NUMBER
void IntrinsicLibrary::genRandomNumber(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  fir::runtime::genRandomNumber(builder, loc, fir::getBase(args[0]));
}

// RANDOM_SEED
void IntrinsicLibrary::genRandomSeed(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  aiir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
  auto getDesc = [&](int i) {
    return isStaticallyPresent(args[i])
               ? fir::getBase(args[i])
               : fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();
  };
  aiir::Value size = getDesc(0);
  aiir::Value put = getDesc(1);
  aiir::Value get = getDesc(2);
  fir::runtime::genRandomSeed(builder, loc, size, put, get);
}

// REDUCE
fir::ExtendedValue
IntrinsicLibrary::genReduce(aiir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 6);

  fir::BoxValue arrayTmp = builder.createBox(loc, args[0]);
  aiir::Value array = fir::getBase(arrayTmp);
  aiir::Value operation = fir::getBase(args[1]);
  int rank = arrayTmp.rank();
  assert(rank >= 1);

  // Arguements to the reduction operation are passed by reference or value?
  bool argByRef = true;
  if (!operation.getDefiningOp())
    TODO(loc, "Distinguigh dummy procedure arguments");
  if (auto embox =
          aiir::dyn_cast_or_null<fir::EmboxProcOp>(operation.getDefiningOp())) {
    auto fctTy = aiir::dyn_cast<aiir::FunctionType>(embox.getFunc().getType());
    argByRef = aiir::isa<fir::ReferenceType>(fctTy.getInput(0));
  } else if (auto load = aiir::dyn_cast_or_null<fir::LoadOp>(
                 operation.getDefiningOp())) {
    auto boxProcTy = aiir::dyn_cast_or_null<fir::BoxProcType>(load.getType());
    assert(boxProcTy && "expect BoxProcType");
    auto fctTy = aiir::dyn_cast<aiir::FunctionType>(boxProcTy.getEleTy());
    argByRef = aiir::isa<fir::ReferenceType>(fctTy.getInput(0));
  }

  aiir::Type ty = array.getType();
  aiir::Type arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
  aiir::Type eleTy = aiir::cast<fir::SequenceType>(arrTy).getElementType();

  // Handle optional arguments
  bool absentDim = isStaticallyAbsent(args[2]);

  auto mask = isStaticallyAbsent(args[3])
                  ? fir::AbsentOp::create(
                        builder, loc, fir::BoxType::get(builder.getI1Type()))
                  : builder.createBox(loc, args[3]);

  aiir::Value identity =
      isStaticallyAbsent(args[4])
          ? fir::AbsentOp::create(builder, loc, fir::ReferenceType::get(eleTy))
          : fir::getBase(args[4]);

  aiir::Value ordered = isStaticallyAbsent(args[5])
                            ? builder.createBool(loc, false)
                            : fir::getBase(args[5]);

  // We call the type specific versions because the result is scalar
  // in the case below.
  if (absentDim || rank == 1) {
    if (fir::isa_complex(eleTy) || fir::isa_derived(eleTy)) {
      aiir::Value result = builder.createTemporary(loc, eleTy);
      fir::runtime::genReduce(builder, loc, array, operation, mask, identity,
                              ordered, result, argByRef);
      if (fir::isa_derived(eleTy))
        return result;
      return fir::LoadOp::create(builder, loc, result);
    }
    if (fir::isa_char(eleTy)) {
      auto charTy = aiir::dyn_cast_or_null<fir::CharacterType>(resultType);
      assert(charTy && "expect CharacterType");
      fir::factory::CharacterExprHelper charHelper(builder, loc);
      aiir::Value len;
      if (charTy.hasDynamicLen())
        len = charHelper.readLengthFromBox(fir::getBase(arrayTmp), charTy);
      else
        len = builder.createIntegerConstant(loc, builder.getI32Type(),
                                            charTy.getLen());
      fir::CharBoxValue temp = charHelper.createCharacterTemp(eleTy, len);
      fir::runtime::genReduce(builder, loc, array, operation, mask, identity,
                              ordered, temp.getBuffer(), argByRef);
      return temp;
    }
    return fir::runtime::genReduce(builder, loc, array, operation, mask,
                                   identity, ordered, argByRef);
  }
  // Handle cases that have an array result.
  // Create mutable fir.box to be passed to the runtime for the result.
  aiir::Type resultArrayType = builder.getVarLenSeqTy(resultType, rank - 1);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  aiir::Value dim = fir::getBase(args[2]);
  fir::runtime::genReduceDim(builder, loc, array, operation, dim, mask,
                             identity, ordered, resultIrBox, argByRef);
  return readAndAddCleanUp(resultMutableBox, resultType, "REDUCE");
}

// RENAME
fir::ExtendedValue
IntrinsicLibrary::genRename(std::optional<aiir::Type> resultType,
                            aiir::ArrayRef<fir::ExtendedValue> args) {
  assert((args.size() == 3 && !resultType.has_value()) ||
         (args.size() == 2 && resultType.has_value()));

  aiir::Value path1 = fir::getBase(args[0]);
  aiir::Value path2 = fir::getBase(args[1]);
  if (!path1 || !path2)
    fir::emitFatalError(loc, "Expected at least two dummy arguments");

  if (resultType.has_value()) {
    // code-gen for the function form of RENAME
    auto statusAddr = builder.createTemporary(loc, *resultType);
    auto statusBox = builder.createBox(loc, statusAddr);
    fir::runtime::genRename(builder, loc, path1, path2, statusBox);
    return fir::LoadOp::create(builder, loc, statusAddr);
  } else {
    // code-gen for the procedure form of RENAME
    aiir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
    auto status = args[2];
    aiir::Value statusBox =
        isStaticallyPresent(status)
            ? fir::getBase(status)
            : fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();
    fir::runtime::genRename(builder, loc, path1, path2, statusBox);
    return {};
  }
}

// REPEAT
fir::ExtendedValue
IntrinsicLibrary::genRepeat(aiir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  aiir::Value string = builder.createBox(loc, args[0]);
  aiir::Value ncopies = fir::getBase(args[1]);
  // Create mutable fir.box to be passed to the runtime for the result.
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genRepeat(builder, loc, resultIrBox, string, ncopies);
  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  return readAndAddCleanUp(resultMutableBox, resultType, "REPEAT");
}

// RESHAPE
fir::ExtendedValue
IntrinsicLibrary::genReshape(aiir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);

  // Handle source argument
  aiir::Value source = builder.createBox(loc, args[0]);

  // Handle shape argument
  aiir::Value shape = builder.createBox(loc, args[1]);
  assert(fir::BoxValue(shape).rank() == 1);
  aiir::Type shapeTy = shape.getType();
  aiir::Type shapeArrTy = fir::dyn_cast_ptrOrBoxEleTy(shapeTy);
  auto resultRank = aiir::cast<fir::SequenceType>(shapeArrTy).getShape()[0];

  if (resultRank == fir::SequenceType::getUnknownExtent())
    TODO(loc, "intrinsic: reshape requires computing rank of result");

  // Handle optional pad argument
  aiir::Value pad =
      isStaticallyAbsent(args[2])
          ? fir::AbsentOp::create(builder, loc,
                                  fir::BoxType::get(builder.getI1Type()))
          : builder.createBox(loc, args[2]);

  // Handle optional order argument
  aiir::Value order =
      isStaticallyAbsent(args[3])
          ? fir::AbsentOp::create(builder, loc,
                                  fir::BoxType::get(builder.getI1Type()))
          : builder.createBox(loc, args[3]);

  // Create mutable fir.box to be passed to the runtime for the result.
  aiir::Type type = builder.getVarLenSeqTy(resultType, resultRank);
  fir::MutableBoxValue resultMutableBox = fir::factory::createTempMutableBox(
      builder, loc, type, {},
      fir::isPolymorphicType(source.getType()) ? source : aiir::Value{});

  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genReshape(builder, loc, resultIrBox, source, shape, pad,
                           order);

  return readAndAddCleanUp(resultMutableBox, resultType, "RESHAPE");
}

// RRSPACING
aiir::Value IntrinsicLibrary::genRRSpacing(aiir::Type resultType,
                                           llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genRRSpacing(builder, loc, fir::getBase(args[0])));
}

// ERFC_SCALED
aiir::Value IntrinsicLibrary::genErfcScaled(aiir::Type resultType,
                                            llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genErfcScaled(builder, loc, fir::getBase(args[0])));
}

// SAME_TYPE_AS
fir::ExtendedValue
IntrinsicLibrary::genSameTypeAs(aiir::Type resultType,
                                llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genSameTypeAs(builder, loc, fir::getBase(args[0]),
                                  fir::getBase(args[1])));
}

// SCALE
aiir::Value IntrinsicLibrary::genScale(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  aiir::FloatType floatTy = aiir::dyn_cast<aiir::FloatType>(resultType);
  if (!floatTy.isF16() && !floatTy.isBF16()) // kind=4,8,10,16
    return builder.createConvert(
        loc, resultType,
        fir::runtime::genScale(builder, loc, args[0], args[1]));

  // Convert kind=2,3 arg X to kind=4. Convert kind=4 result back to kind=2,3.
  aiir::Type i1Ty = builder.getI1Type();
  aiir::Type f32Ty = aiir::Float32Type::get(builder.getContext());
  aiir::Value result = builder.createConvert(
      loc, resultType,
      fir::runtime::genScale(
          builder, loc, builder.createConvert(loc, f32Ty, args[0]), args[1]));

  // kind=4 runtime::genScale call may not signal kind=2,3 exceptions.
  // If X is finite and result is infinite, signal IEEE_OVERFLOW
  // If X is finite and scale(result, -I) != X, signal IEEE_UNDERFLOW
  fir::IfOp outerIfOp =
      fir::IfOp::create(builder, loc, genIsFPClass(i1Ty, args[0], finiteTest),
                        /*withElseRegion=*/false);
  builder.setInsertionPointToStart(&outerIfOp.getThenRegion().front());
  fir::IfOp innerIfOp =
      fir::IfOp::create(builder, loc, genIsFPClass(i1Ty, result, infiniteTest),
                        /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&innerIfOp.getThenRegion().front());
  genRaiseExcept(_FORTRAN_RUNTIME_IEEE_OVERFLOW |
                 _FORTRAN_RUNTIME_IEEE_INEXACT);
  builder.setInsertionPointToStart(&innerIfOp.getElseRegion().front());
  aiir::Value minusI = aiir::arith::MulIOp::create(
      builder, loc, args[1],
      builder.createAllOnesInteger(loc, args[1].getType()));
  aiir::Value reverseResult = builder.createConvert(
      loc, resultType,
      fir::runtime::genScale(
          builder, loc, builder.createConvert(loc, f32Ty, result), minusI));
  genRaiseExcept(
      _FORTRAN_RUNTIME_IEEE_UNDERFLOW | _FORTRAN_RUNTIME_IEEE_INEXACT,
      aiir::arith::CmpFOp::create(builder, loc, aiir::arith::CmpFPredicate::ONE,
                                  args[0], reverseResult));
  builder.setInsertionPointAfter(outerIfOp);
  return result;
}

// SCAN
fir::ExtendedValue
IntrinsicLibrary::genScan(aiir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 4);

  if (isStaticallyAbsent(args[3])) {
    // Kind not specified, so call scan/verify runtime routine that is
    // specialized on the kind of characters in string.

    // Handle required string base arg
    aiir::Value stringBase = fir::getBase(args[0]);

    // Handle required set string base arg
    aiir::Value setBase = fir::getBase(args[1]);

    // Handle kind argument; it is the kind of character in this case
    fir::KindTy kind =
        fir::factory::CharacterExprHelper{builder, loc}.getCharacterKind(
            stringBase.getType());

    // Get string length argument
    aiir::Value stringLen = fir::getLen(args[0]);

    // Get set string length argument
    aiir::Value setLen = fir::getLen(args[1]);

    // Handle optional back argument
    aiir::Value back =
        isStaticallyAbsent(args[2])
            ? builder.createIntegerConstant(loc, builder.getI1Type(), 0)
            : fir::getBase(args[2]);

    return builder.createConvert(loc, resultType,
                                 fir::runtime::genScan(builder, loc, kind,
                                                       stringBase, stringLen,
                                                       setBase, setLen, back));
  }
  // else use the runtime descriptor version of scan/verify

  // Handle optional argument, back
  auto makeRefThenEmbox = [&](aiir::Value b) {
    fir::LogicalType logTy = fir::LogicalType::get(
        builder.getContext(), builder.getKindMap().defaultLogicalKind());
    aiir::Value temp = builder.createTemporary(loc, logTy);
    aiir::Value castb = builder.createConvert(loc, logTy, b);
    fir::StoreOp::create(builder, loc, castb, temp);
    return builder.createBox(loc, temp);
  };
  aiir::Value back =
      fir::isUnboxedValue(args[2])
          ? makeRefThenEmbox(*args[2].getUnboxed())
          : fir::AbsentOp::create(builder, loc,
                                  fir::BoxType::get(builder.getI1Type()));

  // Handle required string argument
  aiir::Value string = builder.createBox(loc, args[0]);

  // Handle required set argument
  aiir::Value set = builder.createBox(loc, args[1]);

  // Handle kind argument
  aiir::Value kind = fir::getBase(args[3]);

  // Create result descriptor
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genScanDescriptor(builder, loc, resultIrBox, string, set, back,
                                  kind);

  // Handle cleanup of allocatable result descriptor and return
  return readAndAddCleanUp(resultMutableBox, resultType, "SCAN");
}

// SECNDS
fir::ExtendedValue
IntrinsicLibrary::genSecnds(aiir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1 && "SECNDS expects one argument");

  aiir::Value refTime = fir::getBase(args[0]);

  if (!refTime)
    fir::emitFatalError(loc, "expected REFERENCE TIME parameter");

  aiir::Value result = fir::runtime::genSecnds(builder, loc, refTime);

  return builder.createConvert(loc, resultType, result);
}

// SECOND
fir::ExtendedValue
IntrinsicLibrary::genSecond(std::optional<aiir::Type> resultType,
                            aiir::ArrayRef<fir::ExtendedValue> args) {
  assert((args.size() == 1 && !resultType) || (args.empty() && resultType));

  fir::ExtendedValue result;

  if (resultType)
    result = builder.createTemporary(loc, *resultType);
  else
    result = args[0];

  llvm::SmallVector<fir::ExtendedValue, 1> subroutineArgs(1, result);
  genCpuTime(subroutineArgs);

  if (resultType)
    return fir::LoadOp::create(builder, loc, fir::getBase(result));
  return {};
}

// RTC
aiir::Value IntrinsicLibrary::genRtc(aiir::Type resultType,
                                     llvm::ArrayRef<aiir::Value> args) {
  assert(args.empty());
  aiir::Value time = fir::runtime::genTime(builder, loc);
  return builder.createConvert(loc, resultType, time);
}

// SELECTED_CHAR_KIND
fir::ExtendedValue
IntrinsicLibrary::genSelectedCharKind(aiir::Type resultType,
                                      llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genSelectedCharKind(builder, loc, fir::getBase(args[0]),
                                        fir::getLen(args[0])));
}

// SELECTED_INT_KIND
aiir::Value
IntrinsicLibrary::genSelectedIntKind(aiir::Type resultType,
                                     llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genSelectedIntKind(builder, loc, fir::getBase(args[0])));
}

// SELECTED_LOGICAL_KIND
aiir::Value
IntrinsicLibrary::genSelectedLogicalKind(aiir::Type resultType,
                                         llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);

  return builder.createConvert(loc, resultType,
                               fir::runtime::genSelectedLogicalKind(
                                   builder, loc, fir::getBase(args[0])));
}

// SELECTED_REAL_KIND
aiir::Value
IntrinsicLibrary::genSelectedRealKind(aiir::Type resultType,
                                      llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 3);

  // Handle optional precision(P) argument
  aiir::Value precision =
      isStaticallyAbsent(args[0])
          ? fir::AbsentOp::create(builder, loc,
                                  fir::ReferenceType::get(builder.getI1Type()))
          : fir::getBase(args[0]);

  // Handle optional range(R) argument
  aiir::Value range =
      isStaticallyAbsent(args[1])
          ? fir::AbsentOp::create(builder, loc,
                                  fir::ReferenceType::get(builder.getI1Type()))
          : fir::getBase(args[1]);

  // Handle optional radix(RADIX) argument
  aiir::Value radix =
      isStaticallyAbsent(args[2])
          ? fir::AbsentOp::create(builder, loc,
                                  fir::ReferenceType::get(builder.getI1Type()))
          : fir::getBase(args[2]);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genSelectedRealKind(builder, loc, precision, range, radix));
}

// SET_EXPONENT
aiir::Value IntrinsicLibrary::genSetExponent(aiir::Type resultType,
                                             llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genSetExponent(builder, loc, fir::getBase(args[0]),
                                   fir::getBase(args[1])));
}

/// Create a fir.box to be passed to the LBOUND/UBOUND runtime.
/// This ensure that local lower bounds of assumed shape are propagated and that
/// a fir.box with equivalent LBOUNDs.
static aiir::Value
createBoxForRuntimeBoundInquiry(aiir::Location loc, fir::FirOpBuilder &builder,
                                const fir::ExtendedValue &array) {
  // Assumed-rank descriptor must always carry accurate lower bound information
  // in lowering since they cannot be tracked on the side in a vector at compile
  // time.
  if (array.hasAssumedRank())
    return builder.createBox(loc, array);

  return array.match(
      [&](const fir::BoxValue &boxValue) -> aiir::Value {
        // This entity is mapped to a fir.box that may not contain the local
        // lower bound information if it is a dummy. Rebox it with the local
        // shape information.
        aiir::Value localShape = builder.createShape(loc, array);
        aiir::Value oldBox = boxValue.getAddr();
        return fir::ReboxOp::create(builder, loc, oldBox.getType(), oldBox,
                                    localShape,
                                    /*slice=*/aiir::Value{});
      },
      [&](const auto &) -> aiir::Value {
        // This is a pointer/allocatable, or an entity not yet tracked with a
        // fir.box. For pointer/allocatable, createBox will forward the
        // descriptor that contains the correct lower bound information. For
        // other entities, a new fir.box will be made with the local lower
        // bounds.
        return builder.createBox(loc, array);
      });
}

/// Generate runtime call to inquire about all the bounds/extents of an
/// array (or an assumed-rank).
template <typename Func>
static fir::ExtendedValue
genBoundInquiry(fir::FirOpBuilder &builder, aiir::Location loc,
                aiir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args,
                int kindPos, Func genRtCall, bool needAccurateLowerBound) {
  const fir::ExtendedValue &array = args[0];
  const bool hasAssumedRank = array.hasAssumedRank();
  aiir::Type resultElementType = fir::unwrapSequenceType(resultType);
  // For assumed-rank arrays, allocate an array with the maximum rank, that is
  // big enough to hold the result but still "small" (15 elements). Static size
  // alloca make stack analysis/manipulation easier.
  int rank = hasAssumedRank ? Fortran::common::maxRank : array.rank();
  aiir::Type allocSeqType = fir::SequenceType::get(rank, resultElementType);
  aiir::Value resultStorage = builder.createTemporary(loc, allocSeqType);
  aiir::Value arrayBox =
      needAccurateLowerBound
          ? createBoxForRuntimeBoundInquiry(loc, builder, array)
          : builder.createBox(loc, array);
  aiir::Value kind = isStaticallyAbsent(args, kindPos)
                         ? builder.createIntegerConstant(
                               loc, builder.getI32Type(),
                               builder.getKindMap().defaultIntegerKind())
                         : fir::getBase(args[kindPos]);
  genRtCall(builder, loc, resultStorage, arrayBox, kind);
  if (hasAssumedRank) {
    // Cast to fir.ref<array<?xik>> since the result extent is not a compile
    // time constant.
    aiir::Type baseType =
        fir::ReferenceType::get(builder.getVarLenSeqTy(resultElementType));
    aiir::Value resultBase =
        builder.createConvert(loc, baseType, resultStorage);
    aiir::Value rankValue =
        fir::BoxRankOp::create(builder, loc, builder.getIndexType(), arrayBox);
    return fir::ArrayBoxValue{resultBase, {rankValue}};
  }
  // Result extent is a compile time constant in the other cases.
  aiir::Value rankValue =
      builder.createIntegerConstant(loc, builder.getIndexType(), rank);
  return fir::ArrayBoxValue{resultStorage, {rankValue}};
}

// SHAPE
fir::ExtendedValue
IntrinsicLibrary::genShape(aiir::Type resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() >= 1);
  const fir::ExtendedValue &array = args[0];
  if (array.hasAssumedRank())
    return genBoundInquiry(builder, loc, resultType, args,
                           /*kindPos=*/1, fir::runtime::genShape,
                           /*needAccurateLowerBound=*/false);
  int rank = array.rank();
  aiir::Type indexType = builder.getIndexType();
  aiir::Type extentType = fir::unwrapSequenceType(resultType);
  aiir::Type seqType = fir::SequenceType::get(
      {static_cast<fir::SequenceType::Extent>(rank)}, extentType);
  aiir::Value shapeArray = builder.createTemporary(loc, seqType);
  aiir::Type shapeAddrType = builder.getRefType(extentType);
  for (int dim = 0; dim < rank; ++dim) {
    aiir::Value extent = fir::factory::readExtent(builder, loc, array, dim);
    extent = builder.createConvert(loc, extentType, extent);
    auto index = builder.createIntegerConstant(loc, indexType, dim);
    auto shapeAddr = fir::CoordinateOp::create(builder, loc, shapeAddrType,
                                               shapeArray, index);
    fir::StoreOp::create(builder, loc, extent, shapeAddr);
  }
  aiir::Value shapeArrayExtent =
      builder.createIntegerConstant(loc, indexType, rank);
  llvm::SmallVector<aiir::Value> extents{shapeArrayExtent};
  return fir::ArrayBoxValue{shapeArray, extents};
}

// SHIFTL, SHIFTR
template <typename Shift>
aiir::Value IntrinsicLibrary::genShift(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);

  // If SHIFT < 0 or SHIFT >= BIT_SIZE(I), return 0. This is not required by
  // the standard. However, several other compilers behave this way, so try and
  // maintain compatibility with them to an extent.

  unsigned bits = resultType.getIntOrFloatBitWidth();
  aiir::Type signlessType =
      aiir::IntegerType::get(builder.getContext(), bits,
                             aiir::IntegerType::SignednessSemantics::Signless);
  aiir::Value bitSize = builder.createIntegerConstant(loc, signlessType, bits);
  aiir::Value zero = builder.createIntegerConstant(loc, signlessType, 0);
  aiir::Value shift = builder.createConvert(loc, signlessType, args[1]);

  aiir::Value tooSmall = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::slt, shift, zero);
  aiir::Value tooLarge = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::sge, shift, bitSize);
  aiir::Value outOfBounds =
      aiir::arith::OrIOp::create(builder, loc, tooSmall, tooLarge);
  aiir::Value word = args[0];
  if (word.getType().isUnsignedInteger())
    word = builder.createConvert(loc, signlessType, word);
  aiir::Value shifted = Shift::create(builder, loc, word, shift);
  aiir::Value result =
      aiir::arith::SelectOp::create(builder, loc, outOfBounds, zero, shifted);
  if (resultType.isUnsignedInteger())
    return builder.createConvert(loc, resultType, result);
  return result;
}

// SHIFTA
aiir::Value IntrinsicLibrary::genShiftA(aiir::Type resultType,
                                        llvm::ArrayRef<aiir::Value> args) {
  unsigned bits = resultType.getIntOrFloatBitWidth();
  aiir::Type signlessType =
      aiir::IntegerType::get(builder.getContext(), bits,
                             aiir::IntegerType::SignednessSemantics::Signless);
  aiir::Value bitSize = builder.createIntegerConstant(loc, signlessType, bits);
  aiir::Value shift = builder.createConvert(loc, signlessType, args[1]);
  aiir::Value shiftGeBitSize = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::uge, shift, bitSize);

  // Lowering of aiir::arith::ShRSIOp is using `ashr`. `ashr` is undefined when
  // the shift amount is equal to the element size.
  // So if SHIFT is equal to the bit width then it is handled as a special case.
  // When negative or larger than the bit width, handle it like other
  // Fortran compiler do (treat it as bit width, minus 1).
  aiir::Value zero = builder.createIntegerConstant(loc, signlessType, 0);
  aiir::Value minusOne = builder.createMinusOneInteger(loc, signlessType);
  aiir::Value word = args[0];
  if (word.getType().isUnsignedInteger())
    word = builder.createConvert(loc, signlessType, word);
  aiir::Value valueIsNeg = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::slt, word, zero);
  aiir::Value specialRes =
      aiir::arith::SelectOp::create(builder, loc, valueIsNeg, minusOne, zero);
  aiir::Value shifted = aiir::arith::ShRSIOp::create(builder, loc, word, shift);
  aiir::Value result = aiir::arith::SelectOp::create(
      builder, loc, shiftGeBitSize, specialRes, shifted);
  if (resultType.isUnsignedInteger())
    return builder.createConvert(loc, resultType, result);
  return result;
}

void IntrinsicLibrary::genShowDescriptor(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1 && "expected single argument for show_descriptor");
  const aiir::Value arg = fir::getBase(args[0]);

  // Use consistent !fir.ref<!fir.box<none>> argument type
  auto targetType = fir::BoxType::get(builder.getNoneType());
  auto targetRefType = fir::ReferenceType::get(targetType);

  aiir::Value descrAddr = nullptr;
  if (fir::isBoxAddress(arg.getType())) {
    // If it's already a reference to a box, convert it to correct type and
    // pass it directly
    descrAddr = builder.createConvert(loc, targetRefType, arg);
  } else {
    // At this point, arg is either SSA descriptor or a non-descriptor entity.
    // If necessary, wrap non-descriptor entity in a descriptor.
    aiir::Value descriptor = nullptr;
    if (fir::isa_box_type(arg.getType())) {
      descriptor = arg;
    } else if (fir::isa_ref_type(arg.getType())) {
      // Note: here use full extended value args[0]
      descriptor = builder.createBox(loc, args[0]);
    } else {
      // arg is a value (e.g. constant), spill it to a temporary
      // because createBox expects a memory reference.
      aiir::Value temp = builder.createTemporary(loc, arg.getType());
      builder.createStoreWithConvert(loc, arg, temp);

      // Note: here use full extended value args[0]
      descriptor = builder.createBox(loc, fir::substBase(args[0], temp));
    }

    // Spill it to the stack
    descrAddr = builder.createTemporary(loc, targetType);
    builder.createStoreWithConvert(loc, descriptor, descrAddr);
  }

  fir::runtime::genShowDescriptor(builder, loc, descrAddr);
}

// SIGNAL
void IntrinsicLibrary::genSignalSubroutine(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2 || args.size() == 3);
  aiir::Value number = fir::getBase(args[0]);
  aiir::Value handler = fir::getBase(args[1]);
  aiir::Value status;
  if (args.size() == 3)
    status = fir::getBase(args[2]);
  fir::runtime::genSignal(builder, loc, number, handler, status);
}

// SIGN
aiir::Value IntrinsicLibrary::genSign(aiir::Type resultType,
                                      llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 2);
  if (aiir::isa<aiir::IntegerType>(resultType)) {
    aiir::Value abs = genAbs(resultType, {args[0]});
    aiir::Value zero = builder.createIntegerConstant(loc, resultType, 0);
    auto neg = aiir::arith::SubIOp::create(builder, loc, zero, abs);
    auto cmp = aiir::arith::CmpIOp::create(
        builder, loc, aiir::arith::CmpIPredicate::slt, args[1], zero);
    return aiir::arith::SelectOp::create(builder, loc, cmp, neg, abs);
  }
  return genRuntimeCall("sign", resultType, args);
}

// SIND
aiir::Value IntrinsicLibrary::genSind(aiir::Type resultType,
                                      llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);
  aiir::AIIRContext *context = builder.getContext();
  aiir::FunctionType ftype =
      aiir::FunctionType::get(context, {resultType}, {args[0].getType()});
  const llvm::fltSemantics &fltSem =
      llvm::cast<aiir::FloatType>(resultType).getFloatSemantics();
  llvm::APFloat pi = llvm::APFloat(fltSem, llvm::numbers::pis);
  aiir::Value factor = builder.createRealConstant(
      loc, resultType, pi / llvm::APFloat(fltSem, "180.0"));
  aiir::Value arg = aiir::arith::MulFOp::create(builder, loc, args[0], factor);
  return getRuntimeCallGenerator("sin", ftype)(builder, loc, {arg});
}

// SINPI
aiir::Value IntrinsicLibrary::genSinpi(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);
  aiir::AIIRContext *context = builder.getContext();
  aiir::FunctionType ftype =
      aiir::FunctionType::get(context, {resultType}, {args[0].getType()});
  llvm::APFloat pi =
      llvm::APFloat(llvm::cast<aiir::FloatType>(resultType).getFloatSemantics(),
                    llvm::numbers::pis);
  aiir::Value factor = builder.createRealConstant(loc, resultType, pi);
  aiir::Value arg = aiir::arith::MulFOp::create(builder, loc, args[0], factor);
  return getRuntimeCallGenerator("sin", ftype)(builder, loc, {arg});
}

// SIZE
fir::ExtendedValue
IntrinsicLibrary::genSize(aiir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  // Note that the value of the KIND argument is already reflected in the
  // resultType
  assert(args.size() == 3);

  // Get the ARRAY argument
  aiir::Value array = builder.createBox(loc, args[0]);

  // The front-end rewrites SIZE without the DIM argument to
  // an array of SIZE with DIM in most cases, but it may not be
  // possible in some cases like when in SIZE(function_call()).
  if (isStaticallyAbsent(args, 1))
    return builder.createConvert(loc, resultType,
                                 fir::runtime::genSize(builder, loc, array));

  // Get the DIM argument.
  aiir::Value dim = fir::getBase(args[1]);
  if (!args[0].hasAssumedRank())
    if (std::optional<std::int64_t> cstDim = fir::getIntIfConstant(dim)) {
      // If both DIM and the rank are compile time constants, skip the runtime
      // call.
      return builder.createConvert(
          loc, resultType,
          fir::factory::readExtent(builder, loc, fir::BoxValue{array},
                                   cstDim.value() - 1));
    }
  if (!fir::isa_ref_type(dim.getType()))
    return builder.createConvert(
        loc, resultType, fir::runtime::genSizeDim(builder, loc, array, dim));

  aiir::Value isDynamicallyAbsent = builder.genIsNullAddr(loc, dim);
  return builder
      .genIfOp(loc, {resultType}, isDynamicallyAbsent,
               /*withElseRegion=*/true)
      .genThen([&]() {
        aiir::Value size = builder.createConvert(
            loc, resultType, fir::runtime::genSize(builder, loc, array));
        fir::ResultOp::create(builder, loc, size);
      })
      .genElse([&]() {
        aiir::Value dimValue = fir::LoadOp::create(builder, loc, dim);
        aiir::Value size = builder.createConvert(
            loc, resultType,
            fir::runtime::genSizeDim(builder, loc, array, dimValue));
        fir::ResultOp::create(builder, loc, size);
      })
      .getResults()[0];
}

// SIZEOF
fir::ExtendedValue
IntrinsicLibrary::genSizeOf(aiir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  aiir::Value box = fir::getBase(args[0]);
  aiir::Value eleSize =
      fir::BoxEleSizeOp::create(builder, loc, resultType, box);
  if (!fir::isArray(args[0]))
    return eleSize;
  aiir::Value arraySize = builder.createConvert(
      loc, resultType, fir::runtime::genSize(builder, loc, box));
  return aiir::arith::MulIOp::create(builder, loc, eleSize, arraySize);
}

// TAND
aiir::Value IntrinsicLibrary::genTand(aiir::Type resultType,
                                      llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);
  aiir::AIIRContext *context = builder.getContext();
  aiir::FunctionType ftype =
      aiir::FunctionType::get(context, {resultType}, {args[0].getType()});
  const llvm::fltSemantics &fltSem =
      llvm::cast<aiir::FloatType>(resultType).getFloatSemantics();
  llvm::APFloat pi = llvm::APFloat(fltSem, llvm::numbers::pis);
  aiir::Value factor = builder.createRealConstant(
      loc, resultType, pi / llvm::APFloat(fltSem, "180.0"));
  aiir::Value arg = aiir::arith::MulFOp::create(builder, loc, args[0], factor);
  return getRuntimeCallGenerator("tan", ftype)(builder, loc, {arg});
}

// TANPI
aiir::Value IntrinsicLibrary::genTanpi(aiir::Type resultType,
                                       llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);
  aiir::AIIRContext *context = builder.getContext();
  aiir::FunctionType ftype =
      aiir::FunctionType::get(context, {resultType}, {args[0].getType()});
  llvm::APFloat pi =
      llvm::APFloat(llvm::cast<aiir::FloatType>(resultType).getFloatSemantics(),
                    llvm::numbers::pis);
  aiir::Value factor = builder.createRealConstant(loc, resultType, pi);
  aiir::Value arg = aiir::arith::MulFOp::create(builder, loc, args[0], factor);
  return getRuntimeCallGenerator("tan", ftype)(builder, loc, {arg});
}

// TEAM_NUMBER
fir::ExtendedValue
IntrinsicLibrary::genTeamNumber(aiir::Type resultType,
                                llvm::ArrayRef<fir::ExtendedValue> args) {
  converter->checkCoarrayEnabled();
  assert(args.size() == 1);

  aiir::Value res = mif::TeamNumberOp::create(builder, loc,
                                              /*team*/ fir::getBase(args[0]));
  return builder.createConvert(loc, resultType, res);
}

// THIS_IMAGE
fir::ExtendedValue
IntrinsicLibrary::genThisImage(aiir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args) {
  converter->checkCoarrayEnabled();
  assert(args.size() >= 1 && args.size() <= 3);
  const bool coarrayIsAbsent = args.size() == 1;
  aiir::Value team = fir::getBase(args[args.size() - 1]);

  if (!coarrayIsAbsent)
    TODO(loc, "this_image with coarray argument.");
  aiir::Value res = mif::ThisImageOp::create(builder, loc, team);
  return builder.createConvert(loc, resultType, res);
}

// TRAILZ
aiir::Value IntrinsicLibrary::genTrailz(aiir::Type resultType,
                                        llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);

  aiir::Value result =
      aiir::math::CountTrailingZerosOp::create(builder, loc, args);

  return builder.createConvert(loc, resultType, result);
}

static bool hasDefaultLowerBound(const fir::ExtendedValue &exv) {
  return exv.match(
      [](const fir::ArrayBoxValue &arr) { return arr.getLBounds().empty(); },
      [](const fir::CharArrayBoxValue &arr) {
        return arr.getLBounds().empty();
      },
      [](const fir::BoxValue &arr) { return arr.getLBounds().empty(); },
      [](const auto &) { return false; });
}

/// Compute the lower bound in dimension \p dim (zero based) of \p array
/// taking care of returning one when the related extent is zero.
static aiir::Value computeLBOUND(fir::FirOpBuilder &builder, aiir::Location loc,
                                 const fir::ExtendedValue &array, unsigned dim,
                                 aiir::Value zero, aiir::Value one) {
  assert(dim < array.rank() && "invalid dimension");
  if (hasDefaultLowerBound(array))
    return one;
  aiir::Value lb = fir::factory::readLowerBound(builder, loc, array, dim, one);
  aiir::Value extent = fir::factory::readExtent(builder, loc, array, dim);
  zero = builder.createConvert(loc, extent.getType(), zero);
  // Note: for assumed size, the extent is -1, and the lower bound should
  // be returned. It is important to test extent == 0 and not extent > 0.
  auto dimIsEmpty = aiir::arith::CmpIOp::create(
      builder, loc, aiir::arith::CmpIPredicate::eq, extent, zero);
  one = builder.createConvert(loc, lb.getType(), one);
  return aiir::arith::SelectOp::create(builder, loc, dimIsEmpty, one, lb);
}

// LBOUND
fir::ExtendedValue
IntrinsicLibrary::genLbound(aiir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2 || args.size() == 3);
  const fir::ExtendedValue &array = args[0];
  // Semantics builds signatures for LBOUND calls as either
  // LBOUND(array, dim, [kind]) or LBOUND(array, [kind]).
  const bool dimIsAbsent = args.size() == 2 || isStaticallyAbsent(args, 1);
  if (array.hasAssumedRank() && dimIsAbsent) {
    int kindPos = args.size() == 2 ? 1 : 2;
    return genBoundInquiry(builder, loc, resultType, args, kindPos,
                           fir::runtime::genLbound,
                           /*needAccurateLowerBound=*/true);
  }

  aiir::Type indexType = builder.getIndexType();

  if (dimIsAbsent) {
    // DIM is absent and the rank of array is a compile time constant.
    aiir::Type lbType = fir::unwrapSequenceType(resultType);
    unsigned rank = array.rank();
    aiir::Type lbArrayType = fir::SequenceType::get(
        {static_cast<fir::SequenceType::Extent>(array.rank())}, lbType);
    aiir::Value lbArray = builder.createTemporary(loc, lbArrayType);
    aiir::Type lbAddrType = builder.getRefType(lbType);
    aiir::Value one = builder.createIntegerConstant(loc, lbType, 1);
    aiir::Value zero = builder.createIntegerConstant(loc, indexType, 0);
    for (unsigned dim = 0; dim < rank; ++dim) {
      aiir::Value lb = computeLBOUND(builder, loc, array, dim, zero, one);
      lb = builder.createConvert(loc, lbType, lb);
      auto index = builder.createIntegerConstant(loc, indexType, dim);
      auto lbAddr =
          fir::CoordinateOp::create(builder, loc, lbAddrType, lbArray, index);
      fir::StoreOp::create(builder, loc, lb, lbAddr);
    }
    aiir::Value lbArrayExtent =
        builder.createIntegerConstant(loc, indexType, rank);
    llvm::SmallVector<aiir::Value> extents{lbArrayExtent};
    return fir::ArrayBoxValue{lbArray, extents};
  }
  // DIM is present.
  aiir::Value dim = fir::getBase(args[1]);

  // If it is a compile time constant and the rank is known, skip the runtime
  // call.
  if (!array.hasAssumedRank())
    if (std::optional<std::int64_t> cstDim = fir::getIntIfConstant(dim)) {
      aiir::Value one = builder.createIntegerConstant(loc, resultType, 1);
      aiir::Value zero = builder.createIntegerConstant(loc, indexType, 0);
      aiir::Value lb =
          computeLBOUND(builder, loc, array, *cstDim - 1, zero, one);
      return builder.createConvert(loc, resultType, lb);
    }

  fir::ExtendedValue box = createBoxForRuntimeBoundInquiry(loc, builder, array);
  return builder.createConvert(
      loc, resultType,
      fir::runtime::genLboundDim(builder, loc, fir::getBase(box), dim));
}

// UBOUND
fir::ExtendedValue
IntrinsicLibrary::genUbound(aiir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3 || args.size() == 2);
  const bool dimIsAbsent = args.size() == 2 || isStaticallyAbsent(args, 1);
  if (!dimIsAbsent) {
    // Handle calls to UBOUND with the DIM argument, which return a scalar
    aiir::Value extent = fir::getBase(genSize(resultType, args));
    aiir::Value lbound = fir::getBase(genLbound(resultType, args));

    aiir::Value one = builder.createIntegerConstant(loc, resultType, 1);
    aiir::Value ubound = aiir::arith::SubIOp::create(builder, loc, lbound, one);
    return aiir::arith::AddIOp::create(builder, loc, ubound, extent);
  }
  // Handle calls to UBOUND without the DIM argument, which return an array
  int kindPos = args.size() == 2 ? 1 : 2;
  return genBoundInquiry(builder, loc, resultType, args, kindPos,
                         fir::runtime::genUbound,
                         /*needAccurateLowerBound=*/true);
}

// SPACING
aiir::Value IntrinsicLibrary::genSpacing(aiir::Type resultType,
                                         llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 1);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genSpacing(builder, loc, fir::getBase(args[0])));
}

// SPREAD
fir::ExtendedValue
IntrinsicLibrary::genSpread(aiir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 3);

  // Handle source argument
  aiir::Value source = builder.createBox(loc, args[0]);
  fir::BoxValue sourceTmp = source;
  unsigned sourceRank = sourceTmp.rank();

  // Handle Dim argument
  aiir::Value dim = fir::getBase(args[1]);

  // Handle ncopies argument
  aiir::Value ncopies = fir::getBase(args[2]);

  // Generate result descriptor
  aiir::Type resultArrayType =
      builder.getVarLenSeqTy(resultType, sourceRank + 1);
  fir::MutableBoxValue resultMutableBox = fir::factory::createTempMutableBox(
      builder, loc, resultArrayType, {},
      fir::isPolymorphicType(source.getType()) ? source : aiir::Value{});
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genSpread(builder, loc, resultIrBox, source, dim, ncopies);

  return readAndAddCleanUp(resultMutableBox, resultType, "SPREAD");
}

// STORAGE_SIZE
fir::ExtendedValue
IntrinsicLibrary::genStorageSize(aiir::Type resultType,
                                 llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2 || args.size() == 1);
  aiir::Value box = fir::getBase(args[0]);
  aiir::Type boxTy = box.getType();
  aiir::Type kindTy = builder.getDefaultIntegerType();
  bool needRuntimeCheck = false;
  std::string errorMsg;

  if (fir::isUnlimitedPolymorphicType(boxTy) &&
      (fir::isAllocatableType(boxTy) || fir::isPointerType(boxTy))) {
    needRuntimeCheck = true;
    errorMsg =
        fir::isPointerType(boxTy)
            ? "unlimited polymorphic disassociated POINTER in STORAGE_SIZE"
            : "unlimited polymorphic unallocated ALLOCATABLE in STORAGE_SIZE";
  }
  const fir::MutableBoxValue *mutBox = args[0].getBoxOf<fir::MutableBoxValue>();
  if (needRuntimeCheck && mutBox) {
    aiir::Value isNotAllocOrAssoc =
        fir::factory::genIsNotAllocatedOrAssociatedTest(builder, loc, *mutBox);
    builder.genIfThen(loc, isNotAllocOrAssoc)
        .genThen([&]() {
          fir::runtime::genReportFatalUserError(builder, loc, errorMsg);
        })
        .end();
  }

  // Handle optional kind argument
  bool absentKind = isStaticallyAbsent(args, 1);
  if (!absentKind) {
    aiir::Operation *defKind = fir::getBase(args[1]).getDefiningOp();
    assert(aiir::isa<aiir::arith::ConstantOp>(*defKind) &&
           "kind not a constant");
    auto constOp = aiir::dyn_cast<aiir::arith::ConstantOp>(*defKind);
    kindTy = builder.getIntegerType(
        builder.getKindMap().getIntegerBitsize(fir::toInt(constOp)));
  }

  box = builder.createBox(loc, args[0],
                          /*isPolymorphic=*/args[0].isPolymorphic());
  aiir::Value eleSize = fir::BoxEleSizeOp::create(builder, loc, kindTy, box);
  aiir::Value c8 = builder.createIntegerConstant(loc, kindTy, 8);
  return aiir::arith::MulIOp::create(builder, loc, eleSize, c8);
}

// SUM
fir::ExtendedValue
IntrinsicLibrary::genSum(aiir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {
  return genReduction(fir::runtime::genSum, fir::runtime::genSumDim, "SUM",
                      resultType, args);
}

// SYSTEM
fir::ExtendedValue
IntrinsicLibrary::genSystem(std::optional<aiir::Type> resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert((!resultType && (args.size() == 2)) ||
         (resultType && (args.size() == 1)));
  aiir::Value command = fir::getBase(args[0]);
  assert(command && "expected COMMAND parameter");

  fir::ExtendedValue exitstat;
  if (resultType) {
    aiir::Value tmp = builder.createTemporary(loc, *resultType);
    exitstat = builder.createBox(loc, tmp);
  } else {
    exitstat = args[1];
  }

  aiir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());

  aiir::Value waitBool = builder.createBool(loc, true);
  aiir::Value exitstatBox =
      isStaticallyPresent(exitstat)
          ? fir::getBase(exitstat)
          : fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();

  // Create a dummmy cmdstat to prevent EXECUTE_COMMAND_LINE terminate itself
  // when cmdstat is assigned with a non-zero value but not present
  aiir::Value tempValue =
      builder.createIntegerConstant(loc, builder.getI16Type(), 0);
  aiir::Value temp = builder.createTemporary(loc, builder.getI16Type());
  fir::StoreOp::create(builder, loc, tempValue, temp);
  aiir::Value cmdstatBox = builder.createBox(loc, temp);

  aiir::Value cmdmsgBox =
      fir::AbsentOp::create(builder, loc, boxNoneTy).getResult();

  fir::runtime::genExecuteCommandLine(builder, loc, command, waitBool,
                                      exitstatBox, cmdstatBox, cmdmsgBox);

  if (resultType) {
    aiir::Value exitstatAddr =
        fir::BoxAddrOp::create(builder, loc, exitstatBox);
    return fir::LoadOp::create(builder, loc, fir::getBase(exitstatAddr));
  }
  return {};
}

// SYSTEM_CLOCK
void IntrinsicLibrary::genSystemClock(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  fir::runtime::genSystemClock(builder, loc, fir::getBase(args[0]),
                               fir::getBase(args[1]), fir::getBase(args[2]));
}

// SLEEP
void IntrinsicLibrary::genSleep(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1 && "SLEEP has one compulsory argument");
  fir::runtime::genSleep(builder, loc, fir::getBase(args[0]));
}

// SPLIT
void IntrinsicLibrary::genSplit(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);

  aiir::Value stringBase = fir::getBase(args[0]);
  aiir::Value stringLen = fir::getLen(args[0]);
  aiir::Value setBase = fir::getBase(args[1]);
  aiir::Value setLen = fir::getLen(args[1]);
  aiir::Value posAddr = fir::getBase(args[2]);

  fir::KindTy kind =
      fir::factory::CharacterExprHelper{builder, loc}.getCharacterKind(
          stringBase.getType());

  // BACK is optional and defaults to .FALSE. when absent.
  aiir::Value back =
      isStaticallyAbsent(args[3])
          ? builder.createIntegerConstant(loc, builder.getI1Type(), 0)
          : fir::getBase(args[3]);

  aiir::Type posRefTy = fir::dyn_cast_ptrEleTy(posAddr.getType());
  aiir::Value posValue = fir::LoadOp::create(builder, loc, posRefTy, posAddr);
  aiir::Type indexTy = builder.getIndexType();
  aiir::Value posIndex = builder.createConvert(loc, indexTy, posValue);

  aiir::Value newPos =
      fir::runtime::genSplit(builder, loc, kind, stringBase, stringLen, setBase,
                             setLen, posIndex, back);

  aiir::Value newPosConverted = builder.createConvert(loc, posRefTy, newPos);
  fir::StoreOp::create(builder, loc, newPosConverted, posAddr);
}

// TOKENIZE
void IntrinsicLibrary::genTokenize(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4 && "TOKENIZE requires 3 or 4 arguments");

  const fir::ExtendedValue &string = args[0];
  const fir::ExtendedValue &set = args[1];

  // Distinguish forms by the element type of the third argument.  For form 1,
  // TOKENS is CHARACTER.  For form 2, FIRST is INTEGER.
  aiir::Type thirdArgEleTy = fir::getElementTypeOf(args[2]);
  bool isForm1 = fir::isa_char(thirdArgEleTy);
  [[maybe_unused]] bool isForm2 = fir::isa_integer(thirdArgEleTy);
  assert((isForm1 || isForm2) &&
         "TOKENIZE third argument must be CHARACTER or INTEGER");

  aiir::Value stringBox = builder.createBox(loc, string);
  aiir::Value setBox = builder.createBox(loc, set);

  aiir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
  aiir::Type boxNoneRefTy = fir::ReferenceType::get(boxNoneTy);

  // A lambda to return the address of the descriptor storage to pass to the
  // runtime. For MutableBoxValue, this also handles any required syncing
  // before/after the runtime call.
  auto getBoxStorageAddr =
      [&](const fir::ExtendedValue &exv, llvm::StringRef what,
          const fir::MutableBoxValue **mutableBoxOut) -> aiir::Value {
    if (const auto *mb = exv.getBoxOf<fir::MutableBoxValue>()) {
      if (mutableBoxOut)
        *mutableBoxOut = mb;
      aiir::Value addr = fir::factory::getMutableIRBox(builder, loc, *mb);
      return builder.createConvert(loc, boxNoneRefTy, addr);
    }
    if (const auto *bv = exv.getBoxOf<fir::BoxValue>()) {
      aiir::Value addr = bv->getAddr();
      if (auto boxTy = fir::dyn_cast_ptrEleTy(addr.getType())) {
        if (aiir::isa<fir::BaseBoxType>(boxTy))
          return builder.createConvert(loc, boxNoneRefTy, addr);
      }
      fir::emitFatalError(loc, llvm::Twine("TOKENIZE: ") + what +
                                   " must be a descriptor address");
    }
    fir::emitFatalError(loc, llvm::Twine("TOKENIZE: ") + what +
                                 " not lowered as a boxed entity");
  };

  if (isForm1) {
    // Form 1: TOKENIZE(STRING, SET, TOKENS [, SEPARATOR])
    const fir::ExtendedValue &tokens = args[2];
    const fir::MutableBoxValue *tokensMutableBox{nullptr};
    aiir::Value tokensBoxAddr =
        getBoxStorageAddr(tokens, "TOKENS", &tokensMutableBox);

    // Handle optional SEPARATOR argument
    aiir::Value separatorBoxAddr;
    const fir::MutableBoxValue *separatorMutableBox{nullptr};
    if (!isStaticallyAbsent(args[3])) {
      const fir::ExtendedValue &separator = args[3];
      separatorBoxAddr =
          getBoxStorageAddr(separator, "SEPARATOR", &separatorMutableBox);
    } else {
      separatorBoxAddr = builder.createNullConstant(loc, boxNoneRefTy);
    }

    // Call the Form 1 runtime function
    fir::runtime::genTokenize(builder, loc, tokensBoxAddr, separatorBoxAddr,
                              stringBox, setBox);

    if (tokensMutableBox)
      fir::factory::syncMutableBoxFromIRBox(builder, loc, *tokensMutableBox);
    if (separatorMutableBox)
      fir::factory::syncMutableBoxFromIRBox(builder, loc, *separatorMutableBox);

  } else {
    // Form 2: TOKENIZE(STRING, SET, FIRST, LAST)
    const fir::ExtendedValue &first = args[2];
    const fir::ExtendedValue &last = args[3];

    const fir::MutableBoxValue *firstMutableBox{nullptr};
    const fir::MutableBoxValue *lastMutableBox{nullptr};
    aiir::Value firstBoxAddr =
        getBoxStorageAddr(first, "FIRST", &firstMutableBox);
    aiir::Value lastBoxAddr = getBoxStorageAddr(last, "LAST", &lastMutableBox);

    // Call the Form 2 runtime function
    fir::runtime::genTokenizePositions(builder, loc, firstBoxAddr, lastBoxAddr,
                                       stringBox, setBox);

    if (firstMutableBox)
      fir::factory::syncMutableBoxFromIRBox(builder, loc, *firstMutableBox);
    if (lastMutableBox)
      fir::factory::syncMutableBoxFromIRBox(builder, loc, *lastMutableBox);
  }
}

// TRANSFER
fir::ExtendedValue
IntrinsicLibrary::genTransfer(aiir::Type resultType,
                              llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() >= 2); // args.size() == 2 when size argument is omitted.

  // Handle source argument
  aiir::Value source = builder.createBox(loc, args[0]);

  // Handle mold argument
  aiir::Value mold = builder.createBox(loc, args[1]);
  fir::BoxValue moldTmp = mold;
  unsigned moldRank = moldTmp.rank();

  bool absentSize = (args.size() == 2);

  // Create mutable fir.box to be passed to the runtime for the result.
  aiir::Type type = (moldRank == 0 && absentSize)
                        ? resultType
                        : builder.getVarLenSeqTy(resultType, 1);
  fir::MutableBoxValue resultMutableBox = fir::factory::createTempMutableBox(
      builder, loc, type, {},
      fir::isPolymorphicType(mold.getType()) ? mold : aiir::Value{});

  if (moldRank == 0 && absentSize) {
    // This result is a scalar in this case.
    aiir::Value resultIrBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    fir::runtime::genTransfer(builder, loc, resultIrBox, source, mold);
  } else {
    // The result is a rank one array in this case.
    aiir::Value resultIrBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    if (absentSize) {
      fir::runtime::genTransfer(builder, loc, resultIrBox, source, mold);
    } else {
      aiir::Value sizeArg = fir::getBase(args[2]);
      fir::runtime::genTransferSize(builder, loc, resultIrBox, source, mold,
                                    sizeArg);
    }
  }
  return readAndAddCleanUp(resultMutableBox, resultType, "TRANSFER");
}

// TRANSPOSE
fir::ExtendedValue
IntrinsicLibrary::genTranspose(aiir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 1);

  // Handle source argument
  aiir::Value source = builder.createBox(loc, args[0]);

  // Create mutable fir.box to be passed to the runtime for the result.
  aiir::Type resultArrayType = builder.getVarLenSeqTy(resultType, 2);
  fir::MutableBoxValue resultMutableBox = fir::factory::createTempMutableBox(
      builder, loc, resultArrayType, {},
      fir::isPolymorphicType(source.getType()) ? source : aiir::Value{});
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genTranspose(builder, loc, resultIrBox, source);
  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  return readAndAddCleanUp(resultMutableBox, resultType, "TRANSPOSE");
}

// TIME
aiir::Value IntrinsicLibrary::genTime(aiir::Type resultType,
                                      llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() == 0);
  return builder.createConvert(loc, resultType,
                               fir::runtime::genTime(builder, loc));
}

// TRIM
fir::ExtendedValue
IntrinsicLibrary::genTrim(aiir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  aiir::Value string = builder.createBox(loc, args[0]);
  // Create mutable fir.box to be passed to the runtime for the result.
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genTrim(builder, loc, resultIrBox, string);
  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  return readAndAddCleanUp(resultMutableBox, resultType, "TRIM");
}

// Compare two FIR values and return boolean result as i1.
template <bool isMax>
static aiir::Value genExtremumResult(aiir::Location loc,
                                     fir::FirOpBuilder &builder,
                                     aiir::Value left, aiir::Value right) {
  aiir::Type type = left.getType();
  if (fir::isa_real(type)) {
    switch (builder.getFPMaxminBehavior()) {
    case Fortran::common::FPMaxminBehavior::Portable:
      // If the left is NaN, return the right whatever it is.
      // Signed zeros are equal, so max/min(zero, zero) always
      // returns the second 'zero'.
      if (aiir::arith::bitEnumContainsAll(
              builder.getFastMathFlags(),
              aiir::arith::FastMathFlags::nnan |
                  aiir::arith::FastMathFlags::nsz)) {
        // If there are no NaNs and signed zeros, we can use a shorter
        // arith.max/minnumf representation.
        if constexpr (isMax)
          return aiir::arith::MaxNumFOp::create(builder, loc, left, right);
        else
          return aiir::arith::MinNumFOp::create(builder, loc, left, right);
      }
      [[fallthrough]];
    case Fortran::common::FPMaxminBehavior::Legacy: {
      static constexpr aiir::arith::CmpFPredicate pred =
          isMax ? aiir::arith::CmpFPredicate::OGT
                : aiir::arith::CmpFPredicate::OLT;
      aiir::Value cmp =
          aiir::arith::CmpFOp::create(builder, loc, pred, left, right);
      return aiir::arith::SelectOp::create(builder, loc, cmp, left, right);
    }
    case Fortran::common::FPMaxminBehavior::Extremum:
      if constexpr (isMax)
        return aiir::arith::MaximumFOp::create(builder, loc, left, right);
      else
        return aiir::arith::MinimumFOp::create(builder, loc, left, right);
    case Fortran::common::FPMaxminBehavior::ExtremeNum:
      if constexpr (isMax)
        return aiir::arith::MaxNumFOp::create(builder, loc, left, right);
      else
        return aiir::arith::MinNumFOp::create(builder, loc, left, right);
    }

    llvm_unreachable("unsupported FPMaxminBehavior");
  } else if (fir::isa_integer(type)) {
    // It is probably okay to use signed index.maxs/mins, but
    // maybe the caller needs to specify signedness.
    // There are currently no callers that pass values of index
    // type, so just emit a TODO.
    if (aiir::isa<aiir::IndexType>(type))
      TODO(loc, "extremum for index type");

    if (type.isUnsignedInteger()) {
      // arith.maxui/minui operands must have singless type.
      aiir::Type signlessType = aiir::IntegerType::get(
          builder.getContext(), type.getIntOrFloatBitWidth(),
          aiir::IntegerType::SignednessSemantics::Signless);
      left = builder.createConvert(loc, signlessType, left);
      right = builder.createConvert(loc, signlessType, right);

      aiir::Value result;
      if constexpr (isMax)
        result = aiir::arith::MaxUIOp::create(builder, loc, left, right);
      else
        result = aiir::arith::MinUIOp::create(builder, loc, left, right);

      return builder.createConvert(loc, type, result);
    } else {
      if constexpr (isMax)
        return aiir::arith::MaxSIOp::create(builder, loc, left, right);
      else
        return aiir::arith::MinSIOp::create(builder, loc, left, right);
    }
  } else if (fir::isa_char(type) || fir::isa_char(fir::unwrapRefType(type))) {
    // TODO: ! character min and max is tricky because the result
    // length is the length of the longest argument!
    // So we may need a temp.
    TODO(loc, "intrinsic: min and max for CHARACTER");
  }
  llvm_unreachable("unsupported extremum");
}

// UNLINK
fir::ExtendedValue
IntrinsicLibrary::genUnlink(std::optional<aiir::Type> resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert((resultType.has_value() && args.size() == 1) ||
         (!resultType.has_value() && args.size() >= 1 && args.size() <= 2));

  aiir::Value path = fir::getBase(args[0]);
  aiir::Value pathLength = fir::getLen(args[0]);
  aiir::Value statusValue =
      fir::runtime::genUnlink(builder, loc, path, pathLength);

  if (resultType.has_value()) {
    // Function form, return status.
    return builder.createConvert(loc, *resultType, statusValue);
  }

  // Subroutine form, store status and return none.
  const fir::ExtendedValue &status = args[1];
  if (!isStaticallyAbsent(status)) {
    aiir::Value statusAddr = fir::getBase(status);
    aiir::Value statusIsPresentAtRuntime =
        builder.genIsNotNullAddr(loc, statusAddr);
    builder.genIfThen(loc, statusIsPresentAtRuntime)
        .genThen([&]() {
          builder.createStoreWithConvert(loc, statusValue, statusAddr);
        })
        .end();
  }

  return {};
}

// UNPACK
fir::ExtendedValue
IntrinsicLibrary::genUnpack(aiir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);

  // Handle required vector argument
  aiir::Value vector = builder.createBox(loc, args[0]);

  // Handle required mask argument
  fir::BoxValue maskBox = builder.createBox(loc, args[1]);
  aiir::Value mask = fir::getBase(maskBox);
  unsigned maskRank = maskBox.rank();

  // Handle required field argument
  aiir::Value field = builder.createBox(loc, args[2]);

  // Create mutable fir.box to be passed to the runtime for the result.
  aiir::Type resultArrayType = builder.getVarLenSeqTy(resultType, maskRank);
  fir::MutableBoxValue resultMutableBox = fir::factory::createTempMutableBox(
      builder, loc, resultArrayType, {},
      fir::isPolymorphicType(vector.getType()) ? vector : aiir::Value{});
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genUnpack(builder, loc, resultIrBox, vector, mask, field);

  return readAndAddCleanUp(resultMutableBox, resultType, "UNPACK");
}

// VERIFY
fir::ExtendedValue
IntrinsicLibrary::genVerify(aiir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 4);

  if (isStaticallyAbsent(args[3])) {
    // Kind not specified, so call scan/verify runtime routine that is
    // specialized on the kind of characters in string.

    // Handle required string base arg
    aiir::Value stringBase = fir::getBase(args[0]);

    // Handle required set string base arg
    aiir::Value setBase = fir::getBase(args[1]);

    // Handle kind argument; it is the kind of character in this case
    fir::KindTy kind =
        fir::factory::CharacterExprHelper{builder, loc}.getCharacterKind(
            stringBase.getType());

    // Get string length argument
    aiir::Value stringLen = fir::getLen(args[0]);

    // Get set string length argument
    aiir::Value setLen = fir::getLen(args[1]);

    // Handle optional back argument
    aiir::Value back =
        isStaticallyAbsent(args[2])
            ? builder.createIntegerConstant(loc, builder.getI1Type(), 0)
            : fir::getBase(args[2]);

    return builder.createConvert(
        loc, resultType,
        fir::runtime::genVerify(builder, loc, kind, stringBase, stringLen,
                                setBase, setLen, back));
  }
  // else use the runtime descriptor version of scan/verify

  // Handle optional argument, back
  auto makeRefThenEmbox = [&](aiir::Value b) {
    fir::LogicalType logTy = fir::LogicalType::get(
        builder.getContext(), builder.getKindMap().defaultLogicalKind());
    aiir::Value temp = builder.createTemporary(loc, logTy);
    aiir::Value castb = builder.createConvert(loc, logTy, b);
    fir::StoreOp::create(builder, loc, castb, temp);
    return builder.createBox(loc, temp);
  };
  aiir::Value back =
      fir::isUnboxedValue(args[2])
          ? makeRefThenEmbox(*args[2].getUnboxed())
          : fir::AbsentOp::create(builder, loc,
                                  fir::BoxType::get(builder.getI1Type()));

  // Handle required string argument
  aiir::Value string = builder.createBox(loc, args[0]);

  // Handle required set argument
  aiir::Value set = builder.createBox(loc, args[1]);

  // Handle kind argument
  aiir::Value kind = fir::getBase(args[3]);

  // Create result descriptor
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genVerifyDescriptor(builder, loc, resultIrBox, string, set,
                                    back, kind);

  // Handle cleanup of allocatable result descriptor and return
  return readAndAddCleanUp(resultMutableBox, resultType, "VERIFY");
}

/// Process calls to Minloc, Maxloc intrinsic functions
template <typename FN, typename FD>
fir::ExtendedValue
IntrinsicLibrary::genExtremumloc(FN func, FD funcDim, llvm::StringRef errMsg,
                                 aiir::Type resultType,
                                 llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 5);

  // Handle required array argument
  aiir::Value array = builder.createBox(loc, args[0]);
  unsigned rank = fir::BoxValue(array).rank();
  assert(rank >= 1);

  // Handle optional mask argument
  auto mask = isStaticallyAbsent(args[2])
                  ? fir::AbsentOp::create(
                        builder, loc, fir::BoxType::get(builder.getI1Type()))
                  : builder.createBox(loc, args[2]);

  // Handle optional kind argument
  auto kind = isStaticallyAbsent(args[3])
                  ? builder.createIntegerConstant(
                        loc, builder.getIndexType(),
                        builder.getKindMap().defaultIntegerKind())
                  : fir::getBase(args[3]);

  // Handle optional back argument
  auto back = isStaticallyAbsent(args[4]) ? builder.createBool(loc, false)
                                          : fir::getBase(args[4]);

  bool absentDim = isStaticallyAbsent(args[1]);

  if (!absentDim && rank == 1) {
    // If dim argument is present and the array is rank 1, then the result is
    // a scalar (since the the result is rank-1 or 0).
    // Therefore, we use a scalar result descriptor with Min/MaxlocDim().
    aiir::Value dim = fir::getBase(args[1]);
    // Create mutable fir.box to be passed to the runtime for the result.
    fir::MutableBoxValue resultMutableBox =
        fir::factory::createTempMutableBox(builder, loc, resultType);
    aiir::Value resultIrBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    funcDim(builder, loc, resultIrBox, array, dim, mask, kind, back);

    // Handle cleanup of allocatable result descriptor and return
    return readAndAddCleanUp(resultMutableBox, resultType, errMsg);
  }

  // Note: The Min/Maxloc/val cases below have an array result.

  // Create mutable fir.box to be passed to the runtime for the result.
  aiir::Type resultArrayType =
      builder.getVarLenSeqTy(resultType, absentDim ? 1 : rank - 1);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  aiir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  if (absentDim) {
    // Handle min/maxloc/val case where there is no dim argument
    // (calls Min/Maxloc()/MinMaxval() runtime routine)
    func(builder, loc, resultIrBox, array, mask, kind, back);
  } else {
    // else handle min/maxloc case with dim argument (calls
    // Min/Max/loc/val/Dim() runtime routine).
    aiir::Value dim = fir::getBase(args[1]);
    funcDim(builder, loc, resultIrBox, array, dim, mask, kind, back);
  }
  return readAndAddCleanUp(resultMutableBox, resultType, errMsg);
}

// MAXLOC
fir::ExtendedValue
IntrinsicLibrary::genMaxloc(aiir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  return genExtremumloc(fir::runtime::genMaxloc, fir::runtime::genMaxlocDim,
                        "MAXLOC", resultType, args);
}

/// Process calls to Maxval and Minval
template <typename FN, typename FD, typename FC>
fir::ExtendedValue
IntrinsicLibrary::genExtremumVal(FN func, FD funcDim, FC funcChar,
                                 llvm::StringRef errMsg, aiir::Type resultType,
                                 llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 3);

  // Handle required array argument
  fir::BoxValue arryTmp = builder.createBox(loc, args[0]);
  aiir::Value array = fir::getBase(arryTmp);
  int rank = arryTmp.rank();
  assert(rank >= 1);
  bool hasCharacterResult = arryTmp.isCharacter();

  // Handle optional mask argument
  auto mask = isStaticallyAbsent(args[2])
                  ? fir::AbsentOp::create(
                        builder, loc, fir::BoxType::get(builder.getI1Type()))
                  : builder.createBox(loc, args[2]);

  bool absentDim = isStaticallyAbsent(args[1]);

  // For Maxval/MinVal, we call the type specific versions of
  // Maxval/Minval because the result is scalar in the case below.
  if (!hasCharacterResult && (absentDim || rank == 1))
    return func(builder, loc, array, mask);

  if (hasCharacterResult && (absentDim || rank == 1)) {
    // Create mutable fir.box to be passed to the runtime for the result.
    fir::MutableBoxValue resultMutableBox =
        fir::factory::createTempMutableBox(builder, loc, resultType);
    aiir::Value resultIrBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    funcChar(builder, loc, resultIrBox, array, mask);

    // Handle cleanup of allocatable result descriptor and return
    return readAndAddCleanUp(resultMutableBox, resultType, errMsg);
  }

  // Handle Min/Maxval cases that have an array result.
  auto resultMutableBox =
      genFuncDim(funcDim, resultType, builder, loc, array, args[1], mask, rank);
  return readAndAddCleanUp(resultMutableBox, resultType, errMsg);
}

// MAXVAL
fir::ExtendedValue
IntrinsicLibrary::genMaxval(aiir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  return genExtremumVal(fir::runtime::genMaxval, fir::runtime::genMaxvalDim,
                        fir::runtime::genMaxvalChar, "MAXVAL", resultType,
                        args);
}

// MINLOC
fir::ExtendedValue
IntrinsicLibrary::genMinloc(aiir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  return genExtremumloc(fir::runtime::genMinloc, fir::runtime::genMinlocDim,
                        "MINLOC", resultType, args);
}

// MINVAL
fir::ExtendedValue
IntrinsicLibrary::genMinval(aiir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  return genExtremumVal(fir::runtime::genMinval, fir::runtime::genMinvalDim,
                        fir::runtime::genMinvalChar, "MINVAL", resultType,
                        args);
}

// MIN and MAX
template <bool isMax>
aiir::Value IntrinsicLibrary::genExtremum(aiir::Type,
                                          llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() >= 1);
  aiir::Value result = args[0];
  for (auto arg : args.drop_front())
    result = genExtremumResult<isMax>(loc, builder, result, arg);
  return result;
}

//===----------------------------------------------------------------------===//
// Argument lowering rules interface for intrinsic or intrinsic module
// procedure.
//===----------------------------------------------------------------------===//

const IntrinsicArgumentLoweringRules *
getIntrinsicArgumentLowering(llvm::StringRef specificName) {
  llvm::StringRef name = genericName(specificName);
  if (const IntrinsicHandler *handler = findIntrinsicHandler(name))
    if (!handler->argLoweringRules.hasDefaultRules())
      return &handler->argLoweringRules;
  if (const IntrinsicHandler *ppcHandler = findPPCIntrinsicHandler(name))
    if (!ppcHandler->argLoweringRules.hasDefaultRules())
      return &ppcHandler->argLoweringRules;
  if (const IntrinsicHandler *cudaHandler = findCUDAIntrinsicHandler(name))
    if (!cudaHandler->argLoweringRules.hasDefaultRules())
      return &cudaHandler->argLoweringRules;
  return nullptr;
}

const IntrinsicArgumentLoweringRules *
IntrinsicHandlerEntry::getArgumentLoweringRules() const {
  if (const IntrinsicHandler *const *handler =
          std::get_if<const IntrinsicHandler *>(&entry)) {
    assert(*handler);
    if (!(*handler)->argLoweringRules.hasDefaultRules())
      return &(*handler)->argLoweringRules;
  }
  return nullptr;
}

/// Return how argument \p argName should be lowered given the rules for the
/// intrinsic function.
fir::ArgLoweringRule
lowerIntrinsicArgumentAs(const IntrinsicArgumentLoweringRules &rules,
                         unsigned position) {
  assert(position < sizeof(rules.args) / (sizeof(decltype(*rules.args))) &&
         "invalid argument");
  return {rules.args[position].lowerAs,
          rules.args[position].handleDynamicOptional};
}

//===----------------------------------------------------------------------===//
// Public intrinsic call helpers
//===----------------------------------------------------------------------===//

std::pair<fir::ExtendedValue, bool>
genIntrinsicCall(fir::FirOpBuilder &builder, aiir::Location loc,
                 llvm::StringRef name, std::optional<aiir::Type> resultType,
                 llvm::ArrayRef<fir::ExtendedValue> args,
                 Fortran::lower::AbstractConverter *converter) {
  return IntrinsicLibrary{builder, loc, converter}.genIntrinsicCall(
      name, resultType, args);
}

aiir::Value genMax(fir::FirOpBuilder &builder, aiir::Location loc,
                   llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() > 0 && "max requires at least one argument");
  return IntrinsicLibrary{builder, loc}.genExtremum</*isMax=*/true>(
      args[0].getType(), args);
}

aiir::Value genMin(fir::FirOpBuilder &builder, aiir::Location loc,
                   llvm::ArrayRef<aiir::Value> args) {
  assert(args.size() > 0 && "min requires at least one argument");
  return IntrinsicLibrary{builder, loc}.genExtremum</*isMax=*/false>(
      args[0].getType(), args);
}

aiir::Value genDivC(fir::FirOpBuilder &builder, aiir::Location loc,
                    aiir::Type type, aiir::Value x, aiir::Value y) {
  return IntrinsicLibrary{builder, loc}.genRuntimeCall("divc", type, {x, y});
}

aiir::Value genPow(fir::FirOpBuilder &builder, aiir::Location loc,
                   aiir::Type type, aiir::Value x, aiir::Value y) {
  // TODO: since there is no libm version of pow with integer exponent,
  //       we have to provide an alternative implementation for
  //       "precise/strict" FP mode.
  //       One option is to generate internal function with inlined
  //       implementation and mark it 'strictfp'.
  //       Another option is to implement it in Fortran runtime library
  //       (just like matmul).
  if (type.isUnsignedInteger()) {
    assert(x.getType().isUnsignedInteger() && y.getType().isUnsignedInteger() &&
           "unsigned pow requires unsigned arguments");
    return IntrinsicLibrary{builder, loc}.genRuntimeCall("pow-unsigned", type,
                                                         {x, y});
  }
  assert(!x.getType().isUnsignedInteger() && !y.getType().isUnsignedInteger() &&
         "non-unsigned pow requires non-unsigned arguments");
  return IntrinsicLibrary{builder, loc}.genRuntimeCall("pow", type, {x, y});
}

aiir::SymbolRefAttr
getUnrestrictedIntrinsicSymbolRefAttr(fir::FirOpBuilder &builder,
                                      aiir::Location loc, llvm::StringRef name,
                                      aiir::FunctionType signature) {
  return IntrinsicLibrary{builder, loc}.getUnrestrictedIntrinsicSymbolRefAttr(
      name, signature);
}
} // namespace fir
