//===-- PPCIntrinsicCall.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper routines for constructing the FIR dialect of MLIR for PowerPC
// intrinsics. Extensive use of MLIR interfaces and MLIR's coding style
// (https://mlir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/PPCIntrinsicCall.h"
#include "flang/Evaluate/common.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace fir {

using PI = PPCIntrinsicLibrary;

// PPC specific intrinsic handlers.
static constexpr IntrinsicHandler ppcHandlers[]{
    {"__ppc_mma_assemble_acc",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::AssembleAcc, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr},
       {"arg1", asValue},
       {"arg2", asValue},
       {"arg3", asValue},
       {"arg4", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_assemble_pair",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::AssemblePair, MMAHandlerOp::SubToFunc>),
     {{{"pair", asAddr}, {"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_build_acc",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::AssembleAcc,
                         MMAHandlerOp::SubToFuncReverseArgOnLE>),
     {{{"acc", asAddr},
       {"arg1", asValue},
       {"arg2", asValue},
       {"arg3", asValue},
       {"arg4", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_disassemble_acc",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::DisassembleAcc, MMAHandlerOp::SubToFunc>),
     {{{"data", asAddr}, {"acc", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_disassemble_pair",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::DisassemblePair, MMAHandlerOp::SubToFunc>),
     {{{"data", asAddr}, {"pair", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mtfsf",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(&PI::genMtfsf<false>),
     {{{"mask", asValue}, {"r", asValue}}},
     /*isElemental=*/false},
    {"__ppc_mtfsfi",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(&PI::genMtfsf<true>),
     {{{"bf", asValue}, {"i", asValue}}},
     /*isElemental=*/false},
    {"__ppc_vec_abs",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(&PI::genVecAbs),
     {{{"arg1", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_add",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecAddAndMulSubXor<VecOp::Add>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_and",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecAddAndMulSubXor<VecOp::And>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_any_ge",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecAnyCompare<VecOp::Anyge>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_cmpge",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecCmp<VecOp::Cmpge>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_cmpgt",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecCmp<VecOp::Cmpgt>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_cmple",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecCmp<VecOp::Cmple>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_cmplt",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecCmp<VecOp::Cmplt>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_convert",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecConvert<VecOp::Convert>),
     {{{"v", asValue}, {"mold", asValue}}},
     /*isElemental=*/false},
    {"__ppc_vec_ctf",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecConvert<VecOp::Ctf>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_cvf",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecConvert<VecOp::Cvf>),
     {{{"arg1", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_extract",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(&PI::genVecExtract),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_insert",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(&PI::genVecInsert),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_mergeh",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecMerge<VecOp::Mergeh>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_mergel",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecMerge<VecOp::Mergel>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_msub",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecNmaddMsub<VecOp::Msub>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_mul",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecAddAndMulSubXor<VecOp::Mul>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_nmadd",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecNmaddMsub<VecOp::Nmadd>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_perm",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecPerm<VecOp::Perm>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_permi",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecPerm<VecOp::Permi>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_sel",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(&PI::genVecSel),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_sl",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecShift<VecOp::Sl>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_sld",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecShift<VecOp::Sld>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_sldw",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecShift<VecOp::Sldw>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_sll",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecShift<VecOp::Sll>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_slo",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecShift<VecOp::Slo>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_sr",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecShift<VecOp::Sr>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_srl",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecShift<VecOp::Srl>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_sro",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecShift<VecOp::Sro>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_st",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genVecStore<VecOp::St>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asAddr}}},
     /*isElemental=*/false},
    {"__ppc_vec_ste",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genVecStore<VecOp::Ste>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asAddr}}},
     /*isElemental=*/false},
    {"__ppc_vec_stxv",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genVecXStore<VecOp::Stxv>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asAddr}}},
     /*isElemental=*/false},
    {"__ppc_vec_stxvp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genVecStore<VecOp::Stxvp>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asAddr}}},
     /*isElemental=*/false},
    {"__ppc_vec_sub",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecAddAndMulSubXor<VecOp::Sub>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_xor",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecAddAndMulSubXor<VecOp::Xor>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_xst",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genVecXStore<VecOp::Xst>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asAddr}}},
     /*isElemental=*/false},
    {"__ppc_vec_xst_be",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genVecXStore<VecOp::Xst_be>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asAddr}}},
     /*isElemental=*/false},
    {"__ppc_vec_xstd2_",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genVecXStore<VecOp::Xstd2>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asAddr}}},
     /*isElemental=*/false},
    {"__ppc_vec_xstw4_",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genVecXStore<VecOp::Xstw4>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asAddr}}},
     /*isElemental=*/false},
};

static constexpr MathOperation ppcMathOperations[] = {
    // fcfi is just another name for fcfid, there is no llvm.ppc.fcfi.
    {"__ppc_fcfi", "llvm.ppc.fcfid", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fcfid", "llvm.ppc.fcfid", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fcfud", "llvm.ppc.fcfud", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fctid", "llvm.ppc.fctid", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fctidz", "llvm.ppc.fctidz", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fctiw", "llvm.ppc.fctiw", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fctiwz", "llvm.ppc.fctiwz", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fctudz", "llvm.ppc.fctudz", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fctuwz", "llvm.ppc.fctuwz", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fmadd", "llvm.fma.f32",
     genFuncType<Ty::Real<4>, Ty::Real<4>, Ty::Real<4>, Ty::Real<4>>,
     genMathOp<mlir::math::FmaOp>},
    {"__ppc_fmadd", "llvm.fma.f64",
     genFuncType<Ty::Real<8>, Ty::Real<8>, Ty::Real<8>, Ty::Real<8>>,
     genMathOp<mlir::math::FmaOp>},
    {"__ppc_fmsub", "llvm.ppc.fmsubs",
     genFuncType<Ty::Real<4>, Ty::Real<4>, Ty::Real<4>, Ty::Real<4>>,
     genLibCall},
    {"__ppc_fmsub", "llvm.ppc.fmsub",
     genFuncType<Ty::Real<8>, Ty::Real<8>, Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fnabs", "llvm.ppc.fnabss", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genLibCall},
    {"__ppc_fnabs", "llvm.ppc.fnabs", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fnmadd", "llvm.ppc.fnmadds",
     genFuncType<Ty::Real<4>, Ty::Real<4>, Ty::Real<4>, Ty::Real<4>>,
     genLibCall},
    {"__ppc_fnmadd", "llvm.ppc.fnmadd",
     genFuncType<Ty::Real<8>, Ty::Real<8>, Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fnmsub", "llvm.ppc.fnmsub.f32",
     genFuncType<Ty::Real<4>, Ty::Real<4>, Ty::Real<4>, Ty::Real<4>>,
     genLibCall},
    {"__ppc_fnmsub", "llvm.ppc.fnmsub.f64",
     genFuncType<Ty::Real<8>, Ty::Real<8>, Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fre", "llvm.ppc.fre", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fres", "llvm.ppc.fres", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genLibCall},
    {"__ppc_frsqrte", "llvm.ppc.frsqrte", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_frsqrtes", "llvm.ppc.frsqrtes",
     genFuncType<Ty::Real<4>, Ty::Real<4>>, genLibCall},
    {"__ppc_vec_cvbf16spn", "llvm.ppc.vsx.xvcvbf16spn",
     genFuncType<Ty::UnsignedVector<1>, Ty::UnsignedVector<1>>, genLibCall},
    {"__ppc_vec_cvspbf16_", "llvm.ppc.vsx.xvcvspbf16",
     genFuncType<Ty::UnsignedVector<1>, Ty::UnsignedVector<1>>, genLibCall},
    {"__ppc_vec_madd", "llvm.fma.v4f32",
     genFuncType<Ty::RealVector<4>, Ty::RealVector<4>, Ty::RealVector<4>,
                 Ty::RealVector<4>>,
     genLibCall},
    {"__ppc_vec_madd", "llvm.fma.v2f64",
     genFuncType<Ty::RealVector<8>, Ty::RealVector<8>, Ty::RealVector<8>,
                 Ty::RealVector<8>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.altivec.vmaxsb",
     genFuncType<Ty::IntegerVector<1>, Ty::IntegerVector<1>,
                 Ty::IntegerVector<1>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.altivec.vmaxsh",
     genFuncType<Ty::IntegerVector<2>, Ty::IntegerVector<2>,
                 Ty::IntegerVector<2>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.altivec.vmaxsw",
     genFuncType<Ty::IntegerVector<4>, Ty::IntegerVector<4>,
                 Ty::IntegerVector<4>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.altivec.vmaxsd",
     genFuncType<Ty::IntegerVector<8>, Ty::IntegerVector<8>,
                 Ty::IntegerVector<8>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.altivec.vmaxub",
     genFuncType<Ty::UnsignedVector<1>, Ty::UnsignedVector<1>,
                 Ty::UnsignedVector<1>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.altivec.vmaxuh",
     genFuncType<Ty::UnsignedVector<2>, Ty::UnsignedVector<2>,
                 Ty::UnsignedVector<2>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.altivec.vmaxuw",
     genFuncType<Ty::UnsignedVector<4>, Ty::UnsignedVector<4>,
                 Ty::UnsignedVector<4>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.altivec.vmaxud",
     genFuncType<Ty::UnsignedVector<8>, Ty::UnsignedVector<8>,
                 Ty::UnsignedVector<8>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.vsx.xvmaxsp",
     genFuncType<Ty::RealVector<4>, Ty::RealVector<4>, Ty::RealVector<4>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.vsx.xvmaxdp",
     genFuncType<Ty::RealVector<8>, Ty::RealVector<8>, Ty::RealVector<8>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.altivec.vminsb",
     genFuncType<Ty::IntegerVector<1>, Ty::IntegerVector<1>,
                 Ty::IntegerVector<1>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.altivec.vminsh",
     genFuncType<Ty::IntegerVector<2>, Ty::IntegerVector<2>,
                 Ty::IntegerVector<2>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.altivec.vminsw",
     genFuncType<Ty::IntegerVector<4>, Ty::IntegerVector<4>,
                 Ty::IntegerVector<4>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.altivec.vminsd",
     genFuncType<Ty::IntegerVector<8>, Ty::IntegerVector<8>,
                 Ty::IntegerVector<8>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.altivec.vminub",
     genFuncType<Ty::UnsignedVector<1>, Ty::UnsignedVector<1>,
                 Ty::UnsignedVector<1>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.altivec.vminuh",
     genFuncType<Ty::UnsignedVector<2>, Ty::UnsignedVector<2>,
                 Ty::UnsignedVector<2>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.altivec.vminuw",
     genFuncType<Ty::UnsignedVector<4>, Ty::UnsignedVector<4>,
                 Ty::UnsignedVector<4>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.altivec.vminud",
     genFuncType<Ty::UnsignedVector<8>, Ty::UnsignedVector<8>,
                 Ty::UnsignedVector<8>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.vsx.xvminsp",
     genFuncType<Ty::RealVector<4>, Ty::RealVector<4>, Ty::RealVector<4>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.vsx.xvmindp",
     genFuncType<Ty::RealVector<8>, Ty::RealVector<8>, Ty::RealVector<8>>,
     genLibCall},
    {"__ppc_vec_nmsub", "llvm.ppc.fnmsub.v4f32",
     genFuncType<Ty::RealVector<4>, Ty::RealVector<4>, Ty::RealVector<4>,
                 Ty::RealVector<4>>,
     genLibCall},
    {"__ppc_vec_nmsub", "llvm.ppc.fnmsub.v2f64",
     genFuncType<Ty::RealVector<8>, Ty::RealVector<8>, Ty::RealVector<8>,
                 Ty::RealVector<8>>,
     genLibCall},
};

const IntrinsicHandler *findPPCIntrinsicHandler(llvm::StringRef name) {
  auto compare = [](const IntrinsicHandler &ppcHandler, llvm::StringRef name) {
    return name.compare(ppcHandler.name) > 0;
  };
  auto result = llvm::lower_bound(ppcHandlers, name, compare);
  return result != std::end(ppcHandlers) && result->name == name ? result
                                                                 : nullptr;
}

using RtMap = Fortran::common::StaticMultimapView<MathOperation>;
static constexpr RtMap ppcMathOps(ppcMathOperations);
static_assert(ppcMathOps.Verify() && "map must be sorted");

std::pair<const MathOperation *, const MathOperation *>
checkPPCMathOperationsRange(llvm::StringRef name) {
  return ppcMathOps.equal_range(name);
}

// Helper functions for vector element ordering.
bool PPCIntrinsicLibrary::isBEVecElemOrderOnLE() {
  return (Fortran::evaluate::isHostLittleEndian &&
          converter->getLoweringOptions().getNoPPCNativeVecElemOrder());
}
bool PPCIntrinsicLibrary::isNativeVecElemOrderOnLE() {
  return (Fortran::evaluate::isHostLittleEndian &&
          !converter->getLoweringOptions().getNoPPCNativeVecElemOrder());
}
bool PPCIntrinsicLibrary::changeVecElemOrder() {
  return (Fortran::evaluate::isHostLittleEndian !=
          converter->getLoweringOptions().getNoPPCNativeVecElemOrder());
}

static mlir::FunctionType genMmaVpFuncType(mlir::MLIRContext *context,
                                           int quadCnt, int pairCnt, int vecCnt,
                                           int intCnt = 0,
                                           int vecElemBitSize = 8,
                                           int intBitSize = 32) {
  // Constructs a function type with the following signature:
  // Result type: __vector_pair
  // Arguments:
  //   quadCnt: number of arguments that has __vector_quad type, followed by
  //   pairCnt: number of arguments that has __vector_pair type, followed by
  //   vecCnt: number of arguments that has vector(integer) type, followed by
  //   intCnt: number of arguments that has integer type
  //   vecElemBitSize: specifies the size of vector elements in bits
  //   intBitSize: specifies the size of integer arguments in bits
  auto vType{mlir::VectorType::get(
      128 / vecElemBitSize, mlir::IntegerType::get(context, vecElemBitSize))};
  auto vpType{fir::VectorType::get(256, mlir::IntegerType::get(context, 1))};
  auto vqType{fir::VectorType::get(512, mlir::IntegerType::get(context, 1))};
  auto iType{mlir::IntegerType::get(context, intBitSize)};
  llvm::SmallVector<mlir::Type> argTypes;
  for (int i = 0; i < quadCnt; ++i) {
    argTypes.push_back(vqType);
  }
  for (int i = 0; i < pairCnt; ++i) {
    argTypes.push_back(vpType);
  }
  for (int i = 0; i < vecCnt; ++i) {
    argTypes.push_back(vType);
  }
  for (int i = 0; i < intCnt; ++i) {
    argTypes.push_back(iType);
  }

  return mlir::FunctionType::get(context, argTypes, {vpType});
}

static mlir::FunctionType genMmaVqFuncType(mlir::MLIRContext *context,
                                           int quadCnt, int pairCnt, int vecCnt,
                                           int intCnt = 0,
                                           int vecElemBitSize = 8,
                                           int intBitSize = 32) {
  // Constructs a function type with the following signature:
  // Result type: __vector_quad
  // Arguments:
  //   quadCnt: number of arguments that has __vector_quad type, followed by
  //   pairCnt: number of arguments that has __vector_pair type, followed by
  //   vecCnt: number of arguments that has vector(integer) type, followed by
  //   intCnt: number of arguments that has integer type
  //   vecElemBitSize: specifies the size of vector elements in bits
  //   intBitSize: specifies the size of integer arguments in bits
  auto vType{mlir::VectorType::get(
      128 / vecElemBitSize, mlir::IntegerType::get(context, vecElemBitSize))};
  auto vpType{fir::VectorType::get(256, mlir::IntegerType::get(context, 1))};
  auto vqType{fir::VectorType::get(512, mlir::IntegerType::get(context, 1))};
  auto iType{mlir::IntegerType::get(context, intBitSize)};
  llvm::SmallVector<mlir::Type> argTypes;
  for (int i = 0; i < quadCnt; ++i) {
    argTypes.push_back(vqType);
  }
  for (int i = 0; i < pairCnt; ++i) {
    argTypes.push_back(vpType);
  }
  for (int i = 0; i < vecCnt; ++i) {
    argTypes.push_back(vType);
  }
  for (int i = 0; i < intCnt; ++i) {
    argTypes.push_back(iType);
  }

  return mlir::FunctionType::get(context, argTypes, {vqType});
}

mlir::FunctionType genMmaDisassembleFuncType(mlir::MLIRContext *context,
                                             MMAOp mmaOp) {
  auto vType{mlir::VectorType::get(16, mlir::IntegerType::get(context, 8))};
  llvm::SmallVector<mlir::Type> members;

  if (mmaOp == MMAOp::DisassembleAcc) {
    auto vqType{fir::VectorType::get(512, mlir::IntegerType::get(context, 1))};
    members.push_back(vType);
    members.push_back(vType);
    members.push_back(vType);
    members.push_back(vType);
    auto resType{mlir::LLVM::LLVMStructType::getLiteral(context, members)};
    return mlir::FunctionType::get(context, {vqType}, {resType});
  } else if (mmaOp == MMAOp::DisassemblePair) {
    auto vpType{fir::VectorType::get(256, mlir::IntegerType::get(context, 1))};
    members.push_back(vType);
    members.push_back(vType);
    auto resType{mlir::LLVM::LLVMStructType::getLiteral(context, members)};
    return mlir::FunctionType::get(context, {vpType}, {resType});
  } else {
    llvm_unreachable(
        "Unsupported intrinsic code for function signature generator");
  }
}

//===----------------------------------------------------------------------===//
// PowerPC specific intrinsic handlers.
//===----------------------------------------------------------------------===//

// MTFSF, MTFSFI
template <bool isImm>
void PPCIntrinsicLibrary::genMtfsf(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  llvm::SmallVector<mlir::Value> scalarArgs;
  for (const fir::ExtendedValue &arg : args)
    if (arg.getUnboxed())
      scalarArgs.emplace_back(fir::getBase(arg));
    else
      mlir::emitError(loc, "nonscalar intrinsic argument");

  mlir::FunctionType libFuncType;
  mlir::func::FuncOp funcOp;
  if (isImm) {
    libFuncType = genFuncType<Ty::Void, Ty::Integer<4>, Ty::Integer<4>>(
        builder.getContext(), builder);
    funcOp = builder.addNamedFunction(loc, "llvm.ppc.mtfsfi", libFuncType);
  } else {
    libFuncType = genFuncType<Ty::Void, Ty::Integer<4>, Ty::Real<8>>(
        builder.getContext(), builder);
    funcOp = builder.addNamedFunction(loc, "llvm.ppc.mtfsf", libFuncType);
  }
  builder.create<fir::CallOp>(loc, funcOp, scalarArgs);
}

// VEC_ABS
fir::ExtendedValue
PPCIntrinsicLibrary::genVecAbs(mlir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  auto context{builder.getContext()};
  auto argBases{getBasesForArgs(args)};
  auto vTypeInfo{getVecTypeFromFir(argBases[0])};

  mlir::func::FuncOp funcOp{nullptr};
  mlir::FunctionType ftype;
  llvm::StringRef fname{};
  if (vTypeInfo.isFloat()) {
    if (vTypeInfo.isFloat32()) {
      fname = "llvm.fabs.v4f32";
      ftype =
          genFuncType<Ty::RealVector<4>, Ty::RealVector<4>>(context, builder);
    } else if (vTypeInfo.isFloat64()) {
      fname = "llvm.fabs.v2f64";
      ftype =
          genFuncType<Ty::RealVector<8>, Ty::RealVector<8>>(context, builder);
    }

    funcOp = builder.addNamedFunction(loc, fname, ftype);
    auto callOp{builder.create<fir::CallOp>(loc, funcOp, argBases[0])};
    return callOp.getResult(0);
  } else if (auto eleTy = vTypeInfo.eleTy.dyn_cast<mlir::IntegerType>()) {
    // vec_abs(arg1) = max(0 - arg1, arg1)

    auto newVecTy{mlir::VectorType::get(vTypeInfo.len, eleTy)};
    auto varg1{builder.createConvert(loc, newVecTy, argBases[0])};
    // construct vector(0,..)
    auto zeroVal{builder.createIntegerConstant(loc, eleTy, 0)};
    auto vZero{
        builder.create<mlir::vector::BroadcastOp>(loc, newVecTy, zeroVal)};
    auto zeroSubVarg1{builder.create<mlir::arith::SubIOp>(loc, vZero, varg1)};

    mlir::func::FuncOp funcOp{nullptr};
    switch (eleTy.getWidth()) {
    case 8:
      fname = "llvm.ppc.altivec.vmaxsb";
      ftype = genFuncType<Ty::IntegerVector<1>, Ty::IntegerVector<1>,
                          Ty::IntegerVector<1>>(context, builder);
      break;
    case 16:
      fname = "llvm.ppc.altivec.vmaxsh";
      ftype = genFuncType<Ty::IntegerVector<2>, Ty::IntegerVector<2>,
                          Ty::IntegerVector<2>>(context, builder);
      break;
    case 32:
      fname = "llvm.ppc.altivec.vmaxsw";
      ftype = genFuncType<Ty::IntegerVector<4>, Ty::IntegerVector<4>,
                          Ty::IntegerVector<4>>(context, builder);
      break;
    case 64:
      fname = "llvm.ppc.altivec.vmaxsd";
      ftype = genFuncType<Ty::IntegerVector<8>, Ty::IntegerVector<8>,
                          Ty::IntegerVector<8>>(context, builder);
      break;
    default:
      llvm_unreachable("invalid integer size");
    }
    funcOp = builder.addNamedFunction(loc, fname, ftype);

    mlir::Value args[] = {zeroSubVarg1, varg1};
    auto callOp{builder.create<fir::CallOp>(loc, funcOp, args)};
    return builder.createConvert(loc, argBases[0].getType(),
                                 callOp.getResult(0));
  }

  llvm_unreachable("unknown vector type");
}

// VEC_ADD, VEC_AND, VEC_SUB, VEC_MUL, VEC_XOR
template <VecOp vop>
fir::ExtendedValue PPCIntrinsicLibrary::genVecAddAndMulSubXor(
    mlir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  auto argBases{getBasesForArgs(args)};
  auto argsTy{getTypesForArgs(argBases)};
  assert(argsTy[0].isa<fir::VectorType>() && argsTy[1].isa<fir::VectorType>());

  auto vecTyInfo{getVecTypeFromFir(argBases[0])};

  const auto isInteger{vecTyInfo.eleTy.isa<mlir::IntegerType>()};
  const auto isFloat{vecTyInfo.eleTy.isa<mlir::FloatType>()};
  assert((isInteger || isFloat) && "unknown vector type");

  auto vargs{convertVecArgs(builder, loc, vecTyInfo, argBases)};

  mlir::Value r{nullptr};
  switch (vop) {
  case VecOp::Add:
    if (isInteger)
      r = builder.create<mlir::arith::AddIOp>(loc, vargs[0], vargs[1]);
    else if (isFloat)
      r = builder.create<mlir::arith::AddFOp>(loc, vargs[0], vargs[1]);
    break;
  case VecOp::Mul:
    if (isInteger)
      r = builder.create<mlir::arith::MulIOp>(loc, vargs[0], vargs[1]);
    else if (isFloat)
      r = builder.create<mlir::arith::MulFOp>(loc, vargs[0], vargs[1]);
    break;
  case VecOp::Sub:
    if (isInteger)
      r = builder.create<mlir::arith::SubIOp>(loc, vargs[0], vargs[1]);
    else if (isFloat)
      r = builder.create<mlir::arith::SubFOp>(loc, vargs[0], vargs[1]);
    break;
  case VecOp::And:
  case VecOp::Xor: {
    mlir::Value arg1{nullptr};
    mlir::Value arg2{nullptr};
    if (isInteger) {
      arg1 = vargs[0];
      arg2 = vargs[1];
    } else if (isFloat) {
      // bitcast the arguments to integer
      auto wd{vecTyInfo.eleTy.dyn_cast<mlir::FloatType>().getWidth()};
      auto ftype{builder.getIntegerType(wd)};
      auto bcVecTy{mlir::VectorType::get(vecTyInfo.len, ftype)};
      arg1 = builder.create<mlir::vector::BitCastOp>(loc, bcVecTy, vargs[0]);
      arg2 = builder.create<mlir::vector::BitCastOp>(loc, bcVecTy, vargs[1]);
    }
    if (vop == VecOp::And)
      r = builder.create<mlir::arith::AndIOp>(loc, arg1, arg2);
    else if (vop == VecOp::Xor)
      r = builder.create<mlir::arith::XOrIOp>(loc, arg1, arg2);

    if (isFloat)
      r = builder.create<mlir::vector::BitCastOp>(loc, vargs[0].getType(), r);

    break;
  }
  }

  return builder.createConvert(loc, argsTy[0], r);
}

// VEC_ANY_GE
template <VecOp vop>
fir::ExtendedValue
PPCIntrinsicLibrary::genVecAnyCompare(mlir::Type resultType,
                                      llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  assert(vop == VecOp::Anyge && "unknown vector compare operation");
  auto argBases{getBasesForArgs(args)};
  VecTypeInfo vTypeInfo{getVecTypeFromFir(argBases[0])};
  [[maybe_unused]] const auto isSupportedTy{
      mlir::isa<mlir::Float32Type, mlir::Float64Type, mlir::IntegerType>(
          vTypeInfo.eleTy)};
  assert(isSupportedTy && "unsupported vector type");

  // Constants for mapping CR6 bits to predicate result
  enum { CR6_EQ_REV = 1, CR6_LT_REV = 3 };

  auto context{builder.getContext()};

  static std::map<std::pair<ParamTypeId, unsigned>,
                  std::pair<llvm::StringRef, mlir::FunctionType>>
      uiBuiltin{
          {std::make_pair(ParamTypeId::IntegerVector, 8),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtsb.p",
               genFuncType<Ty::Integer<4>, Ty::Integer<4>, Ty::IntegerVector<1>,
                           Ty::IntegerVector<1>>(context, builder))},
          {std::make_pair(ParamTypeId::IntegerVector, 16),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtsh.p",
               genFuncType<Ty::Integer<4>, Ty::Integer<4>, Ty::IntegerVector<2>,
                           Ty::IntegerVector<2>>(context, builder))},
          {std::make_pair(ParamTypeId::IntegerVector, 32),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtsw.p",
               genFuncType<Ty::Integer<4>, Ty::Integer<4>, Ty::IntegerVector<4>,
                           Ty::IntegerVector<4>>(context, builder))},
          {std::make_pair(ParamTypeId::IntegerVector, 64),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtsd.p",
               genFuncType<Ty::Integer<4>, Ty::Integer<4>, Ty::IntegerVector<8>,
                           Ty::IntegerVector<8>>(context, builder))},
          {std::make_pair(ParamTypeId::UnsignedVector, 8),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtub.p",
               genFuncType<Ty::Integer<4>, Ty::Integer<4>,
                           Ty::UnsignedVector<1>, Ty::UnsignedVector<1>>(
                   context, builder))},
          {std::make_pair(ParamTypeId::UnsignedVector, 16),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtuh.p",
               genFuncType<Ty::Integer<4>, Ty::Integer<4>,
                           Ty::UnsignedVector<2>, Ty::UnsignedVector<2>>(
                   context, builder))},
          {std::make_pair(ParamTypeId::UnsignedVector, 32),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtuw.p",
               genFuncType<Ty::Integer<4>, Ty::Integer<4>,
                           Ty::UnsignedVector<4>, Ty::UnsignedVector<4>>(
                   context, builder))},
          {std::make_pair(ParamTypeId::UnsignedVector, 64),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtud.p",
               genFuncType<Ty::Integer<4>, Ty::Integer<4>,
                           Ty::UnsignedVector<8>, Ty::UnsignedVector<8>>(
                   context, builder))},
      };

  mlir::FunctionType ftype{nullptr};
  llvm::StringRef fname;
  const auto i32Ty{mlir::IntegerType::get(context, 32)};
  llvm::SmallVector<mlir::Value> cmpArgs;
  mlir::Value op{nullptr};
  const auto width{vTypeInfo.eleTy.getIntOrFloatBitWidth()};

  if (auto elementTy = mlir::dyn_cast<mlir::IntegerType>(vTypeInfo.eleTy)) {
    std::pair<llvm::StringRef, mlir::FunctionType> bi;
    bi = (elementTy.isUnsignedInteger())
             ? uiBuiltin[std::pair(ParamTypeId::UnsignedVector, width)]
             : uiBuiltin[std::pair(ParamTypeId::IntegerVector, width)];

    fname = std::get<0>(bi);
    ftype = std::get<1>(bi);

    op = builder.createIntegerConstant(loc, i32Ty, CR6_LT_REV);
    cmpArgs.emplace_back(op);
    // reverse the argument order
    cmpArgs.emplace_back(argBases[1]);
    cmpArgs.emplace_back(argBases[0]);
  } else if (vTypeInfo.isFloat()) {
    if (vTypeInfo.isFloat32()) {
      fname = "llvm.ppc.vsx.xvcmpgesp.p";
      ftype = genFuncType<Ty::Integer<4>, Ty::Integer<4>, Ty::RealVector<4>,
                          Ty::RealVector<4>>(context, builder);
    } else {
      fname = "llvm.ppc.vsx.xvcmpgedp.p";
      ftype = genFuncType<Ty::Integer<4>, Ty::Integer<4>, Ty::RealVector<8>,
                          Ty::RealVector<8>>(context, builder);
    }
    op = builder.createIntegerConstant(loc, i32Ty, CR6_EQ_REV);
    cmpArgs.emplace_back(op);
    cmpArgs.emplace_back(argBases[0]);
    cmpArgs.emplace_back(argBases[1]);
  }
  assert((!fname.empty() && ftype) && "invalid type");

  mlir::func::FuncOp funcOp{builder.addNamedFunction(loc, fname, ftype)};
  auto callOp{builder.create<fir::CallOp>(loc, funcOp, cmpArgs)};
  return callOp.getResult(0);
}

static std::pair<llvm::StringRef, mlir::FunctionType>
getVecCmpFuncTypeAndName(VecTypeInfo &vTypeInfo, VecOp vop,
                         fir::FirOpBuilder &builder) {
  auto context{builder.getContext()};
  static std::map<std::pair<ParamTypeId, unsigned>,
                  std::pair<llvm::StringRef, mlir::FunctionType>>
      iuBuiltinName{
          {std::make_pair(ParamTypeId::IntegerVector, 8),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtsb",
               genFuncType<Ty::UnsignedVector<1>, Ty::IntegerVector<1>,
                           Ty::IntegerVector<1>>(context, builder))},
          {std::make_pair(ParamTypeId::IntegerVector, 16),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtsh",
               genFuncType<Ty::UnsignedVector<2>, Ty::IntegerVector<2>,
                           Ty::IntegerVector<2>>(context, builder))},
          {std::make_pair(ParamTypeId::IntegerVector, 32),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtsw",
               genFuncType<Ty::UnsignedVector<4>, Ty::IntegerVector<4>,
                           Ty::IntegerVector<4>>(context, builder))},
          {std::make_pair(ParamTypeId::IntegerVector, 64),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtsd",
               genFuncType<Ty::UnsignedVector<8>, Ty::IntegerVector<8>,
                           Ty::IntegerVector<8>>(context, builder))},
          {std::make_pair(ParamTypeId::UnsignedVector, 8),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtub",
               genFuncType<Ty::UnsignedVector<1>, Ty::UnsignedVector<1>,
                           Ty::UnsignedVector<1>>(context, builder))},
          {std::make_pair(ParamTypeId::UnsignedVector, 16),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtuh",
               genFuncType<Ty::UnsignedVector<2>, Ty::UnsignedVector<2>,
                           Ty::UnsignedVector<2>>(context, builder))},
          {std::make_pair(ParamTypeId::UnsignedVector, 32),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtuw",
               genFuncType<Ty::UnsignedVector<4>, Ty::UnsignedVector<4>,
                           Ty::UnsignedVector<4>>(context, builder))},
          {std::make_pair(ParamTypeId::UnsignedVector, 64),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtud",
               genFuncType<Ty::UnsignedVector<8>, Ty::UnsignedVector<8>,
                           Ty::UnsignedVector<8>>(context, builder))}};

  // VSX only defines GE and GT builtins. Cmple and Cmplt use GE and GT with
  // arguments revsered.
  enum class Cmp { gtOrLt, geOrLe };
  static std::map<std::pair<Cmp, int>,
                  std::pair<llvm::StringRef, mlir::FunctionType>>
      rGBI{{std::make_pair(Cmp::geOrLe, 32),
            std::make_pair("llvm.ppc.vsx.xvcmpgesp",
                           genFuncType<Ty::UnsignedVector<4>, Ty::RealVector<4>,
                                       Ty::RealVector<4>>(context, builder))},
           {std::make_pair(Cmp::geOrLe, 64),
            std::make_pair("llvm.ppc.vsx.xvcmpgedp",
                           genFuncType<Ty::UnsignedVector<8>, Ty::RealVector<8>,
                                       Ty::RealVector<8>>(context, builder))},
           {std::make_pair(Cmp::gtOrLt, 32),
            std::make_pair("llvm.ppc.vsx.xvcmpgtsp",
                           genFuncType<Ty::UnsignedVector<4>, Ty::RealVector<4>,
                                       Ty::RealVector<4>>(context, builder))},
           {std::make_pair(Cmp::gtOrLt, 64),
            std::make_pair("llvm.ppc.vsx.xvcmpgtdp",
                           genFuncType<Ty::UnsignedVector<8>, Ty::RealVector<8>,
                                       Ty::RealVector<8>>(context, builder))}};

  const auto width{vTypeInfo.eleTy.getIntOrFloatBitWidth()};
  std::pair<llvm::StringRef, mlir::FunctionType> specFunc;
  if (auto elementTy = mlir::dyn_cast<mlir::IntegerType>(vTypeInfo.eleTy))
    specFunc =
        (elementTy.isUnsignedInteger())
            ? iuBuiltinName[std::make_pair(ParamTypeId::UnsignedVector, width)]
            : iuBuiltinName[std::make_pair(ParamTypeId::IntegerVector, width)];
  else if (vTypeInfo.isFloat())
    specFunc = (vop == VecOp::Cmpge || vop == VecOp::Cmple)
                   ? rGBI[std::make_pair(Cmp::geOrLe, width)]
                   : rGBI[std::make_pair(Cmp::gtOrLt, width)];

  assert(!std::get<0>(specFunc).empty() && "unknown builtin name");
  assert(std::get<1>(specFunc) && "unknown function type");
  return specFunc;
}

// VEC_CMPGE, VEC_CMPGT, VEC_CMPLE, VEC_CMPLT
template <VecOp vop>
fir::ExtendedValue
PPCIntrinsicLibrary::genVecCmp(mlir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  auto context{builder.getContext()};
  auto argBases{getBasesForArgs(args)};
  VecTypeInfo vecTyInfo{getVecTypeFromFir(argBases[0])};
  auto varg{convertVecArgs(builder, loc, vecTyInfo, argBases)};

  std::pair<llvm::StringRef, mlir::FunctionType> funcTyNam{
      getVecCmpFuncTypeAndName(vecTyInfo, vop, builder)};

  mlir::func::FuncOp funcOp = builder.addNamedFunction(
      loc, std::get<0>(funcTyNam), std::get<1>(funcTyNam));

  mlir::Value res{nullptr};

  if (auto eTy = vecTyInfo.eleTy.dyn_cast<mlir::IntegerType>()) {
    constexpr int firstArg{0};
    constexpr int secondArg{1};
    std::map<VecOp, std::array<int, 2>> argOrder{
        {VecOp::Cmpge, {secondArg, firstArg}},
        {VecOp::Cmple, {firstArg, secondArg}},
        {VecOp::Cmpgt, {firstArg, secondArg}},
        {VecOp::Cmplt, {secondArg, firstArg}}};

    // Construct the function return type, unsigned vector, for conversion.
    auto itype = mlir::IntegerType::get(context, eTy.getWidth(),
                                        mlir::IntegerType::Unsigned);
    auto returnType = fir::VectorType::get(vecTyInfo.len, itype);

    switch (vop) {
    case VecOp::Cmpgt:
    case VecOp::Cmplt: {
      // arg1 > arg2 --> vcmpgt(arg1, arg2)
      // arg1 < arg2 --> vcmpgt(arg2, arg1)
      mlir::Value vargs[]{argBases[argOrder[vop][0]],
                          argBases[argOrder[vop][1]]};
      auto callOp{builder.create<fir::CallOp>(loc, funcOp, vargs)};
      res = callOp.getResult(0);
      break;
    }
    case VecOp::Cmpge:
    case VecOp::Cmple: {
      // arg1 >= arg2 --> vcmpge(arg2, arg1) xor vector(-1)
      // arg1 <= arg2 --> vcmpge(arg1, arg2) xor vector(-1)
      mlir::Value vargs[]{argBases[argOrder[vop][0]],
                          argBases[argOrder[vop][1]]};

      // Construct a constant vector(-1)
      auto negOneVal{builder.createIntegerConstant(
          loc, getConvertedElementType(context, eTy), -1)};
      auto vNegOne{builder.create<mlir::vector::BroadcastOp>(
          loc, vecTyInfo.toMlirVectorType(context), negOneVal)};

      auto callOp{builder.create<fir::CallOp>(loc, funcOp, vargs)};
      mlir::Value callRes{callOp.getResult(0)};
      auto vargs2{
          convertVecArgs(builder, loc, vecTyInfo, mlir::ValueRange{callRes})};
      auto xorRes{builder.create<mlir::arith::XOrIOp>(loc, vargs2[0], vNegOne)};

      res = builder.createConvert(loc, returnType, xorRes);
      break;
    }
    default:
      llvm_unreachable("Invalid vector operation for generator");
    }
  } else if (vecTyInfo.isFloat()) {
    mlir::Value vargs[2];
    switch (vop) {
    case VecOp::Cmpge:
    case VecOp::Cmpgt:
      vargs[0] = argBases[0];
      vargs[1] = argBases[1];
      break;
    case VecOp::Cmple:
    case VecOp::Cmplt:
      // Swap the arguments as xvcmpg[et] is used
      vargs[0] = argBases[1];
      vargs[1] = argBases[0];
      break;
    default:
      llvm_unreachable("Invalid vector operation for generator");
    }
    auto callOp{builder.create<fir::CallOp>(loc, funcOp, vargs)};
    res = callOp.getResult(0);
  } else
    llvm_unreachable("invalid vector type");

  return res;
}

static inline mlir::Value swapVectorWordPairs(fir::FirOpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value arg) {
  auto ty = arg.getType();
  auto context{builder.getContext()};
  auto vtype{mlir::VectorType::get(16, mlir::IntegerType::get(context, 8))};

  if (ty != vtype)
    arg = builder.create<mlir::LLVM::BitcastOp>(loc, vtype, arg).getResult();

  llvm::SmallVector<int64_t, 16> mask{4,  5,  6,  7,  0, 1, 2,  3,
                                      12, 13, 14, 15, 8, 9, 10, 11};
  arg = builder.create<mlir::vector::ShuffleOp>(loc, arg, arg, mask);
  if (ty != vtype)
    arg = builder.create<mlir::LLVM::BitcastOp>(loc, ty, arg);
  return arg;
}

// VEC_CONVERT, VEC_CTF, VEC_CVF
template <VecOp vop>
fir::ExtendedValue
PPCIntrinsicLibrary::genVecConvert(mlir::Type resultType,
                                   llvm::ArrayRef<fir::ExtendedValue> args) {
  auto context{builder.getContext()};
  auto argBases{getBasesForArgs(args)};
  auto vecTyInfo{getVecTypeFromFir(argBases[0])};
  auto mlirTy{vecTyInfo.toMlirVectorType(context)};
  auto vArg1{builder.createConvert(loc, mlirTy, argBases[0])};
  const auto i32Ty{mlir::IntegerType::get(context, 32)};

  switch (vop) {
  case VecOp::Ctf: {
    assert(args.size() == 2);
    auto convArg{builder.createConvert(loc, i32Ty, argBases[1])};
    auto eTy{vecTyInfo.eleTy.dyn_cast<mlir::IntegerType>()};
    assert(eTy && "Unsupported vector type");
    const auto isUnsigned{eTy.isUnsignedInteger()};
    const auto width{eTy.getWidth()};

    if (width == 32) {
      auto ftype{(isUnsigned)
                     ? genFuncType<Ty::RealVector<4>, Ty::UnsignedVector<4>,
                                   Ty::Integer<4>>(context, builder)
                     : genFuncType<Ty::RealVector<4>, Ty::IntegerVector<4>,
                                   Ty::Integer<4>>(context, builder)};
      const llvm::StringRef fname{(isUnsigned) ? "llvm.ppc.altivec.vcfux"
                                               : "llvm.ppc.altivec.vcfsx"};
      auto funcOp{builder.addNamedFunction(loc, fname, ftype)};
      mlir::Value newArgs[] = {argBases[0], convArg};
      auto callOp{builder.create<fir::CallOp>(loc, funcOp, newArgs)};

      return callOp.getResult(0);
    } else if (width == 64) {
      auto fTy{mlir::FloatType::getF64(context)};
      auto ty{mlir::VectorType::get(2, fTy)};

      // vec_vtf(arg1, arg2) = fmul(1.0 / (1 << arg2), llvm.sitofp(arg1))
      auto convOp{(isUnsigned)
                      ? builder.create<mlir::LLVM::UIToFPOp>(loc, ty, vArg1)
                      : builder.create<mlir::LLVM::SIToFPOp>(loc, ty, vArg1)};

      // construct vector<1./(1<<arg1), 1.0/(1<<arg1)>
      auto constInt{
          mlir::dyn_cast<mlir::arith::ConstantOp>(argBases[1].getDefiningOp())
              .getValue()
              .dyn_cast_or_null<mlir::IntegerAttr>()};
      assert(constInt && "expected integer constant argument");
      double f{1.0 / (1 << constInt.getInt())};
      llvm::SmallVector<double> vals{f, f};
      auto constOp{builder.create<mlir::arith::ConstantOp>(
          loc, ty, builder.getF64VectorAttr(vals))};

      auto mulOp{builder.create<mlir::LLVM::FMulOp>(
          loc, ty, convOp->getResult(0), constOp)};

      return builder.createConvert(loc, fir::VectorType::get(2, fTy), mulOp);
    }
    llvm_unreachable("invalid element integer kind");
  }
  case VecOp::Convert: {
    assert(args.size() == 2);
    // resultType has mold type (if scalar) or element type (if array)
    auto resTyInfo{getVecTypeFromFirType(resultType)};
    auto moldTy{resTyInfo.toMlirVectorType(context)};
    auto firTy{resTyInfo.toFirVectorType()};

    // vec_convert(v, mold) = bitcast v to "type of mold"
    auto conv{builder.create<mlir::LLVM::BitcastOp>(loc, moldTy, vArg1)};

    return builder.createConvert(loc, firTy, conv);
  }
  case VecOp::Cvf: {
    assert(args.size() == 1);

    mlir::Value newArgs[]{vArg1};
    if (vecTyInfo.isFloat32()) {
      if (changeVecElemOrder())
        newArgs[0] = swapVectorWordPairs(builder, loc, newArgs[0]);

      const llvm::StringRef fname{"llvm.ppc.vsx.xvcvspdp"};
      auto ftype{
          genFuncType<Ty::RealVector<8>, Ty::RealVector<4>>(context, builder)};
      auto funcOp{builder.addNamedFunction(loc, fname, ftype)};
      auto callOp{builder.create<fir::CallOp>(loc, funcOp, newArgs)};

      return callOp.getResult(0);
    } else if (vecTyInfo.isFloat64()) {
      const llvm::StringRef fname{"llvm.ppc.vsx.xvcvdpsp"};
      auto ftype{
          genFuncType<Ty::RealVector<4>, Ty::RealVector<8>>(context, builder)};
      auto funcOp{builder.addNamedFunction(loc, fname, ftype)};
      newArgs[0] =
          builder.create<fir::CallOp>(loc, funcOp, newArgs).getResult(0);
      auto fvf32Ty{newArgs[0].getType()};
      auto f32type{mlir::FloatType::getF32(context)};
      auto mvf32Ty{mlir::VectorType::get(4, f32type)};
      newArgs[0] = builder.createConvert(loc, mvf32Ty, newArgs[0]);

      if (changeVecElemOrder())
        newArgs[0] = swapVectorWordPairs(builder, loc, newArgs[0]);

      return builder.createConvert(loc, fvf32Ty, newArgs[0]);
    }
    llvm_unreachable("invalid element integer kind");
  }
  default:
    llvm_unreachable("Invalid vector operation for generator");
  }
}

static mlir::Value convertVectorElementOrder(fir::FirOpBuilder &builder,
                                             mlir::Location loc,
                                             VecTypeInfo vecInfo,
                                             mlir::Value idx) {
  mlir::Value numSub1{
      builder.createIntegerConstant(loc, idx.getType(), vecInfo.len - 1)};
  return builder.create<mlir::LLVM::SubOp>(loc, idx.getType(), numSub1, idx);
}

// VEC_EXTRACT
fir::ExtendedValue
PPCIntrinsicLibrary::genVecExtract(mlir::Type resultType,
                                   llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  auto argBases{getBasesForArgs(args)};
  auto argTypes{getTypesForArgs(argBases)};
  auto vecTyInfo{getVecTypeFromFir(argBases[0])};

  auto mlirTy{vecTyInfo.toMlirVectorType(builder.getContext())};
  auto varg0{builder.createConvert(loc, mlirTy, argBases[0])};

  // arg2 modulo the number of elements in arg1 to determine the element
  // position
  auto numEle{builder.createIntegerConstant(loc, argTypes[1], vecTyInfo.len)};
  mlir::Value uremOp{
      builder.create<mlir::LLVM::URemOp>(loc, argBases[1], numEle)};

  if (!isNativeVecElemOrderOnLE())
    uremOp = convertVectorElementOrder(builder, loc, vecTyInfo, uremOp);

  return builder.create<mlir::vector::ExtractElementOp>(loc, varg0, uremOp);
}

// VEC_INSERT
fir::ExtendedValue
PPCIntrinsicLibrary::genVecInsert(mlir::Type resultType,
                                  llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  auto argBases{getBasesForArgs(args)};
  auto argTypes{getTypesForArgs(argBases)};
  auto vecTyInfo{getVecTypeFromFir(argBases[1])};
  auto mlirTy{vecTyInfo.toMlirVectorType(builder.getContext())};
  auto varg1{builder.createConvert(loc, mlirTy, argBases[1])};

  auto numEle{builder.createIntegerConstant(loc, argTypes[2], vecTyInfo.len)};
  mlir::Value uremOp{
      builder.create<mlir::LLVM::URemOp>(loc, argBases[2], numEle)};

  if (!isNativeVecElemOrderOnLE())
    uremOp = convertVectorElementOrder(builder, loc, vecTyInfo, uremOp);

  auto res{builder.create<mlir::vector::InsertElementOp>(loc, argBases[0],
                                                         varg1, uremOp)};
  return builder.create<fir::ConvertOp>(loc, vecTyInfo.toFirVectorType(), res);
}

// VEC_MERGEH, VEC_MERGEL
template <VecOp vop>
fir::ExtendedValue
PPCIntrinsicLibrary::genVecMerge(mlir::Type resultType,
                                 llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  auto argBases{getBasesForArgs(args)};
  auto vecTyInfo{getVecTypeFromFir(argBases[0])};
  llvm::SmallVector<int64_t, 16> mMask; // native vector element order mask
  llvm::SmallVector<int64_t, 16> rMask; // non-native vector element order mask

  switch (vop) {
  case VecOp::Mergeh: {
    switch (vecTyInfo.len) {
    case 2: {
      enum { V1 = 0, V2 = 2 };
      mMask = {V1 + 0, V2 + 0};
      rMask = {V2 + 1, V1 + 1};
      break;
    }
    case 4: {
      enum { V1 = 0, V2 = 4 };
      mMask = {V1 + 0, V2 + 0, V1 + 1, V2 + 1};
      rMask = {V2 + 2, V1 + 2, V2 + 3, V1 + 3};
      break;
    }
    case 8: {
      enum { V1 = 0, V2 = 8 };
      mMask = {V1 + 0, V2 + 0, V1 + 1, V2 + 1, V1 + 2, V2 + 2, V1 + 3, V2 + 3};
      rMask = {V2 + 4, V1 + 4, V2 + 5, V1 + 5, V2 + 6, V1 + 6, V2 + 7, V1 + 7};
      break;
    }
    case 16:
      mMask = {0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13,
               0x04, 0x14, 0x05, 0x15, 0x06, 0x16, 0x07, 0x17};
      rMask = {0x18, 0x08, 0x19, 0x09, 0x1A, 0x0A, 0x1B, 0x0B,
               0x1C, 0x0C, 0x1D, 0x0D, 0x1E, 0x0E, 0x1F, 0x0F};
      break;
    default:
      llvm_unreachable("unexpected vector length");
    }
    break;
  }
  case VecOp::Mergel: {
    switch (vecTyInfo.len) {
    case 2: {
      enum { V1 = 0, V2 = 2 };
      mMask = {V1 + 1, V2 + 1};
      rMask = {V2 + 0, V1 + 0};
      break;
    }
    case 4: {
      enum { V1 = 0, V2 = 4 };
      mMask = {V1 + 2, V2 + 2, V1 + 3, V2 + 3};
      rMask = {V2 + 0, V1 + 0, V2 + 1, V1 + 1};
      break;
    }
    case 8: {
      enum { V1 = 0, V2 = 8 };
      mMask = {V1 + 4, V2 + 4, V1 + 5, V2 + 5, V1 + 6, V2 + 6, V1 + 7, V2 + 7};
      rMask = {V2 + 0, V1 + 0, V2 + 1, V1 + 1, V2 + 2, V1 + 2, V2 + 3, V1 + 3};
      break;
    }
    case 16:
      mMask = {0x08, 0x18, 0x09, 0x19, 0x0A, 0x1A, 0x0B, 0x1B,
               0x0C, 0x1C, 0x0D, 0x1D, 0x0E, 0x1E, 0x0F, 0x1F};
      rMask = {0x10, 0x00, 0x11, 0x01, 0x12, 0x02, 0x13, 0x03,
               0x14, 0x04, 0x15, 0x05, 0x16, 0x06, 0x17, 0x07};
      break;
    default:
      llvm_unreachable("unexpected vector length");
    }
    break;
  }
  default:
    llvm_unreachable("invalid vector operation for generator");
  }

  auto vargs{convertVecArgs(builder, loc, vecTyInfo, argBases)};

  llvm::SmallVector<int64_t, 16> &mergeMask =
      (isBEVecElemOrderOnLE()) ? rMask : mMask;

  auto callOp{builder.create<mlir::vector::ShuffleOp>(loc, vargs[0], vargs[1],
                                                      mergeMask)};
  return builder.createConvert(loc, resultType, callOp);
}

// VEC_NMADD, VEC_MSUB
template <VecOp vop>
fir::ExtendedValue
PPCIntrinsicLibrary::genVecNmaddMsub(mlir::Type resultType,
                                     llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  auto context{builder.getContext()};
  auto argBases{getBasesForArgs(args)};
  auto vTypeInfo{getVecTypeFromFir(argBases[0])};
  auto newArgs{convertVecArgs(builder, loc, vTypeInfo, argBases)};
  const auto width{vTypeInfo.eleTy.getIntOrFloatBitWidth()};

  static std::map<int, std::pair<llvm::StringRef, mlir::FunctionType>> fmaMap{
      {32,
       std::make_pair(
           "llvm.fma.v4f32",
           genFuncType<Ty::RealVector<4>, Ty::RealVector<4>, Ty::RealVector<4>>(
               context, builder))},
      {64,
       std::make_pair(
           "llvm.fma.v2f64",
           genFuncType<Ty::RealVector<8>, Ty::RealVector<8>, Ty::RealVector<8>>(
               context, builder))}};

  auto funcOp{builder.addNamedFunction(loc, std::get<0>(fmaMap[width]),
                                       std::get<1>(fmaMap[width]))};
  if (vop == VecOp::Nmadd) {
    // vec_nmadd(arg1, arg2, arg3) = -fma(arg1, arg2, arg3)
    auto callOp{builder.create<fir::CallOp>(loc, funcOp, newArgs)};

    // We need to convert fir.vector to MLIR vector to use fneg and then back
    // to fir.vector to store.
    auto vCall{builder.createConvert(loc, vTypeInfo.toMlirVectorType(context),
                                     callOp.getResult(0))};
    auto neg{builder.create<mlir::arith::NegFOp>(loc, vCall)};
    return builder.createConvert(loc, vTypeInfo.toFirVectorType(), neg);
  } else if (vop == VecOp::Msub) {
    // vec_msub(arg1, arg2, arg3) = fma(arg1, arg2, -arg3)
    newArgs[2] = builder.create<mlir::arith::NegFOp>(loc, newArgs[2]);

    auto callOp{builder.create<fir::CallOp>(loc, funcOp, newArgs)};
    return callOp.getResult(0);
  }
  llvm_unreachable("Invalid vector operation for generator");
}

// VEC_PERM, VEC_PERMI
template <VecOp vop>
fir::ExtendedValue
PPCIntrinsicLibrary::genVecPerm(mlir::Type resultType,
                                llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  auto context{builder.getContext()};
  auto argBases{getBasesForArgs(args)};
  auto argTypes{getTypesForArgs(argBases)};
  auto vecTyInfo{getVecTypeFromFir(argBases[0])};
  auto mlirTy{vecTyInfo.toMlirVectorType(context)};

  auto vi32Ty{mlir::VectorType::get(4, mlir::IntegerType::get(context, 32))};
  auto vf64Ty{mlir::VectorType::get(2, mlir::FloatType::getF64(context))};

  auto mArg0{builder.createConvert(loc, mlirTy, argBases[0])};
  auto mArg1{builder.createConvert(loc, mlirTy, argBases[1])};

  switch (vop) {
  case VecOp::Perm: {
    VecTypeInfo maskVecTyInfo{getVecTypeFromFir(argBases[2])};
    auto mlirMaskTy{maskVecTyInfo.toMlirVectorType(context)};
    auto mMask{builder.createConvert(loc, mlirMaskTy, argBases[2])};

    if (mlirTy != vi32Ty) {
      mArg0 =
          builder.create<mlir::LLVM::BitcastOp>(loc, vi32Ty, mArg0).getResult();
      mArg1 =
          builder.create<mlir::LLVM::BitcastOp>(loc, vi32Ty, mArg1).getResult();
    }

    auto funcOp{builder.addNamedFunction(
        loc, "llvm.ppc.altivec.vperm",
        genFuncType<Ty::IntegerVector<4>, Ty::IntegerVector<4>,
                    Ty::IntegerVector<4>, Ty::IntegerVector<1>>(context,
                                                                builder))};

    llvm::SmallVector<mlir::Value> newArgs;
    if (isNativeVecElemOrderOnLE()) {
      auto i8Ty{mlir::IntegerType::get(context, 8)};
      auto v8Ty{mlir::VectorType::get(16, i8Ty)};
      auto negOne{builder.createIntegerConstant(loc, i8Ty, -1)};
      auto vNegOne{
          builder.create<mlir::vector::BroadcastOp>(loc, v8Ty, negOne)};

      mMask = builder.create<mlir::arith::XOrIOp>(loc, mMask, vNegOne);
      newArgs = {mArg1, mArg0, mMask};
    } else {
      newArgs = {mArg0, mArg1, mMask};
    }

    auto res{builder.create<fir::CallOp>(loc, funcOp, newArgs).getResult(0)};

    if (res.getType() != argTypes[0]) {
      // fir.call llvm.ppc.altivec.vperm returns !fir.vector<i4:32>
      // convert the result back to the original type
      res = builder.createConvert(loc, vi32Ty, res);
      if (mlirTy != vi32Ty)
        res =
            builder.create<mlir::LLVM::BitcastOp>(loc, mlirTy, res).getResult();
    }
    return builder.createConvert(loc, resultType, res);
  }
  case VecOp::Permi: {
    // arg3 is a constant
    auto constIntOp{
        mlir::dyn_cast<mlir::arith::ConstantOp>(argBases[2].getDefiningOp())
            .getValue()
            .dyn_cast_or_null<mlir::IntegerAttr>()};
    assert(constIntOp && "expected integer constant argument");
    auto constInt{constIntOp.getInt()};
    // arg1, arg2, and result type share same VecTypeInfo
    if (vecTyInfo.isFloat()) {
      mArg0 =
          builder.create<mlir::LLVM::BitcastOp>(loc, vf64Ty, mArg0).getResult();
      mArg1 =
          builder.create<mlir::LLVM::BitcastOp>(loc, vf64Ty, mArg1).getResult();
    }

    llvm::SmallVector<int64_t, 2> nMask; // native vector element order mask
    llvm::SmallVector<int64_t, 2> rMask; // non-native vector element order mask
    enum { V1 = 0, V2 = 2 };
    switch (constInt) {
    case 0:
      nMask = {V1 + 0, V2 + 0};
      rMask = {V2 + 1, V1 + 1};
      break;
    case 1:
      nMask = {V1 + 0, V2 + 1};
      rMask = {V2 + 0, V1 + 1};
      break;
    case 2:
      nMask = {V1 + 1, V2 + 0};
      rMask = {V2 + 1, V1 + 0};
      break;
    case 3:
      nMask = {V1 + 1, V2 + 1};
      rMask = {V2 + 0, V1 + 0};
      break;
    default:
      llvm_unreachable("unexpected arg3 value for vec_permi");
    }

    llvm::SmallVector<int64_t, 2> mask =
        (isBEVecElemOrderOnLE()) ? rMask : nMask;
    auto res{builder.create<mlir::vector::ShuffleOp>(loc, mArg0, mArg1, mask)};
    if (res.getType() != mlirTy) {
      auto cast{builder.create<mlir::LLVM::BitcastOp>(loc, mlirTy, res)};
      return builder.createConvert(loc, resultType, cast);
    }
    return builder.createConvert(loc, resultType, res);
  }
  default:
    llvm_unreachable("invalid vector operation for generator");
  }
}

// VEC_SEL
fir::ExtendedValue
PPCIntrinsicLibrary::genVecSel(mlir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  auto argBases{getBasesForArgs(args)};
  llvm::SmallVector<VecTypeInfo, 4> vecTyInfos;
  for (size_t i = 0; i < argBases.size(); i++) {
    vecTyInfos.push_back(getVecTypeFromFir(argBases[i]));
  }
  auto vargs{convertVecArgs(builder, loc, vecTyInfos, argBases)};

  auto i8Ty{mlir::IntegerType::get(builder.getContext(), 8)};
  auto negOne{builder.createIntegerConstant(loc, i8Ty, -1)};

  // construct a constant <16 x i8> vector with value -1 for bitcast
  auto bcVecTy{mlir::VectorType::get(16, i8Ty)};
  auto vNegOne{builder.create<mlir::vector::BroadcastOp>(loc, bcVecTy, negOne)};

  // bitcast arguments to bcVecTy
  auto arg1{builder.create<mlir::vector::BitCastOp>(loc, bcVecTy, vargs[0])};
  auto arg2{builder.create<mlir::vector::BitCastOp>(loc, bcVecTy, vargs[1])};
  auto arg3{builder.create<mlir::vector::BitCastOp>(loc, bcVecTy, vargs[2])};

  // vec_sel(arg1, arg2, arg3) =
  //   (arg2 and arg3) or (arg1 and (arg3 xor vector(-1,...)))
  auto comp{builder.create<mlir::arith::XOrIOp>(loc, arg3, vNegOne)};
  auto a1AndComp{builder.create<mlir::arith::AndIOp>(loc, arg1, comp)};
  auto a1OrA2{builder.create<mlir::arith::AndIOp>(loc, arg2, arg3)};
  auto res{builder.create<mlir::arith::OrIOp>(loc, a1AndComp, a1OrA2)};

  auto bcRes{
      builder.create<mlir::vector::BitCastOp>(loc, vargs[0].getType(), res)};

  return builder.createConvert(loc, vecTyInfos[0].toFirVectorType(), bcRes);
}

// VEC_SL, VEC_SLD, VEC_SLDW, VEC_SLL, VEC_SLO, VEC_SR, VEC_SRL, VEC_SRO
template <VecOp vop>
fir::ExtendedValue
PPCIntrinsicLibrary::genVecShift(mlir::Type resultType,
                                 llvm::ArrayRef<fir::ExtendedValue> args) {
  auto context{builder.getContext()};
  auto argBases{getBasesForArgs(args)};
  auto argTypes{getTypesForArgs(argBases)};

  llvm::SmallVector<VecTypeInfo, 2> vecTyInfoArgs;
  vecTyInfoArgs.push_back(getVecTypeFromFir(argBases[0]));
  vecTyInfoArgs.push_back(getVecTypeFromFir(argBases[1]));

  // Convert the first two arguments to MLIR vectors
  llvm::SmallVector<mlir::Type, 2> mlirTyArgs;
  mlirTyArgs.push_back(vecTyInfoArgs[0].toMlirVectorType(context));
  mlirTyArgs.push_back(vecTyInfoArgs[1].toMlirVectorType(context));

  llvm::SmallVector<mlir::Value, 2> mlirVecArgs;
  mlirVecArgs.push_back(builder.createConvert(loc, mlirTyArgs[0], argBases[0]));
  mlirVecArgs.push_back(builder.createConvert(loc, mlirTyArgs[1], argBases[1]));

  mlir::Value shftRes{nullptr};

  if (vop == VecOp::Sl || vop == VecOp::Sr) {
    assert(args.size() == 2);
    // Construct the mask
    auto width{
        mlir::dyn_cast<mlir::IntegerType>(vecTyInfoArgs[1].eleTy).getWidth()};
    auto vecVal{builder.createIntegerConstant(
        loc, getConvertedElementType(context, vecTyInfoArgs[0].eleTy), width)};
    auto mask{
        builder.create<mlir::vector::BroadcastOp>(loc, mlirTyArgs[1], vecVal)};
    auto shft{builder.create<mlir::arith::RemUIOp>(loc, mlirVecArgs[1], mask)};

    mlir::Value res{nullptr};
    if (vop == VecOp::Sr)
      res = builder.create<mlir::arith::ShRUIOp>(loc, mlirVecArgs[0], shft);
    else if (vop == VecOp::Sl)
      res = builder.create<mlir::arith::ShLIOp>(loc, mlirVecArgs[0], shft);

    shftRes = builder.createConvert(loc, argTypes[0], res);
  } else if (vop == VecOp::Sll || vop == VecOp::Slo || vop == VecOp::Srl ||
             vop == VecOp::Sro) {
    assert(args.size() == 2);

    // Bitcast to vector<4xi32>
    auto bcVecTy{mlir::VectorType::get(4, builder.getIntegerType(32))};
    if (mlirTyArgs[0] != bcVecTy)
      mlirVecArgs[0] =
          builder.create<mlir::vector::BitCastOp>(loc, bcVecTy, mlirVecArgs[0]);
    if (mlirTyArgs[1] != bcVecTy)
      mlirVecArgs[1] =
          builder.create<mlir::vector::BitCastOp>(loc, bcVecTy, mlirVecArgs[1]);

    llvm::StringRef funcName;
    switch (vop) {
    case VecOp::Srl:
      funcName = "llvm.ppc.altivec.vsr";
      break;
    case VecOp::Sro:
      funcName = "llvm.ppc.altivec.vsro";
      break;
    case VecOp::Sll:
      funcName = "llvm.ppc.altivec.vsl";
      break;
    case VecOp::Slo:
      funcName = "llvm.ppc.altivec.vslo";
      break;
    default:
      llvm_unreachable("unknown vector shift operation");
    }
    auto funcTy{genFuncType<Ty::IntegerVector<4>, Ty::IntegerVector<4>,
                            Ty::IntegerVector<4>>(context, builder)};
    mlir::func::FuncOp funcOp{builder.addNamedFunction(loc, funcName, funcTy)};
    auto callOp{builder.create<fir::CallOp>(loc, funcOp, mlirVecArgs)};

    // If the result vector type is different from the original type, need
    // to convert to mlir vector, bitcast and then convert back to fir vector.
    if (callOp.getResult(0).getType() != argTypes[0]) {
      auto res = builder.createConvert(loc, bcVecTy, callOp.getResult(0));
      res = builder.create<mlir::vector::BitCastOp>(loc, mlirTyArgs[0], res);
      shftRes = builder.createConvert(loc, argTypes[0], res);
    } else {
      shftRes = callOp.getResult(0);
    }
  } else if (vop == VecOp::Sld || vop == VecOp::Sldw) {
    assert(args.size() == 3);
    auto constIntOp =
        mlir::dyn_cast<mlir::arith::ConstantOp>(argBases[2].getDefiningOp())
            .getValue()
            .dyn_cast_or_null<mlir::IntegerAttr>();
    assert(constIntOp && "expected integer constant argument");

    // Bitcast to vector<16xi8>
    auto vi8Ty{mlir::VectorType::get(16, builder.getIntegerType(8))};
    if (mlirTyArgs[0] != vi8Ty) {
      mlirVecArgs[0] =
          builder.create<mlir::LLVM::BitcastOp>(loc, vi8Ty, mlirVecArgs[0])
              .getResult();
      mlirVecArgs[1] =
          builder.create<mlir::LLVM::BitcastOp>(loc, vi8Ty, mlirVecArgs[1])
              .getResult();
    }

    // Construct the mask for shuffling
    auto shiftVal{constIntOp.getInt()};
    if (vop == VecOp::Sldw)
      shiftVal = shiftVal << 2;
    shiftVal &= 0xF;
    llvm::SmallVector<int64_t, 16> mask;
    for (int i = 16; i < 32; ++i)
      mask.push_back(i - shiftVal);

    // Shuffle with mask
    shftRes = builder.create<mlir::vector::ShuffleOp>(loc, mlirVecArgs[1],
                                                      mlirVecArgs[0], mask);

    // Bitcast to the original type
    if (shftRes.getType() != mlirTyArgs[0])
      shftRes =
          builder.create<mlir::LLVM::BitcastOp>(loc, mlirTyArgs[0], shftRes);

    return builder.createConvert(loc, resultType, shftRes);
  } else
    llvm_unreachable("Invalid vector operation for generator");

  return shftRes;
}

const char *getMmaIrIntrName(MMAOp mmaOp) {
  switch (mmaOp) {
  case MMAOp::AssembleAcc:
    return "llvm.ppc.mma.assemble.acc";
  case MMAOp::AssemblePair:
    return "llvm.ppc.vsx.assemble.pair";
  case MMAOp::DisassembleAcc:
    return "llvm.ppc.mma.disassemble.acc";
  case MMAOp::DisassemblePair:
    return "llvm.ppc.vsx.disassemble.pair";
  }
  llvm_unreachable("getMmaIrIntrName");
}

mlir::FunctionType getMmaIrFuncType(mlir::MLIRContext *context, MMAOp mmaOp) {
  switch (mmaOp) {
  case MMAOp::AssembleAcc:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 4);
  case MMAOp::AssemblePair:
    return genMmaVpFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::DisassembleAcc:
    return genMmaDisassembleFuncType(context, mmaOp);
  case MMAOp::DisassemblePair:
    return genMmaDisassembleFuncType(context, mmaOp);
  }
  llvm_unreachable("getMmaIrFuncType");
}

template <MMAOp IntrId, MMAHandlerOp HandlerOp>
void PPCIntrinsicLibrary::genMmaIntr(llvm::ArrayRef<fir::ExtendedValue> args) {
  auto context{builder.getContext()};
  mlir::FunctionType intrFuncType{getMmaIrFuncType(context, IntrId)};
  mlir::func::FuncOp funcOp{
      builder.addNamedFunction(loc, getMmaIrIntrName(IntrId), intrFuncType)};
  llvm::SmallVector<mlir::Value> intrArgs;

  // Depending on SubToFunc, change the subroutine call to a function call.
  // First argument represents the result. Rest of the arguments
  // are shifted one position to form the actual argument list.
  size_t argStart{0};
  size_t argStep{1};
  size_t e{args.size()};
  if (HandlerOp == MMAHandlerOp::SubToFunc) {
    // The first argument becomes function result. Start from the second
    // argument.
    argStart = 1;
  } else if (HandlerOp == MMAHandlerOp::SubToFuncReverseArgOnLE) {
    // Reverse argument order on little-endian target only.
    // The reversal does not depend on the setting of non-native-order option.
    if (Fortran::evaluate::isHostLittleEndian) {
      // Load the arguments in reverse order.
      argStart = args.size() - 1;
      // The first argument becomes function result. Stop at the second
      // argument.
      e = 0;
      argStep = -1;
    } else {
      // Load the arguments in natural order.
      // The first argument becomes function result. Start from the second
      // argument.
      argStart = 1;
    }
  }

  for (size_t i = argStart, j = 0; i != e; i += argStep, ++j) {
    auto v{fir::getBase(args[i])};
    if (i == 0 && HandlerOp == MMAHandlerOp::FirstArgIsResult) {
      // First argument is passed in as an address. We need to load
      // the content to match the LLVM interface.
      v = builder.create<fir::LoadOp>(loc, v);
    }
    auto vType{v.getType()};
    mlir::Type targetType{intrFuncType.getInput(j)};
    if (vType != targetType) {
      if (targetType.isa<mlir::VectorType>()) {
        // Perform vector type conversion for arguments passed by value.
        auto eleTy{vType.dyn_cast<fir::VectorType>().getEleTy()};
        auto len{vType.dyn_cast<fir::VectorType>().getLen()};
        mlir::VectorType mlirType = mlir::VectorType::get(len, eleTy);
        auto v0{builder.createConvert(loc, mlirType, v)};
        auto v1{builder.create<mlir::vector::BitCastOp>(loc, targetType, v0)};
        intrArgs.push_back(v1);
      } else if (targetType.isa<mlir::IntegerType>() &&
                 vType.isa<mlir::IntegerType>()) {
        auto v0{builder.createConvert(loc, targetType, v)};
        intrArgs.push_back(v0);
      } else {
        llvm::errs() << "\nUnexpected type conversion requested: "
                     << " from " << vType << " to " << targetType << "\n";
        llvm_unreachable("Unsupported type conversion for argument to PowerPC "
                         "MMA intrinsic");
      }
    } else {
      intrArgs.push_back(v);
    }
  }
  auto callSt{builder.create<fir::CallOp>(loc, funcOp, intrArgs)};
  if (HandlerOp == MMAHandlerOp::SubToFunc ||
      HandlerOp == MMAHandlerOp::SubToFuncReverseArgOnLE ||
      HandlerOp == MMAHandlerOp::FirstArgIsResult) {
    // Convert pointer type if needed.
    mlir::Value callResult{callSt.getResult(0)};
    mlir::Value destPtr{fir::getBase(args[0])};
    mlir::Type callResultPtrType{builder.getRefType(callResult.getType())};
    if (destPtr.getType() != callResultPtrType) {
      destPtr = builder.create<fir::ConvertOp>(loc, callResultPtrType, destPtr);
    }
    // Copy the result.
    builder.create<fir::StoreOp>(loc, callResult, destPtr);
  }
}

static mlir::Value addOffsetToAddress(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value baseAddr,
                                      mlir::Value offset) {
  auto typeExtent{fir::SequenceType::getUnknownExtent()};
  // Construct an !fir.ref<!ref.array<?xi8>> type
  auto arrRefTy{builder.getRefType(fir::SequenceType::get(
      {typeExtent}, mlir::IntegerType::get(builder.getContext(), 8)))};
  // Convert arg to !fir.ref<!ref.array<?xi8>>
  auto resAddr{builder.create<fir::ConvertOp>(loc, arrRefTy, baseAddr)};

  return builder.create<fir::CoordinateOp>(loc, arrRefTy, resAddr, offset);
}

static mlir::Value reverseVectorElements(fir::FirOpBuilder &builder,
                                         mlir::Location loc, mlir::Value v,
                                         int64_t len) {
  assert(v.getType().isa<mlir::VectorType>());
  assert(len > 0);
  llvm::SmallVector<int64_t, 16> mask;
  for (int64_t i = 0; i < len; ++i) {
    mask.push_back(len - 1 - i);
  }

  auto undefVec{builder.create<fir::UndefOp>(loc, v.getType())};
  return builder.create<mlir::vector::ShuffleOp>(loc, v, undefVec, mask);
}

// VEC_ST, VEC_STE
template <VecOp vop>
void PPCIntrinsicLibrary::genVecStore(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);

  auto context{builder.getContext()};
  auto argBases{getBasesForArgs(args)};
  auto arg1TyInfo{getVecTypeFromFir(argBases[0])};

  auto addr{addOffsetToAddress(builder, loc, argBases[2], argBases[1])};

  llvm::StringRef fname{};
  mlir::VectorType stTy{nullptr};
  auto i32ty{mlir::IntegerType::get(context, 32)};
  switch (vop) {
  case VecOp::St:
    stTy = mlir::VectorType::get(4, i32ty);
    fname = "llvm.ppc.altivec.stvx";
    break;
  case VecOp::Ste: {
    const auto width{arg1TyInfo.eleTy.getIntOrFloatBitWidth()};
    const auto len{arg1TyInfo.len};

    if (arg1TyInfo.isFloat32()) {
      stTy = mlir::VectorType::get(len, i32ty);
      fname = "llvm.ppc.altivec.stvewx";
    } else if (arg1TyInfo.eleTy.isa<mlir::IntegerType>()) {
      stTy = mlir::VectorType::get(len, mlir::IntegerType::get(context, width));

      switch (width) {
      case 8:
        fname = "llvm.ppc.altivec.stvebx";
        break;
      case 16:
        fname = "llvm.ppc.altivec.stvehx";
        break;
      case 32:
        fname = "llvm.ppc.altivec.stvewx";
        break;
      default:
        assert(false && "invalid element size");
      }
    } else
      assert(false && "unknown type");
    break;
  }
  case VecOp::Stxvp:
    // __vector_pair type
    stTy = mlir::VectorType::get(256, mlir::IntegerType::get(context, 1));
    fname = "llvm.ppc.vsx.stxvp";
    break;
  default:
    llvm_unreachable("invalid vector operation for generator");
  }

  auto funcType{
      mlir::FunctionType::get(context, {stTy, addr.getType()}, std::nullopt)};
  mlir::func::FuncOp funcOp = builder.addNamedFunction(loc, fname, funcType);

  llvm::SmallVector<mlir::Value, 4> biArgs;

  if (vop == VecOp::Stxvp) {
    biArgs.push_back(argBases[0]);
    biArgs.push_back(addr);
    builder.create<fir::CallOp>(loc, funcOp, biArgs);
    return;
  }

  auto vecTyInfo{getVecTypeFromFirType(argBases[0].getType())};
  auto cnv{builder.createConvert(loc, vecTyInfo.toMlirVectorType(context),
                                 argBases[0])};

  mlir::Value newArg1{nullptr};
  if (stTy != arg1TyInfo.toMlirVectorType(context))
    newArg1 = builder.create<mlir::vector::BitCastOp>(loc, stTy, cnv);
  else
    newArg1 = cnv;

  if (isBEVecElemOrderOnLE())
    newArg1 = builder.createConvert(
        loc, stTy, reverseVectorElements(builder, loc, newArg1, 4));

  biArgs.push_back(newArg1);
  biArgs.push_back(addr);

  builder.create<fir::CallOp>(loc, funcOp, biArgs);
}

static mlir::NamedAttribute getAlignmentAttr(fir::FirOpBuilder &builder,
                                             const int val) {
  auto i64ty{mlir::IntegerType::get(builder.getContext(), 64)};
  auto alignAttr{mlir::IntegerAttr::get(i64ty, val)};
  return builder.getNamedAttr("alignment", alignAttr);
}

// VEC_XST, VEC_XST_BE, VEC_STXV, VEC_XSTD2, VEC_XSTW4
template <VecOp vop>
void PPCIntrinsicLibrary::genVecXStore(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  auto context{builder.getContext()};
  auto argBases{getBasesForArgs(args)};
  VecTypeInfo arg1TyInfo{getVecTypeFromFir(argBases[0])};

  auto addr{addOffsetToAddress(builder, loc, argBases[2], argBases[1])};

  mlir::Value trg{nullptr};
  mlir::Value src{nullptr};

  switch (vop) {
  case VecOp::Xst:
  case VecOp::Xst_be: {
    src = argBases[0];
    trg = builder.createConvert(loc, builder.getRefType(argBases[0].getType()),
                                addr);

    if (vop == VecOp::Xst_be || isBEVecElemOrderOnLE()) {
      auto cnv{builder.createConvert(loc, arg1TyInfo.toMlirVectorType(context),
                                     argBases[0])};
      auto shf{reverseVectorElements(builder, loc, cnv, arg1TyInfo.len)};

      src = builder.createConvert(loc, arg1TyInfo.toFirVectorType(), shf);
    }
    break;
  }
  case VecOp::Xstd2:
  case VecOp::Xstw4: {
    // an 16-byte vector arg1 is treated as two 8-byte elements or
    // four 4-byte elements
    mlir::IntegerType elemTy;
    uint64_t numElem = (vop == VecOp::Xstd2) ? 2 : 4;
    elemTy = builder.getIntegerType(128 / numElem);

    mlir::VectorType mlirVecTy{mlir::VectorType::get(numElem, elemTy)};
    fir::VectorType firVecTy{fir::VectorType::get(numElem, elemTy)};

    auto cnv{builder.createConvert(loc, arg1TyInfo.toMlirVectorType(context),
                                   argBases[0])};

    mlir::Type srcTy{nullptr};
    if (numElem != arg1TyInfo.len) {
      cnv = builder.create<mlir::vector::BitCastOp>(loc, mlirVecTy, cnv);
      srcTy = firVecTy;
    } else {
      srcTy = arg1TyInfo.toFirVectorType();
    }

    trg = builder.createConvert(loc, builder.getRefType(srcTy), addr);

    if (isBEVecElemOrderOnLE()) {
      cnv = reverseVectorElements(builder, loc, cnv, numElem);
    }

    src = builder.createConvert(loc, srcTy, cnv);
    break;
  }
  case VecOp::Stxv:
    src = argBases[0];
    trg = builder.createConvert(loc, builder.getRefType(argBases[0].getType()),
                                addr);
    break;
  default:
    assert(false && "Invalid vector operation for generator");
  }
  builder.create<fir::StoreOp>(loc, mlir::TypeRange{},
                               mlir::ValueRange{src, trg},
                               getAlignmentAttr(builder, 1));
}

} // namespace fir
