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
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace fir {

using PI = PPCIntrinsicLibrary;

// PPC specific intrinsic handlers.
static constexpr IntrinsicHandler ppcHandlers[]{
    {"__ppc_mtfsf",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(&PI::genMtfsf<false>),
     {{{"mask", asValue}, {"r", asValue}}},
     /*isElemental=*/false},
    {"__ppc_mtfsfi",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(&PI::genMtfsf<true>),
     {{{"bf", asValue}, {"i", asValue}}},
     /*isElemental=*/false},
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
    {"__ppc_vec_mul",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecAddAndMulSubXor<VecOp::Mul>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
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

} // namespace fir
