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

} // namespace fir
