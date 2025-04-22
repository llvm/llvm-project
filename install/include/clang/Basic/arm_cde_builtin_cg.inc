case ARM::BI__builtin_arm_cde_cx1d: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_cx1d), {Val0, Val1});
  Value *Val3 = Builder.CreateExtractValue(Val2, static_cast<unsigned>(1));
  Value *Val4 = Builder.CreateIntCast(Val3, Int64Ty, false);
  Value *Val5 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val6 = Builder.CreateShl(Val4, Val5);
  Value *Val7 = Builder.CreateExtractValue(Val2, static_cast<unsigned>(0));
  Value *Val8 = Builder.CreateIntCast(Val7, Int64Ty, false);
  return Builder.CreateOr(Val6, Val8);
}
case ARM::BI__builtin_arm_cde_vcx1d_u64: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_vcx1, {DoubleTy}), {Val0, Val1});
  llvm::Type *Val3 = Int64Ty;
  return Builder.CreateBitCast(Val2, Val3);
}
case ARM::BI__builtin_arm_cde_vcx1_u32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_vcx1, {FloatTy}), {Val0, Val1});
  llvm::Type *Val3 = Int32Ty;
  return Builder.CreateBitCast(Val2, Val3);
}
case ARM::BI__builtin_arm_cde_cx2d: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_cx2d), {Val0, Val1, Val2});
  Value *Val4 = Builder.CreateExtractValue(Val3, static_cast<unsigned>(1));
  Value *Val5 = Builder.CreateIntCast(Val4, Int64Ty, false);
  Value *Val6 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val7 = Builder.CreateShl(Val5, Val6);
  Value *Val8 = Builder.CreateExtractValue(Val3, static_cast<unsigned>(0));
  Value *Val9 = Builder.CreateIntCast(Val8, Int64Ty, false);
  return Builder.CreateOr(Val7, Val9);
}
case ARM::BI__builtin_arm_cde_cx3d: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = EmitScalarExpr(E->getArg(3));
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_cx3d), {Val0, Val1, Val2, Val3});
  Value *Val5 = Builder.CreateExtractValue(Val4, static_cast<unsigned>(1));
  Value *Val6 = Builder.CreateIntCast(Val5, Int64Ty, false);
  Value *Val7 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val8 = Builder.CreateShl(Val6, Val7);
  Value *Val9 = Builder.CreateExtractValue(Val4, static_cast<unsigned>(0));
  Value *Val10 = Builder.CreateIntCast(Val9, Int64Ty, false);
  return Builder.CreateOr(Val8, Val10);
}
case ARM::BI__builtin_arm_cde_vcx1q_m_f16:
case ARM::BI__builtin_arm_cde_vcx1q_m_f32:
case ARM::BI__builtin_arm_cde_vcx1q_m_s16:
case ARM::BI__builtin_arm_cde_vcx1q_m_s32:
case ARM::BI__builtin_arm_cde_vcx1q_m_s64:
case ARM::BI__builtin_arm_cde_vcx1q_m_s8:
case ARM::BI__builtin_arm_cde_vcx1q_m_u16:
case ARM::BI__builtin_arm_cde_vcx1q_m_u32:
case ARM::BI__builtin_arm_cde_vcx1q_m_u64:
case ARM::BI__builtin_arm_cde_vcx1q_m_u8:
case ARM::BI__builtin_arm_cde_vcx1qa_m_f16:
case ARM::BI__builtin_arm_cde_vcx1qa_m_f32:
case ARM::BI__builtin_arm_cde_vcx1qa_m_s16:
case ARM::BI__builtin_arm_cde_vcx1qa_m_s32:
case ARM::BI__builtin_arm_cde_vcx1qa_m_s64:
case ARM::BI__builtin_arm_cde_vcx1qa_m_s8:
case ARM::BI__builtin_arm_cde_vcx1qa_m_u16:
case ARM::BI__builtin_arm_cde_vcx1qa_m_u32:
case ARM::BI__builtin_arm_cde_vcx1qa_m_u64:
case ARM::BI__builtin_arm_cde_vcx1qa_m_u8: {
  llvm::Type * Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx1q_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx1q_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx1q_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx1q_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx1q_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx1q_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx1q_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx1q_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx1q_m_s64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = Intrinsic::arm_cde_vcx1q_predicated;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_cde_vcx1q_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_cde_vcx1q_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_cde_vcx1q_m_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx1q_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx1q_m_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx1q_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx1q_m_u64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = Intrinsic::arm_cde_vcx1q_predicated;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_cde_vcx1q_m_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_cde_vcx1q_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_cde_vcx1qa_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx1qa_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx1qa_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx1qa_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx1qa_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx1qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx1qa_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx1qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx1qa_m_s64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = Intrinsic::arm_cde_vcx1qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_cde_vcx1qa_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_cde_vcx1qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_cde_vcx1qa_m_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx1qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx1qa_m_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx1qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx1qa_m_u64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = Intrinsic::arm_cde_vcx1qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_cde_vcx1qa_m_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_cde_vcx1qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = EmitScalarExpr(E->getArg(3));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val4});
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2, Param0}), {Val0, Val1, Val2, Val5});
}
case ARM::BI__builtin_arm_cde_vcx2q_m_impl_f16:
case ARM::BI__builtin_arm_cde_vcx2q_m_impl_f32:
case ARM::BI__builtin_arm_cde_vcx2q_m_impl_s16:
case ARM::BI__builtin_arm_cde_vcx2q_m_impl_s32:
case ARM::BI__builtin_arm_cde_vcx2q_m_impl_s64:
case ARM::BI__builtin_arm_cde_vcx2q_m_impl_s8:
case ARM::BI__builtin_arm_cde_vcx2q_m_impl_u16:
case ARM::BI__builtin_arm_cde_vcx2q_m_impl_u32:
case ARM::BI__builtin_arm_cde_vcx2q_m_impl_u64:
case ARM::BI__builtin_arm_cde_vcx2q_m_impl_u8:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_f16:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_f32:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_s16:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_s32:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_s64:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_s8:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_u16:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_u32:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_u64:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_u8: {
  llvm::Type * Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx2q_m_impl_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx2q_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_m_impl_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx2q_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_m_impl_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx2q_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_m_impl_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx2q_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_m_impl_s64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = Intrinsic::arm_cde_vcx2q_predicated;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_m_impl_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_cde_vcx2q_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_m_impl_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx2q_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_m_impl_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx2q_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_m_impl_u64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = Intrinsic::arm_cde_vcx2q_predicated;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_m_impl_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_cde_vcx2q_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx2qa_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx2qa_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx2qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx2qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_s64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = Intrinsic::arm_cde_vcx2qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_cde_vcx2qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx2qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx2qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_u64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = Intrinsic::arm_cde_vcx2qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_cde_vcx2qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = EmitScalarExpr(E->getArg(3));
  Value *Val4 = EmitScalarExpr(E->getArg(4));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2, Param0}), {Val0, Val1, Val2, Val3, Val6});
}
case ARM::BI__builtin_arm_cde_vcx3q_m_impl_f16:
case ARM::BI__builtin_arm_cde_vcx3q_m_impl_f32:
case ARM::BI__builtin_arm_cde_vcx3q_m_impl_s16:
case ARM::BI__builtin_arm_cde_vcx3q_m_impl_s32:
case ARM::BI__builtin_arm_cde_vcx3q_m_impl_s64:
case ARM::BI__builtin_arm_cde_vcx3q_m_impl_s8:
case ARM::BI__builtin_arm_cde_vcx3q_m_impl_u16:
case ARM::BI__builtin_arm_cde_vcx3q_m_impl_u32:
case ARM::BI__builtin_arm_cde_vcx3q_m_impl_u64:
case ARM::BI__builtin_arm_cde_vcx3q_m_impl_u8:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_f16:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_f32:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_s16:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_s32:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_s64:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_s8:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_u16:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_u32:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_u64:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_u8: {
  llvm::Type * Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx3q_m_impl_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx3q_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_m_impl_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx3q_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_m_impl_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx3q_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_m_impl_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx3q_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_m_impl_s64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = Intrinsic::arm_cde_vcx3q_predicated;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_m_impl_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_cde_vcx3q_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_m_impl_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx3q_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_m_impl_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx3q_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_m_impl_u64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = Intrinsic::arm_cde_vcx3q_predicated;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_m_impl_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_cde_vcx3q_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx3qa_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx3qa_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx3qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx3qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_s64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = Intrinsic::arm_cde_vcx3qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_cde_vcx3qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_cde_vcx3qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_cde_vcx3qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_u64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = Intrinsic::arm_cde_vcx3qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_cde_vcx3qa_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = EmitScalarExpr(E->getArg(3));
  Value *Val4 = EmitScalarExpr(E->getArg(4));
  Value *Val5 = EmitScalarExpr(E->getArg(5));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2, Param0}), {Val0, Val1, Val2, Val3, Val4, Val7});
}
case ARM::BI__builtin_arm_cde_cx3a: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = EmitScalarExpr(E->getArg(3));
  Value *Val4 = EmitScalarExpr(E->getArg(4));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_cx3a), {Val0, Val1, Val2, Val3, Val4});
}
case ARM::BI__builtin_arm_cde_cx2a:
case ARM::BI__builtin_arm_cde_cx3: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_cx2a:
    Param0 = Intrinsic::arm_cde_cx2a;
    break;
  case ARM::BI__builtin_arm_cde_cx3:
    Param0 = Intrinsic::arm_cde_cx3;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = EmitScalarExpr(E->getArg(3));
  return Builder.CreateCall(CGM.getIntrinsic(Param0), {Val0, Val1, Val2, Val3});
}
case ARM::BI__builtin_arm_cde_cx1a:
case ARM::BI__builtin_arm_cde_cx2: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_cx1a:
    Param0 = Intrinsic::arm_cde_cx1a;
    break;
  case ARM::BI__builtin_arm_cde_cx2:
    Param0 = Intrinsic::arm_cde_cx2;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  return Builder.CreateCall(CGM.getIntrinsic(Param0), {Val0, Val1, Val2});
}
case ARM::BI__builtin_arm_cde_vcx1da_u64:
case ARM::BI__builtin_arm_cde_vcx2d_u64: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx1da_u64:
    Param0 = Intrinsic::arm_cde_vcx1a;
    break;
  case ARM::BI__builtin_arm_cde_vcx2d_u64:
    Param0 = Intrinsic::arm_cde_vcx2;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = DoubleTy;
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Param0, {DoubleTy}), {Val0, Val3, Val4});
  llvm::Type *Val6 = Int64Ty;
  return Builder.CreateBitCast(Val5, Val6);
}
case ARM::BI__builtin_arm_cde_vcx2da_u64:
case ARM::BI__builtin_arm_cde_vcx3d_u64: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx2da_u64:
    Param0 = Intrinsic::arm_cde_vcx2a;
    break;
  case ARM::BI__builtin_arm_cde_vcx3d_u64:
    Param0 = Intrinsic::arm_cde_vcx3;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = DoubleTy;
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  llvm::Type *Val5 = DoubleTy;
  Value *Val6 = Builder.CreateBitCast(Val4, Val5);
  Value *Val7 = EmitScalarExpr(E->getArg(3));
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Param0, {DoubleTy}), {Val0, Val3, Val6, Val7});
  llvm::Type *Val9 = Int64Ty;
  return Builder.CreateBitCast(Val8, Val9);
}
case ARM::BI__builtin_arm_cde_vcx3da_u64: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = DoubleTy;
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  llvm::Type *Val5 = DoubleTy;
  Value *Val6 = Builder.CreateBitCast(Val4, Val5);
  Value *Val7 = EmitScalarExpr(E->getArg(3));
  llvm::Type *Val8 = DoubleTy;
  Value *Val9 = Builder.CreateBitCast(Val7, Val8);
  Value *Val10 = EmitScalarExpr(E->getArg(4));
  Value *Val11 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_vcx3a, {DoubleTy}), {Val0, Val3, Val6, Val9, Val10});
  llvm::Type *Val12 = Int64Ty;
  return Builder.CreateBitCast(Val11, Val12);
}
case ARM::BI__builtin_arm_cde_vcx1a_u32:
case ARM::BI__builtin_arm_cde_vcx2_u32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx1a_u32:
    Param0 = Intrinsic::arm_cde_vcx1a;
    break;
  case ARM::BI__builtin_arm_cde_vcx2_u32:
    Param0 = Intrinsic::arm_cde_vcx2;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = FloatTy;
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Param0, {FloatTy}), {Val0, Val3, Val4});
  llvm::Type *Val6 = Int32Ty;
  return Builder.CreateBitCast(Val5, Val6);
}
case ARM::BI__builtin_arm_cde_vcx2a_u32:
case ARM::BI__builtin_arm_cde_vcx3_u32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx2a_u32:
    Param0 = Intrinsic::arm_cde_vcx2a;
    break;
  case ARM::BI__builtin_arm_cde_vcx3_u32:
    Param0 = Intrinsic::arm_cde_vcx3;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = FloatTy;
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  llvm::Type *Val5 = FloatTy;
  Value *Val6 = Builder.CreateBitCast(Val4, Val5);
  Value *Val7 = EmitScalarExpr(E->getArg(3));
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Param0, {FloatTy}), {Val0, Val3, Val6, Val7});
  llvm::Type *Val9 = Int32Ty;
  return Builder.CreateBitCast(Val8, Val9);
}
case ARM::BI__builtin_arm_cde_vcx3a_u32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = FloatTy;
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  llvm::Type *Val5 = FloatTy;
  Value *Val6 = Builder.CreateBitCast(Val4, Val5);
  Value *Val7 = EmitScalarExpr(E->getArg(3));
  llvm::Type *Val8 = FloatTy;
  Value *Val9 = Builder.CreateBitCast(Val7, Val8);
  Value *Val10 = EmitScalarExpr(E->getArg(4));
  Value *Val11 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_vcx3a, {FloatTy}), {Val0, Val3, Val6, Val9, Val10});
  llvm::Type *Val12 = Int32Ty;
  return Builder.CreateBitCast(Val11, Val12);
}
case ARM::BI__builtin_arm_cde_vcx1qa_f32:
case ARM::BI__builtin_arm_cde_vcx2q_f32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx1qa_f32:
    Param0 = Intrinsic::arm_cde_vcx1qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_f32:
    Param0 = Intrinsic::arm_cde_vcx2q;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Param0), {Val0, Val3, Val4});
  llvm::Type *Val6 = llvm::FixedVectorType::get(FloatTy, 4);
  return Builder.CreateBitCast(Val5, Val6);
}
case ARM::BI__builtin_arm_cde_vcx1qa_f16:
case ARM::BI__builtin_arm_cde_vcx2q_f16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx1qa_f16:
    Param0 = Intrinsic::arm_cde_vcx1qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_f16:
    Param0 = Intrinsic::arm_cde_vcx2q;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Param0), {Val0, Val3, Val4});
  llvm::Type *Val6 = llvm::FixedVectorType::get(HalfTy, 8);
  return Builder.CreateBitCast(Val5, Val6);
}
case ARM::BI__builtin_arm_cde_vcx1qa_s16:
case ARM::BI__builtin_arm_cde_vcx1qa_u16:
case ARM::BI__builtin_arm_cde_vcx2q_s16:
case ARM::BI__builtin_arm_cde_vcx2q_u16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx1qa_s16:
    Param0 = Intrinsic::arm_cde_vcx1qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx1qa_u16:
    Param0 = Intrinsic::arm_cde_vcx1qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_s16:
    Param0 = Intrinsic::arm_cde_vcx2q;
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_u16:
    Param0 = Intrinsic::arm_cde_vcx2q;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Param0), {Val0, Val3, Val4});
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int16Ty, 8);
  return Builder.CreateBitCast(Val5, Val6);
}
case ARM::BI__builtin_arm_cde_vcx1qa_s32:
case ARM::BI__builtin_arm_cde_vcx1qa_u32:
case ARM::BI__builtin_arm_cde_vcx2q_s32:
case ARM::BI__builtin_arm_cde_vcx2q_u32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx1qa_s32:
    Param0 = Intrinsic::arm_cde_vcx1qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx1qa_u32:
    Param0 = Intrinsic::arm_cde_vcx1qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_s32:
    Param0 = Intrinsic::arm_cde_vcx2q;
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_u32:
    Param0 = Intrinsic::arm_cde_vcx2q;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Param0), {Val0, Val3, Val4});
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int32Ty, 4);
  return Builder.CreateBitCast(Val5, Val6);
}
case ARM::BI__builtin_arm_cde_vcx1qa_s64:
case ARM::BI__builtin_arm_cde_vcx1qa_u64:
case ARM::BI__builtin_arm_cde_vcx2q_s64:
case ARM::BI__builtin_arm_cde_vcx2q_u64: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx1qa_s64:
    Param0 = Intrinsic::arm_cde_vcx1qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx1qa_u64:
    Param0 = Intrinsic::arm_cde_vcx1qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_s64:
    Param0 = Intrinsic::arm_cde_vcx2q;
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_u64:
    Param0 = Intrinsic::arm_cde_vcx2q;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Param0), {Val0, Val3, Val4});
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int64Ty, 2);
  return Builder.CreateBitCast(Val5, Val6);
}
case ARM::BI__builtin_arm_cde_vcx1qa_s8:
case ARM::BI__builtin_arm_cde_vcx1qa_u8:
case ARM::BI__builtin_arm_cde_vcx2q_s8:
case ARM::BI__builtin_arm_cde_vcx2q_u8: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx1qa_s8:
    Param0 = Intrinsic::arm_cde_vcx1qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx1qa_u8:
    Param0 = Intrinsic::arm_cde_vcx1qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_s8:
    Param0 = Intrinsic::arm_cde_vcx2q;
    break;
  case ARM::BI__builtin_arm_cde_vcx2q_u8:
    Param0 = Intrinsic::arm_cde_vcx2q;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Param0), {Val0, Val3, Val4});
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int8Ty, 16);
  return Builder.CreateBitCast(Val5, Val6);
}
case ARM::BI__builtin_arm_cde_vcx2qa_impl_f32:
case ARM::BI__builtin_arm_cde_vcx3q_impl_f32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx2qa_impl_f32:
    Param0 = Intrinsic::arm_cde_vcx2qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_impl_f32:
    Param0 = Intrinsic::arm_cde_vcx3q;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Param0), {Val0, Val3, Val4, Val5});
  llvm::Type *Val7 = llvm::FixedVectorType::get(FloatTy, 4);
  return Builder.CreateBitCast(Val6, Val7);
}
case ARM::BI__builtin_arm_cde_vcx2qa_impl_f16:
case ARM::BI__builtin_arm_cde_vcx3q_impl_f16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx2qa_impl_f16:
    Param0 = Intrinsic::arm_cde_vcx2qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_impl_f16:
    Param0 = Intrinsic::arm_cde_vcx3q;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Param0), {Val0, Val3, Val4, Val5});
  llvm::Type *Val7 = llvm::FixedVectorType::get(HalfTy, 8);
  return Builder.CreateBitCast(Val6, Val7);
}
case ARM::BI__builtin_arm_cde_vcx2qa_impl_s16:
case ARM::BI__builtin_arm_cde_vcx2qa_impl_u16:
case ARM::BI__builtin_arm_cde_vcx3q_impl_s16:
case ARM::BI__builtin_arm_cde_vcx3q_impl_u16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx2qa_impl_s16:
    Param0 = Intrinsic::arm_cde_vcx2qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx2qa_impl_u16:
    Param0 = Intrinsic::arm_cde_vcx2qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_impl_s16:
    Param0 = Intrinsic::arm_cde_vcx3q;
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_impl_u16:
    Param0 = Intrinsic::arm_cde_vcx3q;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Param0), {Val0, Val3, Val4, Val5});
  llvm::Type *Val7 = llvm::FixedVectorType::get(Int16Ty, 8);
  return Builder.CreateBitCast(Val6, Val7);
}
case ARM::BI__builtin_arm_cde_vcx2qa_impl_s32:
case ARM::BI__builtin_arm_cde_vcx2qa_impl_u32:
case ARM::BI__builtin_arm_cde_vcx3q_impl_s32:
case ARM::BI__builtin_arm_cde_vcx3q_impl_u32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx2qa_impl_s32:
    Param0 = Intrinsic::arm_cde_vcx2qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx2qa_impl_u32:
    Param0 = Intrinsic::arm_cde_vcx2qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_impl_s32:
    Param0 = Intrinsic::arm_cde_vcx3q;
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_impl_u32:
    Param0 = Intrinsic::arm_cde_vcx3q;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Param0), {Val0, Val3, Val4, Val5});
  llvm::Type *Val7 = llvm::FixedVectorType::get(Int32Ty, 4);
  return Builder.CreateBitCast(Val6, Val7);
}
case ARM::BI__builtin_arm_cde_vcx2qa_impl_s64:
case ARM::BI__builtin_arm_cde_vcx2qa_impl_u64:
case ARM::BI__builtin_arm_cde_vcx3q_impl_s64:
case ARM::BI__builtin_arm_cde_vcx3q_impl_u64: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx2qa_impl_s64:
    Param0 = Intrinsic::arm_cde_vcx2qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx2qa_impl_u64:
    Param0 = Intrinsic::arm_cde_vcx2qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_impl_s64:
    Param0 = Intrinsic::arm_cde_vcx3q;
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_impl_u64:
    Param0 = Intrinsic::arm_cde_vcx3q;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Param0), {Val0, Val3, Val4, Val5});
  llvm::Type *Val7 = llvm::FixedVectorType::get(Int64Ty, 2);
  return Builder.CreateBitCast(Val6, Val7);
}
case ARM::BI__builtin_arm_cde_vcx2qa_impl_s8:
case ARM::BI__builtin_arm_cde_vcx2qa_impl_u8:
case ARM::BI__builtin_arm_cde_vcx3q_impl_s8:
case ARM::BI__builtin_arm_cde_vcx3q_impl_u8: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_vcx2qa_impl_s8:
    Param0 = Intrinsic::arm_cde_vcx2qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx2qa_impl_u8:
    Param0 = Intrinsic::arm_cde_vcx2qa;
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_impl_s8:
    Param0 = Intrinsic::arm_cde_vcx3q;
    break;
  case ARM::BI__builtin_arm_cde_vcx3q_impl_u8:
    Param0 = Intrinsic::arm_cde_vcx3q;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Param0), {Val0, Val3, Val4, Val5});
  llvm::Type *Val7 = llvm::FixedVectorType::get(Int8Ty, 16);
  return Builder.CreateBitCast(Val6, Val7);
}
case ARM::BI__builtin_arm_cde_vcx3qa_impl_f32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = EmitScalarExpr(E->getArg(4));
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_vcx3qa), {Val0, Val3, Val4, Val5, Val6});
  llvm::Type *Val8 = llvm::FixedVectorType::get(FloatTy, 4);
  return Builder.CreateBitCast(Val7, Val8);
}
case ARM::BI__builtin_arm_cde_vcx3qa_impl_f16: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = EmitScalarExpr(E->getArg(4));
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_vcx3qa), {Val0, Val3, Val4, Val5, Val6});
  llvm::Type *Val8 = llvm::FixedVectorType::get(HalfTy, 8);
  return Builder.CreateBitCast(Val7, Val8);
}
case ARM::BI__builtin_arm_cde_vcx3qa_impl_s16:
case ARM::BI__builtin_arm_cde_vcx3qa_impl_u16: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = EmitScalarExpr(E->getArg(4));
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_vcx3qa), {Val0, Val3, Val4, Val5, Val6});
  llvm::Type *Val8 = llvm::FixedVectorType::get(Int16Ty, 8);
  return Builder.CreateBitCast(Val7, Val8);
}
case ARM::BI__builtin_arm_cde_vcx3qa_impl_s32:
case ARM::BI__builtin_arm_cde_vcx3qa_impl_u32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = EmitScalarExpr(E->getArg(4));
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_vcx3qa), {Val0, Val3, Val4, Val5, Val6});
  llvm::Type *Val8 = llvm::FixedVectorType::get(Int32Ty, 4);
  return Builder.CreateBitCast(Val7, Val8);
}
case ARM::BI__builtin_arm_cde_vcx3qa_impl_s64:
case ARM::BI__builtin_arm_cde_vcx3qa_impl_u64: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = EmitScalarExpr(E->getArg(4));
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_vcx3qa), {Val0, Val3, Val4, Val5, Val6});
  llvm::Type *Val8 = llvm::FixedVectorType::get(Int64Ty, 2);
  return Builder.CreateBitCast(Val7, Val8);
}
case ARM::BI__builtin_arm_cde_vcx3qa_impl_s8:
case ARM::BI__builtin_arm_cde_vcx3qa_impl_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = EmitScalarExpr(E->getArg(4));
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_vcx3qa), {Val0, Val3, Val4, Val5, Val6});
  llvm::Type *Val8 = llvm::FixedVectorType::get(Int8Ty, 16);
  return Builder.CreateBitCast(Val7, Val8);
}
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_f16:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_f32:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_s16:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_s32:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_s64:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_s8:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_u16:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_u32:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_u64:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_vcx3q), {Val0, Val3, Val4, Val5});
}
case ARM::BI__builtin_arm_cde_vcx2q_u8_f16:
case ARM::BI__builtin_arm_cde_vcx2q_u8_f32:
case ARM::BI__builtin_arm_cde_vcx2q_u8_s16:
case ARM::BI__builtin_arm_cde_vcx2q_u8_s32:
case ARM::BI__builtin_arm_cde_vcx2q_u8_s64:
case ARM::BI__builtin_arm_cde_vcx2q_u8_s8:
case ARM::BI__builtin_arm_cde_vcx2q_u8_u16:
case ARM::BI__builtin_arm_cde_vcx2q_u8_u32:
case ARM::BI__builtin_arm_cde_vcx2q_u8_u64:
case ARM::BI__builtin_arm_cde_vcx2q_u8_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val3 = Builder.CreateBitCast(Val1, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_vcx2q), {Val0, Val3, Val4});
}
case ARM::BI__builtin_arm_cde_cx1:
case ARM::BI__builtin_arm_cde_vcx1q_u8: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_cde_cx1:
    Param0 = Intrinsic::arm_cde_cx1;
    break;
  case ARM::BI__builtin_arm_cde_vcx1q_u8:
    Param0 = Intrinsic::arm_cde_vcx1q;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateCall(CGM.getIntrinsic(Param0), {Val0, Val1});
}
case ARM::BI__builtin_arm_cde_vreinterpretq_u8_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  return (Val0);
}
case ARM::BI__builtin_arm_cde_cx1da: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val2 = Builder.CreateLShr(Val0, Val1);
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val5 = EmitScalarExpr(E->getArg(0));
  Value *Val6 = EmitScalarExpr(E->getArg(2));
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_cx1da), {Val5, Val4, Val3, Val6});
  Value *Val8 = Builder.CreateExtractValue(Val7, static_cast<unsigned>(1));
  Value *Val9 = Builder.CreateIntCast(Val8, Int64Ty, false);
  Value *Val10 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val11 = Builder.CreateShl(Val9, Val10);
  Value *Val12 = Builder.CreateExtractValue(Val7, static_cast<unsigned>(0));
  Value *Val13 = Builder.CreateIntCast(Val12, Int64Ty, false);
  return Builder.CreateOr(Val11, Val13);
}
case ARM::BI__builtin_arm_cde_cx2da: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val2 = Builder.CreateLShr(Val0, Val1);
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val5 = EmitScalarExpr(E->getArg(0));
  Value *Val6 = EmitScalarExpr(E->getArg(2));
  Value *Val7 = EmitScalarExpr(E->getArg(3));
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_cx2da), {Val5, Val4, Val3, Val6, Val7});
  Value *Val9 = Builder.CreateExtractValue(Val8, static_cast<unsigned>(1));
  Value *Val10 = Builder.CreateIntCast(Val9, Int64Ty, false);
  Value *Val11 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val12 = Builder.CreateShl(Val10, Val11);
  Value *Val13 = Builder.CreateExtractValue(Val8, static_cast<unsigned>(0));
  Value *Val14 = Builder.CreateIntCast(Val13, Int64Ty, false);
  return Builder.CreateOr(Val12, Val14);
}
case ARM::BI__builtin_arm_cde_cx3da: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val2 = Builder.CreateLShr(Val0, Val1);
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val5 = EmitScalarExpr(E->getArg(0));
  Value *Val6 = EmitScalarExpr(E->getArg(2));
  Value *Val7 = EmitScalarExpr(E->getArg(3));
  Value *Val8 = EmitScalarExpr(E->getArg(4));
  Value *Val9 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_cde_cx3da), {Val5, Val4, Val3, Val6, Val7, Val8});
  Value *Val10 = Builder.CreateExtractValue(Val9, static_cast<unsigned>(1));
  Value *Val11 = Builder.CreateIntCast(Val10, Int64Ty, false);
  Value *Val12 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val13 = Builder.CreateShl(Val11, Val12);
  Value *Val14 = Builder.CreateExtractValue(Val9, static_cast<unsigned>(0));
  Value *Val15 = Builder.CreateIntCast(Val14, Int64Ty, false);
  return Builder.CreateOr(Val13, Val15);
}
