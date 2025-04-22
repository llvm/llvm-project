case ARM::BI__builtin_arm_mve_vldrdq_gather_base_wb_s64:
case ARM::BI__builtin_arm_mve_vldrdq_gather_base_wb_u64:
case ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_f32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_s32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_u32: {
  llvm::Type * Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vldrdq_gather_base_wb_s64:
    Param0 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vldrdq_gather_base_wb_u64:
    Param0 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_f32:
    Param0 = llvm::FixedVectorType::get(FloatTy, 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_s32:
    Param0 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_u32:
    Param0 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreateLoad(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vldr_gather_base_wb, {Param0, Param1}), {Val1, Val2});
  Value *Val4 = Builder.CreateExtractValue(Val3, static_cast<unsigned>(1));
  Builder.CreateStore(Val4, Val0);
  return Builder.CreateExtractValue(Val3, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vddupq_wb_u16:
case ARM::BI__builtin_arm_mve_vddupq_wb_u32:
case ARM::BI__builtin_arm_mve_vddupq_wb_u8:
case ARM::BI__builtin_arm_mve_vidupq_wb_u16:
case ARM::BI__builtin_arm_mve_vidupq_wb_u32:
case ARM::BI__builtin_arm_mve_vidupq_wb_u8: {
  Intrinsic::ID  Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vddupq_wb_u16:
    Param0 = Intrinsic::arm_mve_vddup;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vddupq_wb_u32:
    Param0 = Intrinsic::arm_mve_vddup;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vddupq_wb_u8:
    Param0 = Intrinsic::arm_mve_vddup;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vidupq_wb_u16:
    Param0 = Intrinsic::arm_mve_vidup;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vidupq_wb_u32:
    Param0 = Intrinsic::arm_mve_vidup;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vidupq_wb_u8:
    Param0 = Intrinsic::arm_mve_vidup;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreateLoad(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Param0, {Param1}), {Val1, Val2});
  Value *Val4 = Builder.CreateExtractValue(Val3, static_cast<unsigned>(1));
  Builder.CreateStore(Val4, Val0);
  return Builder.CreateExtractValue(Val3, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vstrdq_scatter_base_wb_s64:
case ARM::BI__builtin_arm_mve_vstrdq_scatter_base_wb_u64:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_f32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_s32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_u32: {
  llvm::Type * Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vstrdq_scatter_base_wb_s64:
    Param0 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vstrdq_scatter_base_wb_u64:
    Param0 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_f32:
    Param0 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_s32:
    Param0 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_u32:
    Param0 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreateLoad(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vstr_scatter_base_wb, {Param0, Param1}), {Val1, Val2, Val3});
  return Builder.CreateStore(Val4, Val0);
}
case ARM::BI__builtin_arm_mve_vdwdupq_wb_u16:
case ARM::BI__builtin_arm_mve_vdwdupq_wb_u32:
case ARM::BI__builtin_arm_mve_vdwdupq_wb_u8:
case ARM::BI__builtin_arm_mve_viwdupq_wb_u16:
case ARM::BI__builtin_arm_mve_viwdupq_wb_u32:
case ARM::BI__builtin_arm_mve_viwdupq_wb_u8: {
  Intrinsic::ID  Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vdwdupq_wb_u16:
    Param0 = Intrinsic::arm_mve_vdwdup;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vdwdupq_wb_u32:
    Param0 = Intrinsic::arm_mve_vdwdup;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vdwdupq_wb_u8:
    Param0 = Intrinsic::arm_mve_vdwdup;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_wb_u16:
    Param0 = Intrinsic::arm_mve_viwdup;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_wb_u32:
    Param0 = Intrinsic::arm_mve_viwdup;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_wb_u8:
    Param0 = Intrinsic::arm_mve_viwdup;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreateLoad(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Param0, {Param1}), {Val1, Val2, Val3});
  Value *Val5 = Builder.CreateExtractValue(Val4, static_cast<unsigned>(1));
  Builder.CreateStore(Val5, Val0);
  return Builder.CreateExtractValue(Val4, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vldrdq_gather_base_wb_z_s64:
case ARM::BI__builtin_arm_mve_vldrdq_gather_base_wb_z_u64:
case ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_z_f32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_z_s32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_z_u32: {
  llvm::Type * Param0;
  llvm::Type * Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vldrdq_gather_base_wb_z_s64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vldrdq_gather_base_wb_z_u64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_z_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_z_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_z_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreateLoad(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val4});
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vldr_gather_base_wb_predicated, {Param1, Param2, Param0}), {Val1, Val2, Val5});
  Value *Val7 = Builder.CreateExtractValue(Val6, static_cast<unsigned>(1));
  Builder.CreateStore(Val7, Val0);
  return Builder.CreateExtractValue(Val6, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vstrdq_scatter_base_wb_p_s64:
case ARM::BI__builtin_arm_mve_vstrdq_scatter_base_wb_p_u64:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_p_f32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_p_s32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_p_u32: {
  llvm::Type * Param0;
  llvm::Type * Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vstrdq_scatter_base_wb_p_s64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vstrdq_scatter_base_wb_p_u64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_p_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_p_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_p_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreateLoad(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  Value *Val4 = EmitScalarExpr(E->getArg(3));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val5});
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vstr_scatter_base_wb_predicated, {Param1, Param2, Param0}), {Val1, Val2, Val3, Val6});
  return Builder.CreateStore(Val7, Val0);
}
case ARM::BI__builtin_arm_mve_vld1q_f32:
case ARM::BI__builtin_arm_mve_vldrwq_f32: {
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreatePointerCast((Val0.emitRawPointer(*this)), Builder.getPtrTy());
  Address Val2 = Address(Val1, llvm::FixedVectorType::get(FloatTy, 4), CharUnits::fromQuantity(4));
  return Builder.CreateLoad(Val2);
}
case ARM::BI__builtin_arm_mve_vld1q_f16:
case ARM::BI__builtin_arm_mve_vldrhq_f16: {
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreatePointerCast((Val0.emitRawPointer(*this)), Builder.getPtrTy());
  Address Val2 = Address(Val1, llvm::FixedVectorType::get(HalfTy, 8), CharUnits::fromQuantity(2));
  return Builder.CreateLoad(Val2);
}
case ARM::BI__builtin_arm_mve_vldrhq_s32:
case ARM::BI__builtin_arm_mve_vldrhq_u32: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vldrhq_s32:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_u32:
    Param0 = 1;
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreatePointerCast((Val0.emitRawPointer(*this)), Builder.getPtrTy());
  Address Val2 = Address(Val1, llvm::FixedVectorType::get(Int16Ty, 4), CharUnits::fromQuantity(2));
  Value *Val3 = Builder.CreateLoad(Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int32Ty, 4);
  return SignOrZeroExtend(Builder, Val3, Val4, static_cast<bool>(Param0));
}
case ARM::BI__builtin_arm_mve_vld1q_s16:
case ARM::BI__builtin_arm_mve_vld1q_u16:
case ARM::BI__builtin_arm_mve_vldrhq_s16:
case ARM::BI__builtin_arm_mve_vldrhq_u16: {
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreatePointerCast((Val0.emitRawPointer(*this)), Builder.getPtrTy());
  Address Val2 = Address(Val1, llvm::FixedVectorType::get(Int16Ty, 8), CharUnits::fromQuantity(2));
  return Builder.CreateLoad(Val2);
}
case ARM::BI__builtin_arm_mve_vld1q_s32:
case ARM::BI__builtin_arm_mve_vld1q_u32:
case ARM::BI__builtin_arm_mve_vldrwq_s32:
case ARM::BI__builtin_arm_mve_vldrwq_u32: {
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreatePointerCast((Val0.emitRawPointer(*this)), Builder.getPtrTy());
  Address Val2 = Address(Val1, llvm::FixedVectorType::get(Int32Ty, 4), CharUnits::fromQuantity(4));
  return Builder.CreateLoad(Val2);
}
case ARM::BI__builtin_arm_mve_vld1q_s8:
case ARM::BI__builtin_arm_mve_vld1q_u8:
case ARM::BI__builtin_arm_mve_vldrbq_s8:
case ARM::BI__builtin_arm_mve_vldrbq_u8: {
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreatePointerCast((Val0.emitRawPointer(*this)), Builder.getPtrTy());
  Address Val2 = Address(Val1, llvm::FixedVectorType::get(Int8Ty, 16), CharUnits::fromQuantity(1));
  return Builder.CreateLoad(Val2);
}
case ARM::BI__builtin_arm_mve_vldrbq_s32:
case ARM::BI__builtin_arm_mve_vldrbq_u32: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vldrbq_s32:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vldrbq_u32:
    Param0 = 1;
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreatePointerCast((Val0.emitRawPointer(*this)), Builder.getPtrTy());
  Address Val2 = Address(Val1, llvm::FixedVectorType::get(Int8Ty, 4), CharUnits::fromQuantity(1));
  Value *Val3 = Builder.CreateLoad(Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int32Ty, 4);
  return SignOrZeroExtend(Builder, Val3, Val4, static_cast<bool>(Param0));
}
case ARM::BI__builtin_arm_mve_vldrbq_s16:
case ARM::BI__builtin_arm_mve_vldrbq_u16: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vldrbq_s16:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vldrbq_u16:
    Param0 = 1;
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreatePointerCast((Val0.emitRawPointer(*this)), Builder.getPtrTy());
  Address Val2 = Address(Val1, llvm::FixedVectorType::get(Int8Ty, 8), CharUnits::fromQuantity(1));
  Value *Val3 = Builder.CreateLoad(Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int16Ty, 8);
  return SignOrZeroExtend(Builder, Val3, Val4, static_cast<bool>(Param0));
}
case ARM::BI__builtin_arm_mve_vld1q_z_s8:
case ARM::BI__builtin_arm_mve_vld1q_z_u8:
case ARM::BI__builtin_arm_mve_vldrbq_z_s8:
case ARM::BI__builtin_arm_mve_vldrbq_z_u8: {
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreatePointerCast((Val0.emitRawPointer(*this)), Builder.getPtrTy());
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, 1);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val4});
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val7 = llvm::Constant::getNullValue(Val6);
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::masked_load, {llvm::FixedVectorType::get(Int8Ty, 16), Builder.getPtrTy()}), {Val1, Val2, Val5, Val7});
}
case ARM::BI__builtin_arm_mve_vldrbq_z_s32:
case ARM::BI__builtin_arm_mve_vldrbq_z_u32: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vldrbq_z_s32:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vldrbq_z_u32:
    Param0 = 1;
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreatePointerCast((Val0.emitRawPointer(*this)), Builder.getPtrTy());
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, 1);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val4});
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int8Ty, 4);
  Value *Val7 = llvm::Constant::getNullValue(Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::masked_load, {llvm::FixedVectorType::get(Int8Ty, 4), Builder.getPtrTy()}), {Val1, Val2, Val5, Val7});
  llvm::Type *Val9 = llvm::FixedVectorType::get(Int32Ty, 4);
  return SignOrZeroExtend(Builder, Val8, Val9, static_cast<bool>(Param0));
}
case ARM::BI__builtin_arm_mve_vldrbq_z_s16:
case ARM::BI__builtin_arm_mve_vldrbq_z_u16: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vldrbq_z_s16:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vldrbq_z_u16:
    Param0 = 1;
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreatePointerCast((Val0.emitRawPointer(*this)), Builder.getPtrTy());
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, 1);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val4});
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int8Ty, 8);
  Value *Val7 = llvm::Constant::getNullValue(Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::masked_load, {llvm::FixedVectorType::get(Int8Ty, 8), Builder.getPtrTy()}), {Val1, Val2, Val5, Val7});
  llvm::Type *Val9 = llvm::FixedVectorType::get(Int16Ty, 8);
  return SignOrZeroExtend(Builder, Val8, Val9, static_cast<bool>(Param0));
}
case ARM::BI__builtin_arm_mve_vldrhq_z_s32:
case ARM::BI__builtin_arm_mve_vldrhq_z_u32: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vldrhq_z_s32:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_z_u32:
    Param0 = 1;
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreatePointerCast((Val0.emitRawPointer(*this)), Builder.getPtrTy());
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, 2);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val4});
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int16Ty, 4);
  Value *Val7 = llvm::Constant::getNullValue(Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::masked_load, {llvm::FixedVectorType::get(Int16Ty, 4), Builder.getPtrTy()}), {Val1, Val2, Val5, Val7});
  llvm::Type *Val9 = llvm::FixedVectorType::get(Int32Ty, 4);
  return SignOrZeroExtend(Builder, Val8, Val9, static_cast<bool>(Param0));
}
case ARM::BI__builtin_arm_mve_vld1q_z_f16:
case ARM::BI__builtin_arm_mve_vldrhq_z_f16: {
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreatePointerCast((Val0.emitRawPointer(*this)), Builder.getPtrTy());
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, 2);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val4});
  llvm::Type *Val6 = llvm::FixedVectorType::get(HalfTy, 8);
  Value *Val7 = llvm::Constant::getNullValue(Val6);
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::masked_load, {llvm::FixedVectorType::get(HalfTy, 8), Builder.getPtrTy()}), {Val1, Val2, Val5, Val7});
}
case ARM::BI__builtin_arm_mve_vld1q_z_s16:
case ARM::BI__builtin_arm_mve_vld1q_z_u16:
case ARM::BI__builtin_arm_mve_vldrhq_z_s16:
case ARM::BI__builtin_arm_mve_vldrhq_z_u16: {
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreatePointerCast((Val0.emitRawPointer(*this)), Builder.getPtrTy());
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, 2);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val4});
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val7 = llvm::Constant::getNullValue(Val6);
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::masked_load, {llvm::FixedVectorType::get(Int16Ty, 8), Builder.getPtrTy()}), {Val1, Val2, Val5, Val7});
}
case ARM::BI__builtin_arm_mve_vld1q_z_f32:
case ARM::BI__builtin_arm_mve_vldrwq_z_f32: {
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreatePointerCast((Val0.emitRawPointer(*this)), Builder.getPtrTy());
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, 4);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val4});
  llvm::Type *Val6 = llvm::FixedVectorType::get(FloatTy, 4);
  Value *Val7 = llvm::Constant::getNullValue(Val6);
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::masked_load, {llvm::FixedVectorType::get(FloatTy, 4), Builder.getPtrTy()}), {Val1, Val2, Val5, Val7});
}
case ARM::BI__builtin_arm_mve_vld1q_z_s32:
case ARM::BI__builtin_arm_mve_vld1q_z_u32:
case ARM::BI__builtin_arm_mve_vldrwq_z_s32:
case ARM::BI__builtin_arm_mve_vldrwq_z_u32: {
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = Builder.CreatePointerCast((Val0.emitRawPointer(*this)), Builder.getPtrTy());
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, 4);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val4});
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val7 = llvm::Constant::getNullValue(Val6);
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::masked_load, {llvm::FixedVectorType::get(Int32Ty, 4), Builder.getPtrTy()}), {Val1, Val2, Val5, Val7});
}
case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_s16:
case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_s32:
case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_s8:
case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_u16:
case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_u32:
case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_u8:
case ARM::BI__builtin_arm_mve_vstrdq_scatter_offset_p_s64:
case ARM::BI__builtin_arm_mve_vstrdq_scatter_offset_p_u64:
case ARM::BI__builtin_arm_mve_vstrdq_scatter_shifted_offset_p_s64:
case ARM::BI__builtin_arm_mve_vstrdq_scatter_shifted_offset_p_u64:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_p_f16:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_p_s16:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_p_s32:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_p_u16:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_p_u32:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_p_f16:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_p_s16:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_p_s32:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_p_u16:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_p_u32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_p_f32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_p_s32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_p_u32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_p_f32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_p_s32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_p_u32: {
  uint32_t  Param0;
  uint32_t  Param1;
  llvm::Type * Param2;
  llvm::Type * Param3;
  llvm::Type * Param4;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_s16:
    Param0 = 8;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_s32:
    Param0 = 8;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_s8:
    Param0 = 8;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_u16:
    Param0 = 8;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_u32:
    Param0 = 8;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_u8:
    Param0 = 8;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vstrdq_scatter_offset_p_s64:
    Param0 = 64;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param4 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vstrdq_scatter_offset_p_u64:
    Param0 = 64;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param4 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vstrdq_scatter_shifted_offset_p_s64:
    Param0 = 64;
    Param1 = 3;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param4 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vstrdq_scatter_shifted_offset_p_u64:
    Param0 = 64;
    Param1 = 3;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param4 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_p_f16:
    Param0 = 16;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_p_s16:
    Param0 = 16;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_p_s32:
    Param0 = 16;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_p_u16:
    Param0 = 16;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_p_u32:
    Param0 = 16;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_p_f16:
    Param0 = 16;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_p_s16:
    Param0 = 16;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_p_s32:
    Param0 = 16;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_p_u16:
    Param0 = 16;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_p_u32:
    Param0 = 16;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_p_f32:
    Param0 = 32;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_p_s32:
    Param0 = 32;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_p_u32:
    Param0 = 32;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_p_f32:
    Param0 = 32;
    Param1 = 2;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_p_s32:
    Param0 = 32;
    Param1 = 2;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_p_u32:
    Param0 = 32;
    Param1 = 2;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param2}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vstr_scatter_offset_predicated, {Builder.getPtrTy(), Param3, Param4, Param2}), {(Val0.emitRawPointer(*this)), Val1, Val2, Val3, Val4, Val7});
}
case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_s16:
case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_s32:
case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_s8:
case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_u16:
case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_u32:
case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_u8:
case ARM::BI__builtin_arm_mve_vstrdq_scatter_offset_s64:
case ARM::BI__builtin_arm_mve_vstrdq_scatter_offset_u64:
case ARM::BI__builtin_arm_mve_vstrdq_scatter_shifted_offset_s64:
case ARM::BI__builtin_arm_mve_vstrdq_scatter_shifted_offset_u64:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_f16:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_s16:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_s32:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_u16:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_u32:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_f16:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_s16:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_s32:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_u16:
case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_u32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_f32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_s32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_u32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_f32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_s32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_u32: {
  uint32_t  Param0;
  uint32_t  Param1;
  llvm::Type * Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_s16:
    Param0 = 8;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_s32:
    Param0 = 8;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_s8:
    Param0 = 8;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_u16:
    Param0 = 8;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_u32:
    Param0 = 8;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_u8:
    Param0 = 8;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vstrdq_scatter_offset_s64:
    Param0 = 64;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vstrdq_scatter_offset_u64:
    Param0 = 64;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vstrdq_scatter_shifted_offset_s64:
    Param0 = 64;
    Param1 = 3;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vstrdq_scatter_shifted_offset_u64:
    Param0 = 64;
    Param1 = 3;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_f16:
    Param0 = 16;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_s16:
    Param0 = 16;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_s32:
    Param0 = 16;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_u16:
    Param0 = 16;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_u32:
    Param0 = 16;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_f16:
    Param0 = 16;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_s16:
    Param0 = 16;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_s32:
    Param0 = 16;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_u16:
    Param0 = 16;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_u32:
    Param0 = 16;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_f32:
    Param0 = 32;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_s32:
    Param0 = 32;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_u32:
    Param0 = 32;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_f32:
    Param0 = 32;
    Param1 = 2;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_s32:
    Param0 = 32;
    Param1 = 2;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_u32:
    Param0 = 32;
    Param1 = 2;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param1);
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vstr_scatter_offset, {Builder.getPtrTy(), Param2, Param3}), {(Val0.emitRawPointer(*this)), Val1, Val2, Val3, Val4});
}
case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_s16:
case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_s32:
case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_s8:
case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_u16:
case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_u32:
case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_u8:
case ARM::BI__builtin_arm_mve_vldrdq_gather_offset_z_s64:
case ARM::BI__builtin_arm_mve_vldrdq_gather_offset_z_u64:
case ARM::BI__builtin_arm_mve_vldrdq_gather_shifted_offset_z_s64:
case ARM::BI__builtin_arm_mve_vldrdq_gather_shifted_offset_z_u64:
case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_z_f16:
case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_z_s16:
case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_z_s32:
case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_z_u16:
case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_z_u32:
case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_z_f16:
case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_z_s16:
case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_z_s32:
case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_z_u16:
case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_z_u32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_offset_z_f32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_offset_z_s32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_offset_z_u32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_z_f32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_z_s32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_z_u32: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  llvm::Type * Param3;
  llvm::Type * Param4;
  llvm::Type * Param5;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_s16:
    Param0 = 8;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_s32:
    Param0 = 8;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_s8:
    Param0 = 8;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_u16:
    Param0 = 8;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_u32:
    Param0 = 8;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_u8:
    Param0 = 8;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vldrdq_gather_offset_z_s64:
    Param0 = 64;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param4 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param5 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vldrdq_gather_offset_z_u64:
    Param0 = 64;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param4 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param5 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vldrdq_gather_shifted_offset_z_s64:
    Param0 = 64;
    Param1 = 3;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param4 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param5 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vldrdq_gather_shifted_offset_z_u64:
    Param0 = 64;
    Param1 = 3;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param4 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param5 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_z_f16:
    Param0 = 16;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(HalfTy, 8);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_z_s16:
    Param0 = 16;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_z_s32:
    Param0 = 16;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_z_u16:
    Param0 = 16;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_z_u32:
    Param0 = 16;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_z_f16:
    Param0 = 16;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(HalfTy, 8);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_z_s16:
    Param0 = 16;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_z_s32:
    Param0 = 16;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_z_u16:
    Param0 = 16;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_z_u32:
    Param0 = 16;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_offset_z_f32:
    Param0 = 32;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(FloatTy, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_offset_z_s32:
    Param0 = 32;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_offset_z_u32:
    Param0 = 32;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_z_f32:
    Param0 = 32;
    Param1 = 2;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(FloatTy, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_z_s32:
    Param0 = 32;
    Param1 = 2;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_z_u32:
    Param0 = 32;
    Param1 = 2;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param2);
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param3}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vldr_gather_offset_predicated, {Param4, Builder.getPtrTy(), Param5, Param3}), {(Val0.emitRawPointer(*this)), Val1, Val2, Val3, Val4, Val7});
}
case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_s16:
case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_s32:
case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_s8:
case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_u16:
case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_u32:
case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_u8:
case ARM::BI__builtin_arm_mve_vldrdq_gather_offset_s64:
case ARM::BI__builtin_arm_mve_vldrdq_gather_offset_u64:
case ARM::BI__builtin_arm_mve_vldrdq_gather_shifted_offset_s64:
case ARM::BI__builtin_arm_mve_vldrdq_gather_shifted_offset_u64:
case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_f16:
case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_s16:
case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_s32:
case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_u16:
case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_u32:
case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_f16:
case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_s16:
case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_s32:
case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_u16:
case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_u32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_offset_f32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_offset_s32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_offset_u32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_f32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_s32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_u32: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  llvm::Type * Param3;
  llvm::Type * Param4;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_s16:
    Param0 = 8;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_s32:
    Param0 = 8;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_s8:
    Param0 = 8;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_u16:
    Param0 = 8;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_u32:
    Param0 = 8;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrbq_gather_offset_u8:
    Param0 = 8;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vldrdq_gather_offset_s64:
    Param0 = 64;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param4 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vldrdq_gather_offset_u64:
    Param0 = 64;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param4 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vldrdq_gather_shifted_offset_s64:
    Param0 = 64;
    Param1 = 3;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param4 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vldrdq_gather_shifted_offset_u64:
    Param0 = 64;
    Param1 = 3;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param4 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_f16:
    Param0 = 16;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_s16:
    Param0 = 16;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_s32:
    Param0 = 16;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_u16:
    Param0 = 16;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_offset_u32:
    Param0 = 16;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_f16:
    Param0 = 16;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_s16:
    Param0 = 16;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_s32:
    Param0 = 16;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_u16:
    Param0 = 16;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_u32:
    Param0 = 16;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_offset_f32:
    Param0 = 32;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_offset_s32:
    Param0 = 32;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_offset_u32:
    Param0 = 32;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_f32:
    Param0 = 32;
    Param1 = 2;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_s32:
    Param0 = 32;
    Param1 = 2;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_u32:
    Param0 = 32;
    Param1 = 2;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param2);
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vldr_gather_offset, {Param3, Builder.getPtrTy(), Param4}), {(Val0.emitRawPointer(*this)), Val1, Val2, Val3, Val4});
}
case ARM::BI__builtin_arm_mve_vshlcq_s16:
case ARM::BI__builtin_arm_mve_vshlcq_s32:
case ARM::BI__builtin_arm_mve_vshlcq_s8:
case ARM::BI__builtin_arm_mve_vshlcq_u16:
case ARM::BI__builtin_arm_mve_vshlcq_u32:
case ARM::BI__builtin_arm_mve_vshlcq_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vshlcq_s16:
    Param0 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshlcq_s32:
    Param0 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshlcq_s8:
    Param0 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshlcq_u16:
    Param0 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshlcq_u32:
    Param0 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshlcq_u8:
    Param0 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(1));
  Value *Val1 = Builder.CreateLoad(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vshlc, {Param0}), {Val2, Val1, Val3});
  Value *Val5 = Builder.CreateExtractValue(Val4, static_cast<unsigned>(0));
  Builder.CreateStore(Val5, Val0);
  return Builder.CreateExtractValue(Val4, static_cast<unsigned>(1));
}
case ARM::BI__builtin_arm_mve_vddupq_m_wb_u16:
case ARM::BI__builtin_arm_mve_vddupq_m_wb_u32:
case ARM::BI__builtin_arm_mve_vddupq_m_wb_u8:
case ARM::BI__builtin_arm_mve_vidupq_m_wb_u16:
case ARM::BI__builtin_arm_mve_vidupq_m_wb_u32:
case ARM::BI__builtin_arm_mve_vidupq_m_wb_u8:
case ARM::BI__builtin_arm_mve_vshlcq_m_s16:
case ARM::BI__builtin_arm_mve_vshlcq_m_s32:
case ARM::BI__builtin_arm_mve_vshlcq_m_s8:
case ARM::BI__builtin_arm_mve_vshlcq_m_u16:
case ARM::BI__builtin_arm_mve_vshlcq_m_u32:
case ARM::BI__builtin_arm_mve_vshlcq_m_u8: {
  llvm::Type * Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  unsigned  Param3;
  unsigned  Param4;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vddupq_m_wb_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vddup_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = 1;
    Param4 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vddupq_m_wb_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vddup_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = 1;
    Param4 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vddupq_m_wb_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vddup_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param3 = 1;
    Param4 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vidupq_m_wb_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vidup_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = 1;
    Param4 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vidupq_m_wb_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vidup_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = 1;
    Param4 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vidupq_m_wb_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vidup_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param3 = 1;
    Param4 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vshlcq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vshlc_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = 0;
    Param4 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vshlcq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vshlc_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = 0;
    Param4 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vshlcq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vshlc_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param3 = 0;
    Param4 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vshlcq_m_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vshlc_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = 0;
    Param4 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vshlcq_m_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vshlc_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = 0;
    Param4 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vshlcq_m_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vshlc_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param3 = 0;
    Param4 = 1;
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(1));
  Value *Val1 = Builder.CreateLoad(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  Value *Val4 = EmitScalarExpr(E->getArg(3));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val5});
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2, Param0}), {Val2, Val1, Val3, Val6});
  Value *Val8 = Builder.CreateExtractValue(Val7, static_cast<unsigned>(Param3));
  Builder.CreateStore(Val8, Val0);
  return Builder.CreateExtractValue(Val7, static_cast<unsigned>(Param4));
}
case ARM::BI__builtin_arm_mve_vdwdupq_m_wb_u16:
case ARM::BI__builtin_arm_mve_vdwdupq_m_wb_u32:
case ARM::BI__builtin_arm_mve_vdwdupq_m_wb_u8:
case ARM::BI__builtin_arm_mve_viwdupq_m_wb_u16:
case ARM::BI__builtin_arm_mve_viwdupq_m_wb_u32:
case ARM::BI__builtin_arm_mve_viwdupq_m_wb_u8: {
  llvm::Type * Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vdwdupq_m_wb_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vdwdup_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vdwdupq_m_wb_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vdwdup_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vdwdupq_m_wb_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vdwdup_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_m_wb_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_viwdup_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_m_wb_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_viwdup_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_m_wb_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_viwdup_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Address Val0 = EmitPointerWithAlignment(E->getArg(1));
  Value *Val1 = Builder.CreateLoad(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  Value *Val4 = EmitScalarExpr(E->getArg(3));
  Value *Val5 = EmitScalarExpr(E->getArg(4));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val6});
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2, Param0}), {Val2, Val1, Val3, Val4, Val7});
  Value *Val9 = Builder.CreateExtractValue(Val8, static_cast<unsigned>(1));
  Builder.CreateStore(Val9, Val0);
  return Builder.CreateExtractValue(Val8, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vld2q_f16:
case ARM::BI__builtin_arm_mve_vld2q_f32:
case ARM::BI__builtin_arm_mve_vld2q_s16:
case ARM::BI__builtin_arm_mve_vld2q_s32:
case ARM::BI__builtin_arm_mve_vld2q_s8:
case ARM::BI__builtin_arm_mve_vld2q_u16:
case ARM::BI__builtin_arm_mve_vld2q_u32:
case ARM::BI__builtin_arm_mve_vld2q_u8: {
  CustomCodeGenType = CustomCodeGen::VLD24;
  IRIntr = Intrinsic::arm_mve_vld2q;
  NumVectors = 2;
  break; // custom code gen
}
case ARM::BI__builtin_arm_mve_vld4q_f16:
case ARM::BI__builtin_arm_mve_vld4q_f32:
case ARM::BI__builtin_arm_mve_vld4q_s16:
case ARM::BI__builtin_arm_mve_vld4q_s32:
case ARM::BI__builtin_arm_mve_vld4q_s8:
case ARM::BI__builtin_arm_mve_vld4q_u16:
case ARM::BI__builtin_arm_mve_vld4q_u32:
case ARM::BI__builtin_arm_mve_vld4q_u8: {
  CustomCodeGenType = CustomCodeGen::VLD24;
  IRIntr = Intrinsic::arm_mve_vld4q;
  NumVectors = 4;
  break; // custom code gen
}
case ARM::BI__builtin_arm_mve_vst2q_f16:
case ARM::BI__builtin_arm_mve_vst2q_f32:
case ARM::BI__builtin_arm_mve_vst2q_s16:
case ARM::BI__builtin_arm_mve_vst2q_s32:
case ARM::BI__builtin_arm_mve_vst2q_s8:
case ARM::BI__builtin_arm_mve_vst2q_u16:
case ARM::BI__builtin_arm_mve_vst2q_u32:
case ARM::BI__builtin_arm_mve_vst2q_u8: {
  CustomCodeGenType = CustomCodeGen::VST24;
  IRIntr = Intrinsic::arm_mve_vst2q;
  NumVectors = 2;
  break; // custom code gen
}
case ARM::BI__builtin_arm_mve_vst4q_f16:
case ARM::BI__builtin_arm_mve_vst4q_f32:
case ARM::BI__builtin_arm_mve_vst4q_s16:
case ARM::BI__builtin_arm_mve_vst4q_s32:
case ARM::BI__builtin_arm_mve_vst4q_s8:
case ARM::BI__builtin_arm_mve_vst4q_u16:
case ARM::BI__builtin_arm_mve_vst4q_u32:
case ARM::BI__builtin_arm_mve_vst4q_u8: {
  CustomCodeGenType = CustomCodeGen::VST24;
  IRIntr = Intrinsic::arm_mve_vst4q;
  NumVectors = 4;
  break; // custom code gen
}
case ARM::BI__builtin_arm_mve_vmaxnmaq_f16:
case ARM::BI__builtin_arm_mve_vmaxnmaq_f32:
case ARM::BI__builtin_arm_mve_vminnmaq_f16:
case ARM::BI__builtin_arm_mve_vminnmaq_f32: {
  llvm::Type * Param0;
  Intrinsic::ID  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmaxnmaq_f16:
    Param0 = llvm::FixedVectorType::get(HalfTy, 8);
    Param1 = Intrinsic::maxnum;
    break;
  case ARM::BI__builtin_arm_mve_vmaxnmaq_f32:
    Param0 = llvm::FixedVectorType::get(FloatTy, 4);
    Param1 = Intrinsic::maxnum;
    break;
  case ARM::BI__builtin_arm_mve_vminnmaq_f16:
    Param0 = llvm::FixedVectorType::get(HalfTy, 8);
    Param1 = Intrinsic::minnum;
    break;
  case ARM::BI__builtin_arm_mve_vminnmaq_f32:
    Param0 = llvm::FixedVectorType::get(FloatTy, 4);
    Param1 = Intrinsic::minnum;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::fabs, {Param0}), {Val0});
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::fabs, {Param0}), {Val2});
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {Param0}), {Val1, Val3});
}
case ARM::BI__builtin_arm_mve_vctp16q:
case ARM::BI__builtin_arm_mve_vctp32q:
case ARM::BI__builtin_arm_mve_vctp64q:
case ARM::BI__builtin_arm_mve_vctp8q: {
  Intrinsic::ID  Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vctp16q:
    Param0 = Intrinsic::arm_mve_vctp16;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vctp32q:
    Param0 = Intrinsic::arm_mve_vctp32;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vctp64q:
    Param0 = Intrinsic::arm_mve_vctp64;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    break;
  case ARM::BI__builtin_arm_mve_vctp8q:
    Param0 = Intrinsic::arm_mve_vctp8;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = Builder.CreateCall(CGM.getIntrinsic(Param0), {Val0});
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param1}), {Val1});
  return Builder.CreateIntCast(Val2, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vmvnq_n_s32:
case ARM::BI__builtin_arm_mve_vmvnq_n_u32: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmvnq_n_s32:
    Param0 = true;
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_n_u32:
    Param0 = false;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, Param0);
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  return Builder.CreateNot(Val2);
}
case ARM::BI__builtin_arm_mve_vmvnq_n_s16:
case ARM::BI__builtin_arm_mve_vmvnq_n_u16: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmvnq_n_s16:
    Param0 = true;
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_n_u16:
    Param0 = false;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateIntCast(Val1, Int16Ty, Param0);
  Value *Val3 = ARMMVEVectorSplat(Builder, Val2);
  return Builder.CreateNot(Val3);
}
case ARM::BI__builtin_arm_mve_vmaxavq_s16:
case ARM::BI__builtin_arm_mve_vmaxavq_s8:
case ARM::BI__builtin_arm_mve_vminavq_s16:
case ARM::BI__builtin_arm_mve_vminavq_s8: {
  Intrinsic::ID  Param0;
  llvm::Type * Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmaxavq_s16:
    Param0 = Intrinsic::arm_mve_maxav;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param2 = Int16Ty;
    break;
  case ARM::BI__builtin_arm_mve_vmaxavq_s8:
    Param0 = Intrinsic::arm_mve_maxav;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param2 = Int8Ty;
    break;
  case ARM::BI__builtin_arm_mve_vminavq_s16:
    Param0 = Intrinsic::arm_mve_minav;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param2 = Int16Ty;
    break;
  case ARM::BI__builtin_arm_mve_vminavq_s8:
    Param0 = Intrinsic::arm_mve_minav;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param2 = Int8Ty;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Param0, {Param1}), {Val1, Val2});
  return Builder.CreateIntCast(Val3, Param2, false);
}
case ARM::BI__builtin_arm_mve_vmaxavq_p_s16:
case ARM::BI__builtin_arm_mve_vmaxavq_p_s8:
case ARM::BI__builtin_arm_mve_vminavq_p_s16:
case ARM::BI__builtin_arm_mve_vminavq_p_s8: {
  llvm::Type * Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmaxavq_p_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_maxav_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = Int16Ty;
    break;
  case ARM::BI__builtin_arm_mve_vmaxavq_p_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_maxav_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param3 = Int8Ty;
    break;
  case ARM::BI__builtin_arm_mve_vminavq_p_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_minav_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = Int16Ty;
    break;
  case ARM::BI__builtin_arm_mve_vminavq_p_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_minav_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param3 = Int8Ty;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val4});
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2, Param0}), {Val1, Val2, Val5});
  return Builder.CreateIntCast(Val6, Param3, false);
}
case ARM::BI__builtin_arm_mve_vmaxvq_s16:
case ARM::BI__builtin_arm_mve_vmaxvq_s8:
case ARM::BI__builtin_arm_mve_vmaxvq_u16:
case ARM::BI__builtin_arm_mve_vmaxvq_u8:
case ARM::BI__builtin_arm_mve_vminvq_s16:
case ARM::BI__builtin_arm_mve_vminvq_s8:
case ARM::BI__builtin_arm_mve_vminvq_u16:
case ARM::BI__builtin_arm_mve_vminvq_u8: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  llvm::Type * Param3;
  bool  Param4;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmaxvq_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_maxv;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = Int16Ty;
    Param4 = true;
    break;
  case ARM::BI__builtin_arm_mve_vmaxvq_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_maxv;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param3 = Int8Ty;
    Param4 = true;
    break;
  case ARM::BI__builtin_arm_mve_vmaxvq_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_maxv;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = Int16Ty;
    Param4 = false;
    break;
  case ARM::BI__builtin_arm_mve_vmaxvq_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_maxv;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param3 = Int8Ty;
    Param4 = false;
    break;
  case ARM::BI__builtin_arm_mve_vminvq_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_minv;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = Int16Ty;
    Param4 = true;
    break;
  case ARM::BI__builtin_arm_mve_vminvq_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_minv;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param3 = Int8Ty;
    Param4 = true;
    break;
  case ARM::BI__builtin_arm_mve_vminvq_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_minv;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = Int16Ty;
    Param4 = false;
    break;
  case ARM::BI__builtin_arm_mve_vminvq_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_minv;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param3 = Int8Ty;
    Param4 = false;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2}), {Val1, Val2, Val3});
  return Builder.CreateIntCast(Val4, Param3, Param4);
}
case ARM::BI__builtin_arm_mve_vmaxvq_p_s16:
case ARM::BI__builtin_arm_mve_vmaxvq_p_s8:
case ARM::BI__builtin_arm_mve_vmaxvq_p_u16:
case ARM::BI__builtin_arm_mve_vmaxvq_p_u8:
case ARM::BI__builtin_arm_mve_vminvq_p_s16:
case ARM::BI__builtin_arm_mve_vminvq_p_s8:
case ARM::BI__builtin_arm_mve_vminvq_p_u16:
case ARM::BI__builtin_arm_mve_vminvq_p_u8: {
  uint32_t  Param0;
  llvm::Type * Param1;
  Intrinsic::ID  Param2;
  llvm::Type * Param3;
  llvm::Type * Param4;
  bool  Param5;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmaxvq_p_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_maxv_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = Int16Ty;
    Param5 = true;
    break;
  case ARM::BI__builtin_arm_mve_vmaxvq_p_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_maxv_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param4 = Int8Ty;
    Param5 = true;
    break;
  case ARM::BI__builtin_arm_mve_vmaxvq_p_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_maxv_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = Int16Ty;
    Param5 = false;
    break;
  case ARM::BI__builtin_arm_mve_vmaxvq_p_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_maxv_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param4 = Int8Ty;
    Param5 = false;
    break;
  case ARM::BI__builtin_arm_mve_vminvq_p_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_minv_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = Int16Ty;
    Param5 = true;
    break;
  case ARM::BI__builtin_arm_mve_vminvq_p_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_minv_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param4 = Int8Ty;
    Param5 = true;
    break;
  case ARM::BI__builtin_arm_mve_vminvq_p_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_minv_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = Int16Ty;
    Param5 = false;
    break;
  case ARM::BI__builtin_arm_mve_vminvq_p_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_minv_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param4 = Int8Ty;
    Param5 = false;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param1}), {Val5});
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Param2, {Param3, Param1}), {Val1, Val2, Val3, Val6});
  return Builder.CreateIntCast(Val7, Param4, Param5);
}
case ARM::BI__builtin_arm_mve_vadcq_s32:
case ARM::BI__builtin_arm_mve_vadcq_u32:
case ARM::BI__builtin_arm_mve_vsbcq_s32:
case ARM::BI__builtin_arm_mve_vsbcq_u32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vadcq_s32:
    Param0 = Intrinsic::arm_mve_vadc;
    break;
  case ARM::BI__builtin_arm_mve_vadcq_u32:
    Param0 = Intrinsic::arm_mve_vadc;
    break;
  case ARM::BI__builtin_arm_mve_vsbcq_s32:
    Param0 = Intrinsic::arm_mve_vsbc;
    break;
  case ARM::BI__builtin_arm_mve_vsbcq_u32:
    Param0 = Intrinsic::arm_mve_vsbc;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Address Val2 = EmitPointerWithAlignment(E->getArg(2));
  Value *Val3 = Builder.CreateLoad(Val2);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, 29);
  Value *Val5 = Builder.CreateShl(Val3, Val4);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int32Ty, 4)}), {Val0, Val1, Val5});
  Value *Val7 = llvm::ConstantInt::get(Int32Ty, 1);
  Value *Val8 = Builder.CreateExtractValue(Val6, static_cast<unsigned>(1));
  Value *Val9 = llvm::ConstantInt::get(Int32Ty, 29);
  Value *Val10 = Builder.CreateLShr(Val8, Val9);
  Value *Val11 = Builder.CreateAnd(Val7, Val10);
  Builder.CreateStore(Val11, Val2);
  return Builder.CreateExtractValue(Val6, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vcmpeqq_n_f16:
case ARM::BI__builtin_arm_mve_vcmpeqq_n_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpeqq_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = Builder.CreateFCmpOEQ(Val0, Val2);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val3});
  return Builder.CreateIntCast(Val4, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpgeq_n_f16:
case ARM::BI__builtin_arm_mve_vcmpgeq_n_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpgeq_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgeq_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = Builder.CreateFCmpOGE(Val0, Val2);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val3});
  return Builder.CreateIntCast(Val4, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpgtq_n_f16:
case ARM::BI__builtin_arm_mve_vcmpgtq_n_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpgtq_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgtq_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = Builder.CreateFCmpOGT(Val0, Val2);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val3});
  return Builder.CreateIntCast(Val4, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpleq_n_f16:
case ARM::BI__builtin_arm_mve_vcmpleq_n_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpleq_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpleq_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = Builder.CreateFCmpULE(Val0, Val2);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val3});
  return Builder.CreateIntCast(Val4, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpltq_n_f16:
case ARM::BI__builtin_arm_mve_vcmpltq_n_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpltq_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpltq_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = Builder.CreateFCmpULT(Val0, Val2);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val3});
  return Builder.CreateIntCast(Val4, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpneq_n_f16:
case ARM::BI__builtin_arm_mve_vcmpneq_n_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpneq_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = Builder.CreateFCmpUNE(Val0, Val2);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val3});
  return Builder.CreateIntCast(Val4, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpeqq_n_s16:
case ARM::BI__builtin_arm_mve_vcmpeqq_n_s32:
case ARM::BI__builtin_arm_mve_vcmpeqq_n_s8:
case ARM::BI__builtin_arm_mve_vcmpeqq_n_u16:
case ARM::BI__builtin_arm_mve_vcmpeqq_n_u32:
case ARM::BI__builtin_arm_mve_vcmpeqq_n_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpeqq_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = Builder.CreateICmpEQ(Val0, Val2);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val3});
  return Builder.CreateIntCast(Val4, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpneq_n_s16:
case ARM::BI__builtin_arm_mve_vcmpneq_n_s32:
case ARM::BI__builtin_arm_mve_vcmpneq_n_s8:
case ARM::BI__builtin_arm_mve_vcmpneq_n_u16:
case ARM::BI__builtin_arm_mve_vcmpneq_n_u32:
case ARM::BI__builtin_arm_mve_vcmpneq_n_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpneq_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = Builder.CreateICmpNE(Val0, Val2);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val3});
  return Builder.CreateIntCast(Val4, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpgeq_n_s16:
case ARM::BI__builtin_arm_mve_vcmpgeq_n_s32:
case ARM::BI__builtin_arm_mve_vcmpgeq_n_s8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpgeq_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgeq_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgeq_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = Builder.CreateICmpSGE(Val0, Val2);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val3});
  return Builder.CreateIntCast(Val4, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpgtq_n_s16:
case ARM::BI__builtin_arm_mve_vcmpgtq_n_s32:
case ARM::BI__builtin_arm_mve_vcmpgtq_n_s8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpgtq_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgtq_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgtq_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = Builder.CreateICmpSGT(Val0, Val2);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val3});
  return Builder.CreateIntCast(Val4, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpleq_n_s16:
case ARM::BI__builtin_arm_mve_vcmpleq_n_s32:
case ARM::BI__builtin_arm_mve_vcmpleq_n_s8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpleq_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpleq_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpleq_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = Builder.CreateICmpSLE(Val0, Val2);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val3});
  return Builder.CreateIntCast(Val4, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpltq_n_s16:
case ARM::BI__builtin_arm_mve_vcmpltq_n_s32:
case ARM::BI__builtin_arm_mve_vcmpltq_n_s8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpltq_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpltq_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpltq_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = Builder.CreateICmpSLT(Val0, Val2);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val3});
  return Builder.CreateIntCast(Val4, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpcsq_n_u16:
case ARM::BI__builtin_arm_mve_vcmpcsq_n_u32:
case ARM::BI__builtin_arm_mve_vcmpcsq_n_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpcsq_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpcsq_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpcsq_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = Builder.CreateICmpUGE(Val0, Val2);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val3});
  return Builder.CreateIntCast(Val4, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmphiq_n_u16:
case ARM::BI__builtin_arm_mve_vcmphiq_n_u32:
case ARM::BI__builtin_arm_mve_vcmphiq_n_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmphiq_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmphiq_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmphiq_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = Builder.CreateICmpUGT(Val0, Val2);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val3});
  return Builder.CreateIntCast(Val4, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vqdmullbq_n_s16:
case ARM::BI__builtin_arm_mve_vqdmullbq_n_s32:
case ARM::BI__builtin_arm_mve_vqdmulltq_n_s16:
case ARM::BI__builtin_arm_mve_vqdmulltq_n_s32: {
  uint32_t  Param0;
  llvm::Type * Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vqdmullbq_n_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmullbq_n_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulltq_n_s16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulltq_n_s32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vqdmull, {Param1, Param2}), {Val0, Val2, Val3});
}
case ARM::BI__builtin_arm_mve_vhaddq_n_s16:
case ARM::BI__builtin_arm_mve_vhaddq_n_s32:
case ARM::BI__builtin_arm_mve_vhaddq_n_s8:
case ARM::BI__builtin_arm_mve_vhaddq_n_u16:
case ARM::BI__builtin_arm_mve_vhaddq_n_u32:
case ARM::BI__builtin_arm_mve_vhaddq_n_u8:
case ARM::BI__builtin_arm_mve_vhsubq_n_s16:
case ARM::BI__builtin_arm_mve_vhsubq_n_s32:
case ARM::BI__builtin_arm_mve_vhsubq_n_s8:
case ARM::BI__builtin_arm_mve_vhsubq_n_u16:
case ARM::BI__builtin_arm_mve_vhsubq_n_u32:
case ARM::BI__builtin_arm_mve_vhsubq_n_u8: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vhaddq_n_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vhadd;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_n_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vhadd;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_n_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vhadd;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_n_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vhadd;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_n_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vhadd;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_n_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vhadd;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_n_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vhsub;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_n_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vhsub;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_n_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vhsub;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_n_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vhsub;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_n_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vhsub;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_n_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vhsub;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2}), {Val0, Val2, Val3});
}
case ARM::BI__builtin_arm_mve_vaddq_n_s16:
case ARM::BI__builtin_arm_mve_vaddq_n_s32:
case ARM::BI__builtin_arm_mve_vaddq_n_s8:
case ARM::BI__builtin_arm_mve_vaddq_n_u16:
case ARM::BI__builtin_arm_mve_vaddq_n_u32:
case ARM::BI__builtin_arm_mve_vaddq_n_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  return Builder.CreateAdd(Val0, Val2);
}
case ARM::BI__builtin_arm_mve_vqaddq_n_s16:
case ARM::BI__builtin_arm_mve_vqaddq_n_s32:
case ARM::BI__builtin_arm_mve_vqaddq_n_s8:
case ARM::BI__builtin_arm_mve_vqaddq_n_u16:
case ARM::BI__builtin_arm_mve_vqaddq_n_u32:
case ARM::BI__builtin_arm_mve_vqaddq_n_u8:
case ARM::BI__builtin_arm_mve_vqdmulhq_n_s16:
case ARM::BI__builtin_arm_mve_vqdmulhq_n_s32:
case ARM::BI__builtin_arm_mve_vqdmulhq_n_s8:
case ARM::BI__builtin_arm_mve_vqrdmulhq_n_s16:
case ARM::BI__builtin_arm_mve_vqrdmulhq_n_s32:
case ARM::BI__builtin_arm_mve_vqrdmulhq_n_s8:
case ARM::BI__builtin_arm_mve_vqsubq_n_s16:
case ARM::BI__builtin_arm_mve_vqsubq_n_s32:
case ARM::BI__builtin_arm_mve_vqsubq_n_s8:
case ARM::BI__builtin_arm_mve_vqsubq_n_u16:
case ARM::BI__builtin_arm_mve_vqsubq_n_u32:
case ARM::BI__builtin_arm_mve_vqsubq_n_u8: {
  Intrinsic::ID  Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vqaddq_n_s16:
    Param0 = Intrinsic::sadd_sat;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_n_s32:
    Param0 = Intrinsic::sadd_sat;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_n_s8:
    Param0 = Intrinsic::sadd_sat;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_n_u16:
    Param0 = Intrinsic::uadd_sat;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_n_u32:
    Param0 = Intrinsic::uadd_sat;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_n_u8:
    Param0 = Intrinsic::uadd_sat;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulhq_n_s16:
    Param0 = Intrinsic::arm_mve_vqdmulh;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulhq_n_s32:
    Param0 = Intrinsic::arm_mve_vqdmulh;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulhq_n_s8:
    Param0 = Intrinsic::arm_mve_vqdmulh;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmulhq_n_s16:
    Param0 = Intrinsic::arm_mve_vqrdmulh;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmulhq_n_s32:
    Param0 = Intrinsic::arm_mve_vqrdmulh;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmulhq_n_s8:
    Param0 = Intrinsic::arm_mve_vqrdmulh;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_n_s16:
    Param0 = Intrinsic::ssub_sat;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_n_s32:
    Param0 = Intrinsic::ssub_sat;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_n_s8:
    Param0 = Intrinsic::ssub_sat;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_n_u16:
    Param0 = Intrinsic::usub_sat;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_n_u32:
    Param0 = Intrinsic::usub_sat;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_n_u8:
    Param0 = Intrinsic::usub_sat;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {Param1}), {Val0, Val2});
}
case ARM::BI__builtin_arm_mve_vaddq_n_f16:
case ARM::BI__builtin_arm_mve_vaddq_n_f32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  return Builder.CreateFAdd(Val0, Val2);
}
case ARM::BI__builtin_arm_mve_vmulq_n_f16:
case ARM::BI__builtin_arm_mve_vmulq_n_f32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  return Builder.CreateFMul(Val0, Val2);
}
case ARM::BI__builtin_arm_mve_vsubq_n_f16:
case ARM::BI__builtin_arm_mve_vsubq_n_f32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  return Builder.CreateFSub(Val0, Val2);
}
case ARM::BI__builtin_arm_mve_vmulq_n_s16:
case ARM::BI__builtin_arm_mve_vmulq_n_s32:
case ARM::BI__builtin_arm_mve_vmulq_n_s8:
case ARM::BI__builtin_arm_mve_vmulq_n_u16:
case ARM::BI__builtin_arm_mve_vmulq_n_u32:
case ARM::BI__builtin_arm_mve_vmulq_n_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  return Builder.CreateMul(Val0, Val2);
}
case ARM::BI__builtin_arm_mve_vsubq_n_s16:
case ARM::BI__builtin_arm_mve_vsubq_n_s32:
case ARM::BI__builtin_arm_mve_vsubq_n_s8:
case ARM::BI__builtin_arm_mve_vsubq_n_u16:
case ARM::BI__builtin_arm_mve_vsubq_n_u32:
case ARM::BI__builtin_arm_mve_vsubq_n_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  return Builder.CreateSub(Val0, Val2);
}
case ARM::BI__builtin_arm_mve_vmaxnmavq_f16:
case ARM::BI__builtin_arm_mve_vmaxnmavq_f32:
case ARM::BI__builtin_arm_mve_vmaxnmvq_f16:
case ARM::BI__builtin_arm_mve_vmaxnmvq_f32:
case ARM::BI__builtin_arm_mve_vminnmavq_f16:
case ARM::BI__builtin_arm_mve_vminnmavq_f32:
case ARM::BI__builtin_arm_mve_vminnmvq_f16:
case ARM::BI__builtin_arm_mve_vminnmvq_f32: {
  Intrinsic::ID  Param0;
  llvm::Type * Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmaxnmavq_f16:
    Param0 = Intrinsic::arm_mve_maxnmav;
    Param1 = HalfTy;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmaxnmavq_f32:
    Param0 = Intrinsic::arm_mve_maxnmav;
    Param1 = FloatTy;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmaxnmvq_f16:
    Param0 = Intrinsic::arm_mve_maxnmv;
    Param1 = HalfTy;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmaxnmvq_f32:
    Param0 = Intrinsic::arm_mve_maxnmv;
    Param1 = FloatTy;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vminnmavq_f16:
    Param0 = Intrinsic::arm_mve_minnmav;
    Param1 = HalfTy;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vminnmavq_f32:
    Param0 = Intrinsic::arm_mve_minnmav;
    Param1 = FloatTy;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vminnmvq_f16:
    Param0 = Intrinsic::arm_mve_minnmv;
    Param1 = HalfTy;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vminnmvq_f32:
    Param0 = Intrinsic::arm_mve_minnmv;
    Param1 = FloatTy;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Param0, {Param1, Param2}), {Val0, Val1});
  return (Val2);
}
case ARM::BI__builtin_arm_mve_vddupq_n_u16:
case ARM::BI__builtin_arm_mve_vddupq_n_u32:
case ARM::BI__builtin_arm_mve_vddupq_n_u8:
case ARM::BI__builtin_arm_mve_vidupq_n_u16:
case ARM::BI__builtin_arm_mve_vidupq_n_u32:
case ARM::BI__builtin_arm_mve_vidupq_n_u8: {
  Intrinsic::ID  Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vddupq_n_u16:
    Param0 = Intrinsic::arm_mve_vddup;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vddupq_n_u32:
    Param0 = Intrinsic::arm_mve_vddup;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vddupq_n_u8:
    Param0 = Intrinsic::arm_mve_vddup;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vidupq_n_u16:
    Param0 = Intrinsic::arm_mve_vidup;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vidupq_n_u32:
    Param0 = Intrinsic::arm_mve_vidup;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vidupq_n_u8:
    Param0 = Intrinsic::arm_mve_vidup;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Param0, {Param1}), {Val0, Val1});
  return Builder.CreateExtractValue(Val2, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vmaxavq_s32:
case ARM::BI__builtin_arm_mve_vminavq_s32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmaxavq_s32:
    Param0 = Intrinsic::arm_mve_maxav;
    break;
  case ARM::BI__builtin_arm_mve_vminavq_s32:
    Param0 = Intrinsic::arm_mve_minav;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int32Ty, 4)}), {Val0, Val1});
  return Builder.CreateIntCast(Val2, Int32Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpeqq_f16:
case ARM::BI__builtin_arm_mve_vcmpeqq_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpeqq_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateFCmpOEQ(Val0, Val1);
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val2});
  return Builder.CreateIntCast(Val3, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpgeq_f16:
case ARM::BI__builtin_arm_mve_vcmpgeq_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpgeq_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgeq_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateFCmpOGE(Val0, Val1);
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val2});
  return Builder.CreateIntCast(Val3, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpgtq_f16:
case ARM::BI__builtin_arm_mve_vcmpgtq_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpgtq_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgtq_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateFCmpOGT(Val0, Val1);
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val2});
  return Builder.CreateIntCast(Val3, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpleq_f16:
case ARM::BI__builtin_arm_mve_vcmpleq_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpleq_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpleq_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateFCmpULE(Val0, Val1);
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val2});
  return Builder.CreateIntCast(Val3, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpltq_f16:
case ARM::BI__builtin_arm_mve_vcmpltq_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpltq_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpltq_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateFCmpULT(Val0, Val1);
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val2});
  return Builder.CreateIntCast(Val3, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpneq_f16:
case ARM::BI__builtin_arm_mve_vcmpneq_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpneq_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateFCmpUNE(Val0, Val1);
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val2});
  return Builder.CreateIntCast(Val3, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpeqq_s16:
case ARM::BI__builtin_arm_mve_vcmpeqq_s32:
case ARM::BI__builtin_arm_mve_vcmpeqq_s8:
case ARM::BI__builtin_arm_mve_vcmpeqq_u16:
case ARM::BI__builtin_arm_mve_vcmpeqq_u32:
case ARM::BI__builtin_arm_mve_vcmpeqq_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpeqq_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateICmpEQ(Val0, Val1);
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val2});
  return Builder.CreateIntCast(Val3, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpneq_s16:
case ARM::BI__builtin_arm_mve_vcmpneq_s32:
case ARM::BI__builtin_arm_mve_vcmpneq_s8:
case ARM::BI__builtin_arm_mve_vcmpneq_u16:
case ARM::BI__builtin_arm_mve_vcmpneq_u32:
case ARM::BI__builtin_arm_mve_vcmpneq_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpneq_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateICmpNE(Val0, Val1);
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val2});
  return Builder.CreateIntCast(Val3, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpgeq_s16:
case ARM::BI__builtin_arm_mve_vcmpgeq_s32:
case ARM::BI__builtin_arm_mve_vcmpgeq_s8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpgeq_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgeq_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgeq_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateICmpSGE(Val0, Val1);
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val2});
  return Builder.CreateIntCast(Val3, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vmaxq_s16:
case ARM::BI__builtin_arm_mve_vmaxq_s32:
case ARM::BI__builtin_arm_mve_vmaxq_s8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateICmpSGE(Val0, Val1);
  return Builder.CreateSelect(Val2, Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vcmpgtq_s16:
case ARM::BI__builtin_arm_mve_vcmpgtq_s32:
case ARM::BI__builtin_arm_mve_vcmpgtq_s8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpgtq_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgtq_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgtq_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateICmpSGT(Val0, Val1);
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val2});
  return Builder.CreateIntCast(Val3, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpleq_s16:
case ARM::BI__builtin_arm_mve_vcmpleq_s32:
case ARM::BI__builtin_arm_mve_vcmpleq_s8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpleq_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpleq_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpleq_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateICmpSLE(Val0, Val1);
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val2});
  return Builder.CreateIntCast(Val3, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vminq_s16:
case ARM::BI__builtin_arm_mve_vminq_s32:
case ARM::BI__builtin_arm_mve_vminq_s8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateICmpSLE(Val0, Val1);
  return Builder.CreateSelect(Val2, Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vcmpltq_s16:
case ARM::BI__builtin_arm_mve_vcmpltq_s32:
case ARM::BI__builtin_arm_mve_vcmpltq_s8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpltq_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpltq_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpltq_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateICmpSLT(Val0, Val1);
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val2});
  return Builder.CreateIntCast(Val3, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpcsq_u16:
case ARM::BI__builtin_arm_mve_vcmpcsq_u32:
case ARM::BI__builtin_arm_mve_vcmpcsq_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpcsq_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpcsq_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpcsq_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateICmpUGE(Val0, Val1);
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val2});
  return Builder.CreateIntCast(Val3, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vmaxq_u16:
case ARM::BI__builtin_arm_mve_vmaxq_u32:
case ARM::BI__builtin_arm_mve_vmaxq_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateICmpUGE(Val0, Val1);
  return Builder.CreateSelect(Val2, Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vcmphiq_u16:
case ARM::BI__builtin_arm_mve_vcmphiq_u32:
case ARM::BI__builtin_arm_mve_vcmphiq_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmphiq_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmphiq_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmphiq_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateICmpUGT(Val0, Val1);
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val2});
  return Builder.CreateIntCast(Val3, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vminq_u16:
case ARM::BI__builtin_arm_mve_vminq_u32:
case ARM::BI__builtin_arm_mve_vminq_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateICmpULE(Val0, Val1);
  return Builder.CreateSelect(Val2, Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vbicq_n_s32:
case ARM::BI__builtin_arm_mve_vbicq_n_u32: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vbicq_n_s32:
    Param0 = true;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_n_u32:
    Param0 = false;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateIntCast(Val1, Int32Ty, Param0);
  Value *Val3 = ARMMVEVectorSplat(Builder, Val2);
  Value *Val4 = Builder.CreateNot(Val3);
  return Builder.CreateAnd(Val0, Val4);
}
case ARM::BI__builtin_arm_mve_vorrq_n_s32:
case ARM::BI__builtin_arm_mve_vorrq_n_u32: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vorrq_n_s32:
    Param0 = true;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_n_u32:
    Param0 = false;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateIntCast(Val1, Int32Ty, Param0);
  Value *Val3 = ARMMVEVectorSplat(Builder, Val2);
  return Builder.CreateOr(Val0, Val3);
}
case ARM::BI__builtin_arm_mve_vbicq_n_s16:
case ARM::BI__builtin_arm_mve_vbicq_n_u16: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vbicq_n_s16:
    Param0 = true;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_n_u16:
    Param0 = false;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateIntCast(Val1, Int32Ty, false);
  Value *Val3 = Builder.CreateIntCast(Val2, Int16Ty, Param0);
  Value *Val4 = ARMMVEVectorSplat(Builder, Val3);
  Value *Val5 = Builder.CreateNot(Val4);
  return Builder.CreateAnd(Val0, Val5);
}
case ARM::BI__builtin_arm_mve_vorrq_n_s16:
case ARM::BI__builtin_arm_mve_vorrq_n_u16: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vorrq_n_s16:
    Param0 = true;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_n_u16:
    Param0 = false;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateIntCast(Val1, Int32Ty, false);
  Value *Val3 = Builder.CreateIntCast(Val2, Int16Ty, Param0);
  Value *Val4 = ARMMVEVectorSplat(Builder, Val3);
  return Builder.CreateOr(Val0, Val4);
}
case ARM::BI__builtin_arm_mve_vshlq_n_s16:
case ARM::BI__builtin_arm_mve_vshlq_n_s32:
case ARM::BI__builtin_arm_mve_vshlq_n_s8:
case ARM::BI__builtin_arm_mve_vshlq_n_u16:
case ARM::BI__builtin_arm_mve_vshlq_n_u32:
case ARM::BI__builtin_arm_mve_vshlq_n_u8: {
  llvm::Type * Param0;
  bool  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vshlq_n_s16:
    Param0 = Int16Ty;
    Param1 = true;
    break;
  case ARM::BI__builtin_arm_mve_vshlq_n_s32:
    Param0 = Int32Ty;
    Param1 = true;
    break;
  case ARM::BI__builtin_arm_mve_vshlq_n_s8:
    Param0 = Int8Ty;
    Param1 = true;
    break;
  case ARM::BI__builtin_arm_mve_vshlq_n_u16:
    Param0 = Int16Ty;
    Param1 = false;
    break;
  case ARM::BI__builtin_arm_mve_vshlq_n_u32:
    Param0 = Int32Ty;
    Param1 = false;
    break;
  case ARM::BI__builtin_arm_mve_vshlq_n_u8:
    Param0 = Int8Ty;
    Param1 = false;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateIntCast(Val1, Param0, Param1);
  Value *Val3 = ARMMVEVectorSplat(Builder, Val2);
  return Builder.CreateShl(Val0, Val3);
}
case ARM::BI__builtin_arm_mve_vmlasq_n_s16:
case ARM::BI__builtin_arm_mve_vmlasq_n_s32:
case ARM::BI__builtin_arm_mve_vmlasq_n_s8:
case ARM::BI__builtin_arm_mve_vmlasq_n_u16:
case ARM::BI__builtin_arm_mve_vmlasq_n_u32:
case ARM::BI__builtin_arm_mve_vmlasq_n_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateMul(Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  Value *Val4 = ARMMVEVectorSplat(Builder, Val3);
  return Builder.CreateAdd(Val2, Val4);
}
case ARM::BI__builtin_arm_mve_vbicq_s16:
case ARM::BI__builtin_arm_mve_vbicq_s32:
case ARM::BI__builtin_arm_mve_vbicq_s8:
case ARM::BI__builtin_arm_mve_vbicq_u16:
case ARM::BI__builtin_arm_mve_vbicq_u32:
case ARM::BI__builtin_arm_mve_vbicq_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateNot(Val1);
  return Builder.CreateAnd(Val0, Val2);
}
case ARM::BI__builtin_arm_mve_vornq_s16:
case ARM::BI__builtin_arm_mve_vornq_s32:
case ARM::BI__builtin_arm_mve_vornq_s8:
case ARM::BI__builtin_arm_mve_vornq_u16:
case ARM::BI__builtin_arm_mve_vornq_u32:
case ARM::BI__builtin_arm_mve_vornq_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = Builder.CreateNot(Val1);
  return Builder.CreateOr(Val0, Val2);
}
case ARM::BI__builtin_arm_mve_vadcq_m_s32:
case ARM::BI__builtin_arm_mve_vadcq_m_u32:
case ARM::BI__builtin_arm_mve_vsbcq_m_s32:
case ARM::BI__builtin_arm_mve_vsbcq_m_u32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vadcq_m_s32:
    Param0 = Intrinsic::arm_mve_vadc_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vadcq_m_u32:
    Param0 = Intrinsic::arm_mve_vadc_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsbcq_m_s32:
    Param0 = Intrinsic::arm_mve_vsbc_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsbcq_m_u32:
    Param0 = Intrinsic::arm_mve_vsbc_predicated;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Address Val3 = EmitPointerWithAlignment(E->getArg(3));
  Value *Val4 = Builder.CreateLoad(Val3);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, 29);
  Value *Val6 = Builder.CreateShl(Val4, Val5);
  Value *Val7 = EmitScalarExpr(E->getArg(4));
  Value *Val8 = Builder.CreateIntCast(Val7, Int32Ty, false);
  Value *Val9 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val8});
  Value *Val10 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val0, Val1, Val2, Val6, Val9});
  Value *Val11 = llvm::ConstantInt::get(Int32Ty, 1);
  Value *Val12 = Builder.CreateExtractValue(Val10, static_cast<unsigned>(1));
  Value *Val13 = llvm::ConstantInt::get(Int32Ty, 29);
  Value *Val14 = Builder.CreateLShr(Val12, Val13);
  Value *Val15 = Builder.CreateAnd(Val11, Val14);
  Builder.CreateStore(Val15, Val3);
  return Builder.CreateExtractValue(Val10, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vdwdupq_n_u16:
case ARM::BI__builtin_arm_mve_vdwdupq_n_u32:
case ARM::BI__builtin_arm_mve_vdwdupq_n_u8:
case ARM::BI__builtin_arm_mve_viwdupq_n_u16:
case ARM::BI__builtin_arm_mve_viwdupq_n_u32:
case ARM::BI__builtin_arm_mve_viwdupq_n_u8: {
  Intrinsic::ID  Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vdwdupq_n_u16:
    Param0 = Intrinsic::arm_mve_vdwdup;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vdwdupq_n_u32:
    Param0 = Intrinsic::arm_mve_vdwdup;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vdwdupq_n_u8:
    Param0 = Intrinsic::arm_mve_vdwdup;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_n_u16:
    Param0 = Intrinsic::arm_mve_viwdup;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_n_u32:
    Param0 = Intrinsic::arm_mve_viwdup;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_n_u8:
    Param0 = Intrinsic::arm_mve_viwdup;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Param0, {Param1}), {Val0, Val1, Val2});
  return Builder.CreateExtractValue(Val3, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vmaxnmavq_p_f16:
case ARM::BI__builtin_arm_mve_vmaxnmavq_p_f32:
case ARM::BI__builtin_arm_mve_vmaxnmvq_p_f16:
case ARM::BI__builtin_arm_mve_vmaxnmvq_p_f32:
case ARM::BI__builtin_arm_mve_vminnmavq_p_f16:
case ARM::BI__builtin_arm_mve_vminnmavq_p_f32:
case ARM::BI__builtin_arm_mve_vminnmvq_p_f16:
case ARM::BI__builtin_arm_mve_vminnmvq_p_f32: {
  llvm::Type * Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmaxnmavq_p_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_maxnmav_predicated;
    Param2 = HalfTy;
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmaxnmavq_p_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_maxnmav_predicated;
    Param2 = FloatTy;
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmaxnmvq_p_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_maxnmv_predicated;
    Param2 = HalfTy;
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmaxnmvq_p_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_maxnmv_predicated;
    Param2 = FloatTy;
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vminnmavq_p_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_minnmav_predicated;
    Param2 = HalfTy;
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vminnmavq_p_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_minnmav_predicated;
    Param2 = FloatTy;
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vminnmvq_p_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_minnmv_predicated;
    Param2 = HalfTy;
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vminnmvq_p_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_minnmv_predicated;
    Param2 = FloatTy;
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val3});
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2, Param3, Param0}), {Val0, Val1, Val4});
  return (Val5);
}
case ARM::BI__builtin_arm_mve_vldrdq_gather_base_z_s64:
case ARM::BI__builtin_arm_mve_vldrdq_gather_base_z_u64:
case ARM::BI__builtin_arm_mve_vldrwq_gather_base_z_f32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_base_z_s32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_base_z_u32: {
  llvm::Type * Param0;
  llvm::Type * Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vldrdq_gather_base_z_s64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vldrdq_gather_base_z_u64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_base_z_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_base_z_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_base_z_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val3});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vldr_gather_base_predicated, {Param1, Param2, Param0}), {Val0, Val1, Val4});
}
case ARM::BI__builtin_arm_mve_vmaxaq_m_s16:
case ARM::BI__builtin_arm_mve_vmaxaq_m_s32:
case ARM::BI__builtin_arm_mve_vmaxaq_m_s8:
case ARM::BI__builtin_arm_mve_vmaxnmaq_m_f16:
case ARM::BI__builtin_arm_mve_vmaxnmaq_m_f32:
case ARM::BI__builtin_arm_mve_vminaq_m_s16:
case ARM::BI__builtin_arm_mve_vminaq_m_s32:
case ARM::BI__builtin_arm_mve_vminaq_m_s8:
case ARM::BI__builtin_arm_mve_vminnmaq_m_f16:
case ARM::BI__builtin_arm_mve_vminnmaq_m_f32: {
  llvm::Type * Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmaxaq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vmaxa_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmaxaq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vmaxa_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmaxaq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vmaxa_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmaxnmaq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vmaxnma_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmaxnmaq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vmaxnma_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vminaq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vmina_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vminaq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vmina_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vminaq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vmina_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vminnmaq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vminnma_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vminnmaq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vminnma_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val3});
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2, Param0}), {Val0, Val1, Val4});
}
case ARM::BI__builtin_arm_mve_vmaxavq_p_s32:
case ARM::BI__builtin_arm_mve_vminavq_p_s32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmaxavq_p_s32:
    Param0 = Intrinsic::arm_mve_maxav_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vminavq_p_s32:
    Param0 = Intrinsic::arm_mve_minav_predicated;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val3});
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val0, Val1, Val4});
  return Builder.CreateIntCast(Val5, Int32Ty, false);
}
case ARM::BI__builtin_arm_mve_vmlaq_m_n_s16:
case ARM::BI__builtin_arm_mve_vmlaq_m_n_s8:
case ARM::BI__builtin_arm_mve_vmlaq_m_n_u16:
case ARM::BI__builtin_arm_mve_vmlaq_m_n_u8:
case ARM::BI__builtin_arm_mve_vmlasq_m_n_s16:
case ARM::BI__builtin_arm_mve_vmlasq_m_n_s8:
case ARM::BI__builtin_arm_mve_vmlasq_m_n_u16:
case ARM::BI__builtin_arm_mve_vmlasq_m_n_u8:
case ARM::BI__builtin_arm_mve_vqdmlahq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqdmlahq_m_n_s8:
case ARM::BI__builtin_arm_mve_vqdmlashq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqdmlashq_m_n_s8:
case ARM::BI__builtin_arm_mve_vqrdmlahq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqrdmlahq_m_n_s8:
case ARM::BI__builtin_arm_mve_vqrdmlashq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqrdmlashq_m_n_s8: {
  llvm::Type * Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmlaq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vmla_n_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlaq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vmla_n_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmlaq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vmla_n_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlaq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vmla_n_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmlasq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vmlas_n_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlasq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vmlas_n_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmlasq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vmlas_n_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlasq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vmlas_n_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlahq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vqdmlah_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlahq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vqdmlah_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlashq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vqdmlash_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlashq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vqdmlash_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlahq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vqrdmlah_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlahq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vqrdmlah_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlashq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vqrdmlash_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlashq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vqrdmlash_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = EmitScalarExpr(E->getArg(3));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2, Param0}), {Val0, Val1, Val3, Val6});
}
case ARM::BI__builtin_arm_mve_vqdmlahq_n_s16:
case ARM::BI__builtin_arm_mve_vqdmlahq_n_s8:
case ARM::BI__builtin_arm_mve_vqdmlashq_n_s16:
case ARM::BI__builtin_arm_mve_vqdmlashq_n_s8:
case ARM::BI__builtin_arm_mve_vqrdmlahq_n_s16:
case ARM::BI__builtin_arm_mve_vqrdmlahq_n_s8:
case ARM::BI__builtin_arm_mve_vqrdmlashq_n_s16:
case ARM::BI__builtin_arm_mve_vqrdmlashq_n_s8: {
  Intrinsic::ID  Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vqdmlahq_n_s16:
    Param0 = Intrinsic::arm_mve_vqdmlah;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlahq_n_s8:
    Param0 = Intrinsic::arm_mve_vqdmlah;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlashq_n_s16:
    Param0 = Intrinsic::arm_mve_vqdmlash;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlashq_n_s8:
    Param0 = Intrinsic::arm_mve_vqdmlash;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlahq_n_s16:
    Param0 = Intrinsic::arm_mve_vqrdmlah;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlahq_n_s8:
    Param0 = Intrinsic::arm_mve_vqrdmlah;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlashq_n_s16:
    Param0 = Intrinsic::arm_mve_vqrdmlash;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlashq_n_s8:
    Param0 = Intrinsic::arm_mve_vqrdmlash;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {Param1}), {Val0, Val1, Val3});
}
case ARM::BI__builtin_arm_mve_vddupq_m_n_u16:
case ARM::BI__builtin_arm_mve_vddupq_m_n_u32:
case ARM::BI__builtin_arm_mve_vddupq_m_n_u8:
case ARM::BI__builtin_arm_mve_vidupq_m_n_u16:
case ARM::BI__builtin_arm_mve_vidupq_m_n_u32:
case ARM::BI__builtin_arm_mve_vidupq_m_n_u8: {
  llvm::Type * Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vddupq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vddup_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vddupq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vddup_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vddupq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vddup_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vidupq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vidup_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vidupq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vidup_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vidupq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vidup_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = EmitScalarExpr(E->getArg(3));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val4});
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2, Param0}), {Val0, Val1, Val2, Val5});
  return Builder.CreateExtractValue(Val6, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vstrdq_scatter_base_p_s64:
case ARM::BI__builtin_arm_mve_vstrdq_scatter_base_p_u64:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_p_f32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_p_s32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_p_u32: {
  llvm::Type * Param0;
  llvm::Type * Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vstrdq_scatter_base_p_s64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vstrdq_scatter_base_p_u64:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_p_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_p_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_p_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = EmitScalarExpr(E->getArg(3));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val4});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vstr_scatter_base_predicated, {Param1, Param2, Param0}), {Val0, Val1, Val2, Val5});
}
case ARM::BI__builtin_arm_mve_vbrsrq_m_n_f16:
case ARM::BI__builtin_arm_mve_vbrsrq_m_n_f32:
case ARM::BI__builtin_arm_mve_vbrsrq_m_n_s16:
case ARM::BI__builtin_arm_mve_vbrsrq_m_n_s32:
case ARM::BI__builtin_arm_mve_vbrsrq_m_n_s8:
case ARM::BI__builtin_arm_mve_vbrsrq_m_n_u16:
case ARM::BI__builtin_arm_mve_vbrsrq_m_n_u32:
case ARM::BI__builtin_arm_mve_vbrsrq_m_n_u8:
case ARM::BI__builtin_arm_mve_vmlaq_m_n_s32:
case ARM::BI__builtin_arm_mve_vmlaq_m_n_u32:
case ARM::BI__builtin_arm_mve_vmlasq_m_n_s32:
case ARM::BI__builtin_arm_mve_vmlasq_m_n_u32:
case ARM::BI__builtin_arm_mve_vqdmlahq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqdmlashq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqrdmlahq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqrdmlashq_m_n_s32:
case ARM::BI__builtin_arm_mve_vsliq_m_n_s16:
case ARM::BI__builtin_arm_mve_vsliq_m_n_s32:
case ARM::BI__builtin_arm_mve_vsliq_m_n_s8:
case ARM::BI__builtin_arm_mve_vsliq_m_n_u16:
case ARM::BI__builtin_arm_mve_vsliq_m_n_u32:
case ARM::BI__builtin_arm_mve_vsliq_m_n_u8:
case ARM::BI__builtin_arm_mve_vsriq_m_n_s16:
case ARM::BI__builtin_arm_mve_vsriq_m_n_s32:
case ARM::BI__builtin_arm_mve_vsriq_m_n_s8:
case ARM::BI__builtin_arm_mve_vsriq_m_n_u16:
case ARM::BI__builtin_arm_mve_vsriq_m_n_u32:
case ARM::BI__builtin_arm_mve_vsriq_m_n_u8: {
  llvm::Type * Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vbrsrq_m_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vbrsr_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vbrsrq_m_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vbrsr_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vbrsrq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vbrsr_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vbrsrq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vbrsr_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vbrsrq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vbrsr_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vbrsrq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vbrsr_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vbrsrq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vbrsr_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vbrsrq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vbrsr_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmlaq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vmla_n_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlaq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vmla_n_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlasq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vmlas_n_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlasq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vmlas_n_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlahq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vqdmlah_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlashq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vqdmlash_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlahq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vqrdmlah_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlashq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vqrdmlash_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vsliq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vsli_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vsliq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vsli_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vsliq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vsli_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vsliq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vsli_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vsliq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vsli_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vsliq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vsli_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vsriq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vsri_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vsriq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vsri_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vsriq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vsri_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vsriq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vsri_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vsriq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vsri_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vsriq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vsri_predicated;
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
case ARM::BI__builtin_arm_mve_vdwdupq_m_n_u16:
case ARM::BI__builtin_arm_mve_vdwdupq_m_n_u32:
case ARM::BI__builtin_arm_mve_vdwdupq_m_n_u8:
case ARM::BI__builtin_arm_mve_viwdupq_m_n_u16:
case ARM::BI__builtin_arm_mve_viwdupq_m_n_u32:
case ARM::BI__builtin_arm_mve_viwdupq_m_n_u8: {
  llvm::Type * Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vdwdupq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vdwdup_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vdwdupq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vdwdup_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vdwdupq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vdwdup_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_viwdup_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_viwdup_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_viwdup_predicated;
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
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2, Param0}), {Val0, Val1, Val2, Val3, Val6});
  return Builder.CreateExtractValue(Val7, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vadciq_m_s32:
case ARM::BI__builtin_arm_mve_vadciq_m_u32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, 0);
  Value *Val4 = EmitScalarExpr(E->getArg(4));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val5});
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vadc_predicated, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val0, Val1, Val2, Val3, Val6});
  Value *Val8 = llvm::ConstantInt::get(Int32Ty, 1);
  Value *Val9 = Builder.CreateExtractValue(Val7, static_cast<unsigned>(1));
  Value *Val10 = llvm::ConstantInt::get(Int32Ty, 29);
  Value *Val11 = Builder.CreateLShr(Val9, Val10);
  Value *Val12 = Builder.CreateAnd(Val8, Val11);
  Address Val13 = EmitPointerWithAlignment(E->getArg(3));
  Builder.CreateStore(Val12, Val13);
  return Builder.CreateExtractValue(Val7, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vsbciq_m_s32:
case ARM::BI__builtin_arm_mve_vsbciq_m_u32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, 1);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, 29);
  Value *Val5 = Builder.CreateShl(Val3, Val4);
  Value *Val6 = EmitScalarExpr(E->getArg(4));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val7});
  Value *Val9 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vsbc_predicated, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val0, Val1, Val2, Val5, Val8});
  Value *Val10 = llvm::ConstantInt::get(Int32Ty, 1);
  Value *Val11 = Builder.CreateExtractValue(Val9, static_cast<unsigned>(1));
  Value *Val12 = llvm::ConstantInt::get(Int32Ty, 29);
  Value *Val13 = Builder.CreateLShr(Val11, Val12);
  Value *Val14 = Builder.CreateAnd(Val10, Val13);
  Address Val15 = EmitPointerWithAlignment(E->getArg(3));
  Builder.CreateStore(Val14, Val15);
  return Builder.CreateExtractValue(Val9, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vqdmladhq_m_s16:
case ARM::BI__builtin_arm_mve_vqdmladhq_m_s32:
case ARM::BI__builtin_arm_mve_vqdmladhq_m_s8:
case ARM::BI__builtin_arm_mve_vqdmladhxq_m_s16:
case ARM::BI__builtin_arm_mve_vqdmladhxq_m_s32:
case ARM::BI__builtin_arm_mve_vqdmladhxq_m_s8:
case ARM::BI__builtin_arm_mve_vqdmlsdhq_m_s16:
case ARM::BI__builtin_arm_mve_vqdmlsdhq_m_s32:
case ARM::BI__builtin_arm_mve_vqdmlsdhq_m_s8:
case ARM::BI__builtin_arm_mve_vqdmlsdhxq_m_s16:
case ARM::BI__builtin_arm_mve_vqdmlsdhxq_m_s32:
case ARM::BI__builtin_arm_mve_vqdmlsdhxq_m_s8:
case ARM::BI__builtin_arm_mve_vqrdmladhq_m_s16:
case ARM::BI__builtin_arm_mve_vqrdmladhq_m_s32:
case ARM::BI__builtin_arm_mve_vqrdmladhq_m_s8:
case ARM::BI__builtin_arm_mve_vqrdmladhxq_m_s16:
case ARM::BI__builtin_arm_mve_vqrdmladhxq_m_s32:
case ARM::BI__builtin_arm_mve_vqrdmladhxq_m_s8:
case ARM::BI__builtin_arm_mve_vqrdmlsdhq_m_s16:
case ARM::BI__builtin_arm_mve_vqrdmlsdhq_m_s32:
case ARM::BI__builtin_arm_mve_vqrdmlsdhq_m_s8:
case ARM::BI__builtin_arm_mve_vqrdmlsdhxq_m_s16:
case ARM::BI__builtin_arm_mve_vqrdmlsdhxq_m_s32:
case ARM::BI__builtin_arm_mve_vqrdmlsdhxq_m_s8: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  llvm::Type * Param3;
  llvm::Type * Param4;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vqdmladhq_m_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmladhq_m_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmladhq_m_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqdmladhxq_m_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmladhxq_m_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmladhxq_m_s8:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlsdhq_m_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlsdhq_m_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlsdhq_m_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlsdhxq_m_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlsdhxq_m_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlsdhxq_m_s8:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmladhq_m_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmladhq_m_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmladhq_m_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmladhxq_m_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmladhxq_m_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmladhxq_m_s8:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlsdhq_m_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlsdhq_m_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlsdhq_m_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlsdhxq_m_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlsdhxq_m_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlsdhxq_m_s8:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, Param2);
  Value *Val6 = EmitScalarExpr(E->getArg(3));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param3}), {Val7});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vqdmlad_predicated, {Param4, Param3}), {Val0, Val1, Val2, Val3, Val4, Val5, Val8});
}
case ARM::BI__builtin_arm_mve_vqrshrnbq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqrshrnbq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqrshrnbq_m_n_u16:
case ARM::BI__builtin_arm_mve_vqrshrnbq_m_n_u32:
case ARM::BI__builtin_arm_mve_vqrshrntq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqrshrntq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqrshrntq_m_n_u16:
case ARM::BI__builtin_arm_mve_vqrshrntq_m_n_u32:
case ARM::BI__builtin_arm_mve_vqrshrunbq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqrshrunbq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqrshruntq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqrshruntq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqshrnbq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqshrnbq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqshrnbq_m_n_u16:
case ARM::BI__builtin_arm_mve_vqshrnbq_m_n_u32:
case ARM::BI__builtin_arm_mve_vqshrntq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqshrntq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqshrntq_m_n_u16:
case ARM::BI__builtin_arm_mve_vqshrntq_m_n_u32:
case ARM::BI__builtin_arm_mve_vqshrunbq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqshrunbq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqshruntq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqshruntq_m_n_s32:
case ARM::BI__builtin_arm_mve_vrshrnbq_m_n_s16:
case ARM::BI__builtin_arm_mve_vrshrnbq_m_n_s32:
case ARM::BI__builtin_arm_mve_vrshrnbq_m_n_u16:
case ARM::BI__builtin_arm_mve_vrshrnbq_m_n_u32:
case ARM::BI__builtin_arm_mve_vrshrntq_m_n_s16:
case ARM::BI__builtin_arm_mve_vrshrntq_m_n_s32:
case ARM::BI__builtin_arm_mve_vrshrntq_m_n_u16:
case ARM::BI__builtin_arm_mve_vrshrntq_m_n_u32:
case ARM::BI__builtin_arm_mve_vshrnbq_m_n_s16:
case ARM::BI__builtin_arm_mve_vshrnbq_m_n_s32:
case ARM::BI__builtin_arm_mve_vshrnbq_m_n_u16:
case ARM::BI__builtin_arm_mve_vshrnbq_m_n_u32:
case ARM::BI__builtin_arm_mve_vshrntq_m_n_s16:
case ARM::BI__builtin_arm_mve_vshrntq_m_n_s32:
case ARM::BI__builtin_arm_mve_vshrntq_m_n_u16:
case ARM::BI__builtin_arm_mve_vshrntq_m_n_u32: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  uint32_t  Param3;
  uint32_t  Param4;
  llvm::Type * Param5;
  llvm::Type * Param6;
  llvm::Type * Param7;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vqrshrnbq_m_n_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrnbq_m_n_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrnbq_m_n_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = 1;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrnbq_m_n_u32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = 1;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrntq_m_n_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrntq_m_n_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrntq_m_n_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = 1;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrntq_m_n_u32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = 1;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrunbq_m_n_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrunbq_m_n_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshruntq_m_n_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshruntq_m_n_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshrnbq_m_n_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshrnbq_m_n_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshrnbq_m_n_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = 1;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshrnbq_m_n_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = 1;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshrntq_m_n_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshrntq_m_n_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshrntq_m_n_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = 1;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshrntq_m_n_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = 1;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshrunbq_m_n_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshrunbq_m_n_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshruntq_m_n_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshruntq_m_n_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshrnbq_m_n_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshrnbq_m_n_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshrnbq_m_n_u16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = 1;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshrnbq_m_n_u32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = 1;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshrntq_m_n_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshrntq_m_n_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshrntq_m_n_u16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = 1;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshrntq_m_n_u32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = 1;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshrnbq_m_n_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshrnbq_m_n_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshrnbq_m_n_u16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = 1;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshrnbq_m_n_u32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = 1;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshrntq_m_n_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshrntq_m_n_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshrntq_m_n_u16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = 1;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param6 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param7 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshrntq_m_n_u32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = 1;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param7 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, Param2);
  Value *Val6 = llvm::ConstantInt::get(Int32Ty, Param3);
  Value *Val7 = llvm::ConstantInt::get(Int32Ty, Param4);
  Value *Val8 = EmitScalarExpr(E->getArg(3));
  Value *Val9 = Builder.CreateIntCast(Val8, Int32Ty, false);
  Value *Val10 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param5}), {Val9});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vshrn_predicated, {Param6, Param7, Param5}), {Val0, Val1, Val2, Val3, Val4, Val5, Val6, Val7, Val10});
}
case ARM::BI__builtin_arm_mve_vqrshrnbq_n_s16:
case ARM::BI__builtin_arm_mve_vqrshrnbq_n_s32:
case ARM::BI__builtin_arm_mve_vqrshrnbq_n_u16:
case ARM::BI__builtin_arm_mve_vqrshrnbq_n_u32:
case ARM::BI__builtin_arm_mve_vqrshrntq_n_s16:
case ARM::BI__builtin_arm_mve_vqrshrntq_n_s32:
case ARM::BI__builtin_arm_mve_vqrshrntq_n_u16:
case ARM::BI__builtin_arm_mve_vqrshrntq_n_u32:
case ARM::BI__builtin_arm_mve_vqrshrunbq_n_s16:
case ARM::BI__builtin_arm_mve_vqrshrunbq_n_s32:
case ARM::BI__builtin_arm_mve_vqrshruntq_n_s16:
case ARM::BI__builtin_arm_mve_vqrshruntq_n_s32:
case ARM::BI__builtin_arm_mve_vqshrnbq_n_s16:
case ARM::BI__builtin_arm_mve_vqshrnbq_n_s32:
case ARM::BI__builtin_arm_mve_vqshrnbq_n_u16:
case ARM::BI__builtin_arm_mve_vqshrnbq_n_u32:
case ARM::BI__builtin_arm_mve_vqshrntq_n_s16:
case ARM::BI__builtin_arm_mve_vqshrntq_n_s32:
case ARM::BI__builtin_arm_mve_vqshrntq_n_u16:
case ARM::BI__builtin_arm_mve_vqshrntq_n_u32:
case ARM::BI__builtin_arm_mve_vqshrunbq_n_s16:
case ARM::BI__builtin_arm_mve_vqshrunbq_n_s32:
case ARM::BI__builtin_arm_mve_vqshruntq_n_s16:
case ARM::BI__builtin_arm_mve_vqshruntq_n_s32:
case ARM::BI__builtin_arm_mve_vrshrnbq_n_s16:
case ARM::BI__builtin_arm_mve_vrshrnbq_n_s32:
case ARM::BI__builtin_arm_mve_vrshrnbq_n_u16:
case ARM::BI__builtin_arm_mve_vrshrnbq_n_u32:
case ARM::BI__builtin_arm_mve_vrshrntq_n_s16:
case ARM::BI__builtin_arm_mve_vrshrntq_n_s32:
case ARM::BI__builtin_arm_mve_vrshrntq_n_u16:
case ARM::BI__builtin_arm_mve_vrshrntq_n_u32:
case ARM::BI__builtin_arm_mve_vshrnbq_n_s16:
case ARM::BI__builtin_arm_mve_vshrnbq_n_s32:
case ARM::BI__builtin_arm_mve_vshrnbq_n_u16:
case ARM::BI__builtin_arm_mve_vshrnbq_n_u32:
case ARM::BI__builtin_arm_mve_vshrntq_n_s16:
case ARM::BI__builtin_arm_mve_vshrntq_n_s32:
case ARM::BI__builtin_arm_mve_vshrntq_n_u16:
case ARM::BI__builtin_arm_mve_vshrntq_n_u32: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  uint32_t  Param3;
  uint32_t  Param4;
  llvm::Type * Param5;
  llvm::Type * Param6;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vqrshrnbq_n_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrnbq_n_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrnbq_n_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = 1;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrnbq_n_u32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = 1;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrntq_n_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrntq_n_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrntq_n_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = 1;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrntq_n_u32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = 1;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrunbq_n_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshrunbq_n_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshruntq_n_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshruntq_n_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshrnbq_n_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshrnbq_n_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshrnbq_n_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = 1;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshrnbq_n_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = 1;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshrntq_n_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshrntq_n_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshrntq_n_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = 1;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshrntq_n_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = 1;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshrunbq_n_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshrunbq_n_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshruntq_n_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshruntq_n_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshrnbq_n_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshrnbq_n_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshrnbq_n_u16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = 1;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshrnbq_n_u32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = 1;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshrntq_n_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshrntq_n_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshrntq_n_u16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = 1;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshrntq_n_u32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = 1;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshrnbq_n_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshrnbq_n_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = 0;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshrnbq_n_u16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = 1;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshrnbq_n_u32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = 1;
    Param4 = 0;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshrntq_n_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshrntq_n_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = 0;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshrntq_n_u16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = 1;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param6 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshrntq_n_u32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = 1;
    Param4 = 1;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param6 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, Param2);
  Value *Val6 = llvm::ConstantInt::get(Int32Ty, Param3);
  Value *Val7 = llvm::ConstantInt::get(Int32Ty, Param4);
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vshrn, {Param5, Param6}), {Val0, Val1, Val2, Val3, Val4, Val5, Val6, Val7});
}
case ARM::BI__builtin_arm_mve_vqdmladhq_s16:
case ARM::BI__builtin_arm_mve_vqdmladhq_s32:
case ARM::BI__builtin_arm_mve_vqdmladhq_s8:
case ARM::BI__builtin_arm_mve_vqdmladhxq_s16:
case ARM::BI__builtin_arm_mve_vqdmladhxq_s32:
case ARM::BI__builtin_arm_mve_vqdmladhxq_s8:
case ARM::BI__builtin_arm_mve_vqdmlsdhq_s16:
case ARM::BI__builtin_arm_mve_vqdmlsdhq_s32:
case ARM::BI__builtin_arm_mve_vqdmlsdhq_s8:
case ARM::BI__builtin_arm_mve_vqdmlsdhxq_s16:
case ARM::BI__builtin_arm_mve_vqdmlsdhxq_s32:
case ARM::BI__builtin_arm_mve_vqdmlsdhxq_s8:
case ARM::BI__builtin_arm_mve_vqrdmladhq_s16:
case ARM::BI__builtin_arm_mve_vqrdmladhq_s32:
case ARM::BI__builtin_arm_mve_vqrdmladhq_s8:
case ARM::BI__builtin_arm_mve_vqrdmladhxq_s16:
case ARM::BI__builtin_arm_mve_vqrdmladhxq_s32:
case ARM::BI__builtin_arm_mve_vqrdmladhxq_s8:
case ARM::BI__builtin_arm_mve_vqrdmlsdhq_s16:
case ARM::BI__builtin_arm_mve_vqrdmlsdhq_s32:
case ARM::BI__builtin_arm_mve_vqrdmlsdhq_s8:
case ARM::BI__builtin_arm_mve_vqrdmlsdhxq_s16:
case ARM::BI__builtin_arm_mve_vqrdmlsdhxq_s32:
case ARM::BI__builtin_arm_mve_vqrdmlsdhxq_s8: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vqdmladhq_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmladhq_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmladhq_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqdmladhxq_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmladhxq_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmladhxq_s8:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlsdhq_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlsdhq_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlsdhq_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlsdhxq_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlsdhxq_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlsdhxq_s8:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmladhq_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmladhq_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmladhq_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmladhxq_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmladhxq_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmladhxq_s8:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlsdhq_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlsdhq_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlsdhq_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlsdhxq_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlsdhxq_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlsdhxq_s8:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, Param2);
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vqdmlad, {Param3}), {Val0, Val1, Val2, Val3, Val4, Val5});
}
case ARM::BI__builtin_arm_mve_vstrdq_scatter_base_s64:
case ARM::BI__builtin_arm_mve_vstrdq_scatter_base_u64:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_f32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_s32:
case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_u32: {
  llvm::Type * Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vstrdq_scatter_base_s64:
    Param0 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vstrdq_scatter_base_u64:
    Param0 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_f32:
    Param0 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_s32:
    Param0 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_scatter_base_u32:
    Param0 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vstr_scatter_base, {Param0, Param1}), {Val0, Val1, Val2});
}
case ARM::BI__builtin_arm_mve_vqdmlahq_n_s32:
case ARM::BI__builtin_arm_mve_vqdmlashq_n_s32:
case ARM::BI__builtin_arm_mve_vqrdmlahq_n_s32:
case ARM::BI__builtin_arm_mve_vqrdmlashq_n_s32:
case ARM::BI__builtin_arm_mve_vsliq_n_s16:
case ARM::BI__builtin_arm_mve_vsliq_n_s32:
case ARM::BI__builtin_arm_mve_vsliq_n_s8:
case ARM::BI__builtin_arm_mve_vsliq_n_u16:
case ARM::BI__builtin_arm_mve_vsliq_n_u32:
case ARM::BI__builtin_arm_mve_vsliq_n_u8:
case ARM::BI__builtin_arm_mve_vsriq_n_s16:
case ARM::BI__builtin_arm_mve_vsriq_n_s32:
case ARM::BI__builtin_arm_mve_vsriq_n_s8:
case ARM::BI__builtin_arm_mve_vsriq_n_u16:
case ARM::BI__builtin_arm_mve_vsriq_n_u32:
case ARM::BI__builtin_arm_mve_vsriq_n_u8: {
  Intrinsic::ID  Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vqdmlahq_n_s32:
    Param0 = Intrinsic::arm_mve_vqdmlah;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmlashq_n_s32:
    Param0 = Intrinsic::arm_mve_vqdmlash;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlahq_n_s32:
    Param0 = Intrinsic::arm_mve_vqrdmlah;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmlashq_n_s32:
    Param0 = Intrinsic::arm_mve_vqrdmlash;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vsliq_n_s16:
    Param0 = Intrinsic::arm_mve_vsli;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vsliq_n_s32:
    Param0 = Intrinsic::arm_mve_vsli;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vsliq_n_s8:
    Param0 = Intrinsic::arm_mve_vsli;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vsliq_n_u16:
    Param0 = Intrinsic::arm_mve_vsli;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vsliq_n_u32:
    Param0 = Intrinsic::arm_mve_vsli;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vsliq_n_u8:
    Param0 = Intrinsic::arm_mve_vsli;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vsriq_n_s16:
    Param0 = Intrinsic::arm_mve_vsri;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vsriq_n_s32:
    Param0 = Intrinsic::arm_mve_vsri;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vsriq_n_s8:
    Param0 = Intrinsic::arm_mve_vsri;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vsriq_n_u16:
    Param0 = Intrinsic::arm_mve_vsri;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vsriq_n_u32:
    Param0 = Intrinsic::arm_mve_vsri;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vsriq_n_u8:
    Param0 = Intrinsic::arm_mve_vsri;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {Param1}), {Val0, Val1, Val2});
}
case ARM::BI__builtin_arm_mve_vadciq_s32:
case ARM::BI__builtin_arm_mve_vadciq_u32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, 0);
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vadc, {llvm::FixedVectorType::get(Int32Ty, 4)}), {Val0, Val1, Val2});
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, 1);
  Value *Val5 = Builder.CreateExtractValue(Val3, static_cast<unsigned>(1));
  Value *Val6 = llvm::ConstantInt::get(Int32Ty, 29);
  Value *Val7 = Builder.CreateLShr(Val5, Val6);
  Value *Val8 = Builder.CreateAnd(Val4, Val7);
  Address Val9 = EmitPointerWithAlignment(E->getArg(2));
  Builder.CreateStore(Val8, Val9);
  return Builder.CreateExtractValue(Val3, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vsbciq_s32:
case ARM::BI__builtin_arm_mve_vsbciq_u32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, 1);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, 29);
  Value *Val4 = Builder.CreateShl(Val2, Val3);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vsbc, {llvm::FixedVectorType::get(Int32Ty, 4)}), {Val0, Val1, Val4});
  Value *Val6 = llvm::ConstantInt::get(Int32Ty, 1);
  Value *Val7 = Builder.CreateExtractValue(Val5, static_cast<unsigned>(1));
  Value *Val8 = llvm::ConstantInt::get(Int32Ty, 29);
  Value *Val9 = Builder.CreateLShr(Val7, Val8);
  Value *Val10 = Builder.CreateAnd(Val6, Val9);
  Address Val11 = EmitPointerWithAlignment(E->getArg(2));
  Builder.CreateStore(Val10, Val11);
  return Builder.CreateExtractValue(Val5, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vmaxvq_s32:
case ARM::BI__builtin_arm_mve_vmaxvq_u32:
case ARM::BI__builtin_arm_mve_vminvq_s32:
case ARM::BI__builtin_arm_mve_vminvq_u32: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  bool  Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmaxvq_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_maxv;
    Param2 = true;
    break;
  case ARM::BI__builtin_arm_mve_vmaxvq_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_maxv;
    Param2 = false;
    break;
  case ARM::BI__builtin_arm_mve_vminvq_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_minv;
    Param2 = true;
    break;
  case ARM::BI__builtin_arm_mve_vminvq_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_minv;
    Param2 = false;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Param1, {llvm::FixedVectorType::get(Int32Ty, 4)}), {Val0, Val1, Val2});
  return Builder.CreateIntCast(Val3, Int32Ty, Param2);
}
case ARM::BI__builtin_arm_mve_vmovnbq_m_s16:
case ARM::BI__builtin_arm_mve_vmovnbq_m_s32:
case ARM::BI__builtin_arm_mve_vmovnbq_m_u16:
case ARM::BI__builtin_arm_mve_vmovnbq_m_u32:
case ARM::BI__builtin_arm_mve_vmovntq_m_s16:
case ARM::BI__builtin_arm_mve_vmovntq_m_s32:
case ARM::BI__builtin_arm_mve_vmovntq_m_u16:
case ARM::BI__builtin_arm_mve_vmovntq_m_u32: {
  uint32_t  Param0;
  llvm::Type * Param1;
  llvm::Type * Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmovnbq_m_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmovnbq_m_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmovnbq_m_u16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmovnbq_m_u32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmovntq_m_s16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmovntq_m_s32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmovntq_m_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmovntq_m_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param1}), {Val4});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vmovn_predicated, {Param2, Param3, Param1}), {Val0, Val1, Val2, Val5});
}
case ARM::BI__builtin_arm_mve_vmaxvq_p_s32:
case ARM::BI__builtin_arm_mve_vmaxvq_p_u32:
case ARM::BI__builtin_arm_mve_vminvq_p_s32:
case ARM::BI__builtin_arm_mve_vminvq_p_u32: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  bool  Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmaxvq_p_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_maxv_predicated;
    Param2 = true;
    break;
  case ARM::BI__builtin_arm_mve_vmaxvq_p_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_maxv_predicated;
    Param2 = false;
    break;
  case ARM::BI__builtin_arm_mve_vminvq_p_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_minv_predicated;
    Param2 = true;
    break;
  case ARM::BI__builtin_arm_mve_vminvq_p_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_minv_predicated;
    Param2 = false;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val4});
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Param1, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val0, Val1, Val2, Val5});
  return Builder.CreateIntCast(Val6, Int32Ty, Param2);
}
case ARM::BI__builtin_arm_mve_vcvtbq_m_f16_f32:
case ARM::BI__builtin_arm_mve_vcvtbq_m_f32_f16:
case ARM::BI__builtin_arm_mve_vcvttq_m_f16_f32:
case ARM::BI__builtin_arm_mve_vcvttq_m_f32_f16: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtbq_m_f16_f32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvt_narrow_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vcvtbq_m_f32_f16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvt_widen_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vcvttq_m_f16_f32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvt_narrow_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vcvttq_m_f32_f16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvt_widen_predicated;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val4});
  return Builder.CreateCall(CGM.getIntrinsic(Param1), {Val0, Val1, Val2, Val5});
}
case ARM::BI__builtin_arm_mve_vqmovnbq_m_s16:
case ARM::BI__builtin_arm_mve_vqmovnbq_m_s32:
case ARM::BI__builtin_arm_mve_vqmovnbq_m_u16:
case ARM::BI__builtin_arm_mve_vqmovnbq_m_u32:
case ARM::BI__builtin_arm_mve_vqmovntq_m_s16:
case ARM::BI__builtin_arm_mve_vqmovntq_m_s32:
case ARM::BI__builtin_arm_mve_vqmovntq_m_u16:
case ARM::BI__builtin_arm_mve_vqmovntq_m_u32:
case ARM::BI__builtin_arm_mve_vqmovunbq_m_s16:
case ARM::BI__builtin_arm_mve_vqmovunbq_m_s32:
case ARM::BI__builtin_arm_mve_vqmovuntq_m_s16:
case ARM::BI__builtin_arm_mve_vqmovuntq_m_s32: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  llvm::Type * Param3;
  llvm::Type * Param4;
  llvm::Type * Param5;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vqmovnbq_m_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqmovnbq_m_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqmovnbq_m_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqmovnbq_m_u32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqmovntq_m_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqmovntq_m_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqmovntq_m_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqmovntq_m_u32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqmovunbq_m_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqmovunbq_m_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqmovuntq_m_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqmovuntq_m_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param2);
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param3}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vqmovn_predicated, {Param4, Param5, Param3}), {Val0, Val1, Val2, Val3, Val4, Val7});
}
case ARM::BI__builtin_arm_mve_vqrshlq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqrshlq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqrshlq_m_n_s8:
case ARM::BI__builtin_arm_mve_vqrshlq_m_n_u16:
case ARM::BI__builtin_arm_mve_vqrshlq_m_n_u32:
case ARM::BI__builtin_arm_mve_vqrshlq_m_n_u8:
case ARM::BI__builtin_arm_mve_vqshlq_m_r_s16:
case ARM::BI__builtin_arm_mve_vqshlq_m_r_s32:
case ARM::BI__builtin_arm_mve_vqshlq_m_r_s8:
case ARM::BI__builtin_arm_mve_vqshlq_m_r_u16:
case ARM::BI__builtin_arm_mve_vqshlq_m_r_u32:
case ARM::BI__builtin_arm_mve_vqshlq_m_r_u8:
case ARM::BI__builtin_arm_mve_vrshlq_m_n_s16:
case ARM::BI__builtin_arm_mve_vrshlq_m_n_s32:
case ARM::BI__builtin_arm_mve_vrshlq_m_n_s8:
case ARM::BI__builtin_arm_mve_vrshlq_m_n_u16:
case ARM::BI__builtin_arm_mve_vrshlq_m_n_u32:
case ARM::BI__builtin_arm_mve_vrshlq_m_n_u8:
case ARM::BI__builtin_arm_mve_vshlq_m_r_s16:
case ARM::BI__builtin_arm_mve_vshlq_m_r_s32:
case ARM::BI__builtin_arm_mve_vshlq_m_r_s8:
case ARM::BI__builtin_arm_mve_vshlq_m_r_u16:
case ARM::BI__builtin_arm_mve_vshlq_m_r_u32:
case ARM::BI__builtin_arm_mve_vshlq_m_r_u8: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  llvm::Type * Param3;
  llvm::Type * Param4;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vqrshlq_m_n_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_m_n_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_m_n_s8:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_m_n_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_m_n_u32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_m_n_u8:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_r_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_r_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_r_s8:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_r_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_r_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_r_u8:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_m_n_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_m_n_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_m_n_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_m_n_u16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_m_n_u32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_m_n_u8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_r_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_r_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_r_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_r_u16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_r_u32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_r_u8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param2);
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param3}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vshl_scalar_predicated, {Param4, Param3}), {Val0, Val1, Val2, Val3, Val4, Val7});
}
case ARM::BI__builtin_arm_mve_vqrshlq_n_s16:
case ARM::BI__builtin_arm_mve_vqrshlq_n_s32:
case ARM::BI__builtin_arm_mve_vqrshlq_n_s8:
case ARM::BI__builtin_arm_mve_vqrshlq_n_u16:
case ARM::BI__builtin_arm_mve_vqrshlq_n_u32:
case ARM::BI__builtin_arm_mve_vqrshlq_n_u8:
case ARM::BI__builtin_arm_mve_vqshlq_r_s16:
case ARM::BI__builtin_arm_mve_vqshlq_r_s32:
case ARM::BI__builtin_arm_mve_vqshlq_r_s8:
case ARM::BI__builtin_arm_mve_vqshlq_r_u16:
case ARM::BI__builtin_arm_mve_vqshlq_r_u32:
case ARM::BI__builtin_arm_mve_vqshlq_r_u8:
case ARM::BI__builtin_arm_mve_vrshlq_n_s16:
case ARM::BI__builtin_arm_mve_vrshlq_n_s32:
case ARM::BI__builtin_arm_mve_vrshlq_n_s8:
case ARM::BI__builtin_arm_mve_vrshlq_n_u16:
case ARM::BI__builtin_arm_mve_vrshlq_n_u32:
case ARM::BI__builtin_arm_mve_vrshlq_n_u8:
case ARM::BI__builtin_arm_mve_vshlq_r_s16:
case ARM::BI__builtin_arm_mve_vshlq_r_s32:
case ARM::BI__builtin_arm_mve_vshlq_r_s8:
case ARM::BI__builtin_arm_mve_vshlq_r_u16:
case ARM::BI__builtin_arm_mve_vshlq_r_u32:
case ARM::BI__builtin_arm_mve_vshlq_r_u8: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vqrshlq_n_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_n_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_n_s8:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_n_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_n_u32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_n_u8:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_r_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_r_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_r_s8:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_r_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_r_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_r_u8:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_n_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_n_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_n_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_n_u16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_n_u32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_n_u8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_r_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_r_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_r_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_r_u16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_r_u32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_r_u8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param2);
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vshl_scalar, {Param3}), {Val0, Val1, Val2, Val3, Val4});
}
case ARM::BI__builtin_arm_mve_vqmovnbq_s16:
case ARM::BI__builtin_arm_mve_vqmovnbq_s32:
case ARM::BI__builtin_arm_mve_vqmovnbq_u16:
case ARM::BI__builtin_arm_mve_vqmovnbq_u32:
case ARM::BI__builtin_arm_mve_vqmovntq_s16:
case ARM::BI__builtin_arm_mve_vqmovntq_s32:
case ARM::BI__builtin_arm_mve_vqmovntq_u16:
case ARM::BI__builtin_arm_mve_vqmovntq_u32:
case ARM::BI__builtin_arm_mve_vqmovunbq_s16:
case ARM::BI__builtin_arm_mve_vqmovunbq_s32:
case ARM::BI__builtin_arm_mve_vqmovuntq_s16:
case ARM::BI__builtin_arm_mve_vqmovuntq_s32:
case ARM::BI__builtin_arm_mve_vqrshlq_s16:
case ARM::BI__builtin_arm_mve_vqrshlq_s32:
case ARM::BI__builtin_arm_mve_vqrshlq_s8:
case ARM::BI__builtin_arm_mve_vqrshlq_u16:
case ARM::BI__builtin_arm_mve_vqrshlq_u32:
case ARM::BI__builtin_arm_mve_vqrshlq_u8:
case ARM::BI__builtin_arm_mve_vqshlq_s16:
case ARM::BI__builtin_arm_mve_vqshlq_s32:
case ARM::BI__builtin_arm_mve_vqshlq_s8:
case ARM::BI__builtin_arm_mve_vqshlq_u16:
case ARM::BI__builtin_arm_mve_vqshlq_u32:
case ARM::BI__builtin_arm_mve_vqshlq_u8:
case ARM::BI__builtin_arm_mve_vrshlq_s16:
case ARM::BI__builtin_arm_mve_vrshlq_s32:
case ARM::BI__builtin_arm_mve_vrshlq_s8:
case ARM::BI__builtin_arm_mve_vrshlq_u16:
case ARM::BI__builtin_arm_mve_vrshlq_u32:
case ARM::BI__builtin_arm_mve_vrshlq_u8:
case ARM::BI__builtin_arm_mve_vshlq_s16:
case ARM::BI__builtin_arm_mve_vshlq_s32:
case ARM::BI__builtin_arm_mve_vshlq_s8:
case ARM::BI__builtin_arm_mve_vshlq_u16:
case ARM::BI__builtin_arm_mve_vshlq_u32:
case ARM::BI__builtin_arm_mve_vshlq_u8: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  Intrinsic::ID  Param3;
  llvm::Type * Param4;
  llvm::Type * Param5;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vqmovnbq_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vqmovn;
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqmovnbq_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vqmovn;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqmovnbq_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vqmovn;
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqmovnbq_u32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vqmovn;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqmovntq_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vqmovn;
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqmovntq_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vqmovn;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqmovntq_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vqmovn;
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqmovntq_u32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vqmovn;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqmovunbq_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vqmovn;
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqmovunbq_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vqmovn;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqmovuntq_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vqmovn;
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqmovuntq_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vqmovn;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_s8:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_u32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_u8:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_s8:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_u8:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_u16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_u32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_u8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_u16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_u32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_u8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vshl_vector;
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param2);
  return Builder.CreateCall(CGM.getIntrinsic(Param3, {Param4, Param5}), {Val0, Val1, Val2, Val3, Val4});
}
case ARM::BI__builtin_arm_mve_vmullbq_int_s16:
case ARM::BI__builtin_arm_mve_vmullbq_int_s32:
case ARM::BI__builtin_arm_mve_vmullbq_int_s8:
case ARM::BI__builtin_arm_mve_vmullbq_int_u16:
case ARM::BI__builtin_arm_mve_vmullbq_int_u32:
case ARM::BI__builtin_arm_mve_vmullbq_int_u8:
case ARM::BI__builtin_arm_mve_vmulltq_int_s16:
case ARM::BI__builtin_arm_mve_vmulltq_int_s32:
case ARM::BI__builtin_arm_mve_vmulltq_int_s8:
case ARM::BI__builtin_arm_mve_vmulltq_int_u16:
case ARM::BI__builtin_arm_mve_vmulltq_int_u32:
case ARM::BI__builtin_arm_mve_vmulltq_int_u8:
case ARM::BI__builtin_arm_mve_vshllbq_n_s16:
case ARM::BI__builtin_arm_mve_vshllbq_n_s8:
case ARM::BI__builtin_arm_mve_vshllbq_n_u16:
case ARM::BI__builtin_arm_mve_vshllbq_n_u8:
case ARM::BI__builtin_arm_mve_vshlltq_n_s16:
case ARM::BI__builtin_arm_mve_vshlltq_n_s8:
case ARM::BI__builtin_arm_mve_vshlltq_n_u16:
case ARM::BI__builtin_arm_mve_vshlltq_n_u8: {
  uint32_t  Param0;
  uint32_t  Param1;
  Intrinsic::ID  Param2;
  llvm::Type * Param3;
  llvm::Type * Param4;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmullbq_int_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_vmull;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmullbq_int_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_vmull;
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmullbq_int_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_vmull;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmullbq_int_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_vmull;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmullbq_int_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_vmull;
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmullbq_int_u8:
    Param0 = 1;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_vmull;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_vmull;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_vmull;
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_vmull;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_vmull;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_u32:
    Param0 = 1;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_vmull;
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_u8:
    Param0 = 1;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_vmull;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshllbq_n_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_vshll_imm;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshllbq_n_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_vshll_imm;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshllbq_n_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_vshll_imm;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshllbq_n_u8:
    Param0 = 1;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_vshll_imm;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshlltq_n_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_vshll_imm;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshlltq_n_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_vshll_imm;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshlltq_n_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_vshll_imm;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshlltq_n_u8:
    Param0 = 1;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_vshll_imm;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param1);
  return Builder.CreateCall(CGM.getIntrinsic(Param2, {Param3, Param4}), {Val0, Val1, Val2, Val3});
}
case ARM::BI__builtin_arm_mve_vcvtbq_f16_f32:
case ARM::BI__builtin_arm_mve_vcvttq_f16_f32: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtbq_f16_f32:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vcvttq_f16_f32:
    Param0 = 1;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcvt_narrow), {Val0, Val1, Val2});
}
case ARM::BI__builtin_arm_mve_vmullbq_poly_p16:
case ARM::BI__builtin_arm_mve_vmullbq_poly_p8:
case ARM::BI__builtin_arm_mve_vmulltq_poly_p16:
case ARM::BI__builtin_arm_mve_vmulltq_poly_p8:
case ARM::BI__builtin_arm_mve_vqdmullbq_s16:
case ARM::BI__builtin_arm_mve_vqdmullbq_s32:
case ARM::BI__builtin_arm_mve_vqdmulltq_s16:
case ARM::BI__builtin_arm_mve_vqdmulltq_s32: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmullbq_poly_p16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vmull_poly;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmullbq_poly_p8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vmull_poly;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_poly_p16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vmull_poly;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_poly_p8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vmull_poly;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqdmullbq_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vqdmull;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmullbq_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vqdmull;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulltq_s16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vqdmull;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulltq_s32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vqdmull;
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2, Param3}), {Val0, Val1, Val2});
}
case ARM::BI__builtin_arm_mve_vabdq_f16:
case ARM::BI__builtin_arm_mve_vabdq_f32:
case ARM::BI__builtin_arm_mve_vabdq_s16:
case ARM::BI__builtin_arm_mve_vabdq_s32:
case ARM::BI__builtin_arm_mve_vabdq_s8:
case ARM::BI__builtin_arm_mve_vabdq_u16:
case ARM::BI__builtin_arm_mve_vabdq_u32:
case ARM::BI__builtin_arm_mve_vabdq_u8:
case ARM::BI__builtin_arm_mve_vhaddq_s16:
case ARM::BI__builtin_arm_mve_vhaddq_s32:
case ARM::BI__builtin_arm_mve_vhaddq_s8:
case ARM::BI__builtin_arm_mve_vhaddq_u16:
case ARM::BI__builtin_arm_mve_vhaddq_u32:
case ARM::BI__builtin_arm_mve_vhaddq_u8:
case ARM::BI__builtin_arm_mve_vhsubq_s16:
case ARM::BI__builtin_arm_mve_vhsubq_s32:
case ARM::BI__builtin_arm_mve_vhsubq_s8:
case ARM::BI__builtin_arm_mve_vhsubq_u16:
case ARM::BI__builtin_arm_mve_vhsubq_u32:
case ARM::BI__builtin_arm_mve_vhsubq_u8:
case ARM::BI__builtin_arm_mve_vmulhq_s16:
case ARM::BI__builtin_arm_mve_vmulhq_s32:
case ARM::BI__builtin_arm_mve_vmulhq_s8:
case ARM::BI__builtin_arm_mve_vmulhq_u16:
case ARM::BI__builtin_arm_mve_vmulhq_u32:
case ARM::BI__builtin_arm_mve_vmulhq_u8:
case ARM::BI__builtin_arm_mve_vqshlq_n_s16:
case ARM::BI__builtin_arm_mve_vqshlq_n_s32:
case ARM::BI__builtin_arm_mve_vqshlq_n_s8:
case ARM::BI__builtin_arm_mve_vqshlq_n_u16:
case ARM::BI__builtin_arm_mve_vqshlq_n_u32:
case ARM::BI__builtin_arm_mve_vqshlq_n_u8:
case ARM::BI__builtin_arm_mve_vrhaddq_s16:
case ARM::BI__builtin_arm_mve_vrhaddq_s32:
case ARM::BI__builtin_arm_mve_vrhaddq_s8:
case ARM::BI__builtin_arm_mve_vrhaddq_u16:
case ARM::BI__builtin_arm_mve_vrhaddq_u32:
case ARM::BI__builtin_arm_mve_vrhaddq_u8:
case ARM::BI__builtin_arm_mve_vrmulhq_s16:
case ARM::BI__builtin_arm_mve_vrmulhq_s32:
case ARM::BI__builtin_arm_mve_vrmulhq_s8:
case ARM::BI__builtin_arm_mve_vrmulhq_u16:
case ARM::BI__builtin_arm_mve_vrmulhq_u32:
case ARM::BI__builtin_arm_mve_vrmulhq_u8:
case ARM::BI__builtin_arm_mve_vrshrq_n_s16:
case ARM::BI__builtin_arm_mve_vrshrq_n_s32:
case ARM::BI__builtin_arm_mve_vrshrq_n_s8:
case ARM::BI__builtin_arm_mve_vrshrq_n_u16:
case ARM::BI__builtin_arm_mve_vrshrq_n_u32:
case ARM::BI__builtin_arm_mve_vrshrq_n_u8: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vabdq_f16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vabd;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vabdq_f32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vabd;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vabdq_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vabd;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vabdq_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vabd;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vabdq_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vabd;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vabdq_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vabd;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vabdq_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vabd;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vabdq_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vabd;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vhadd;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vhadd;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vhadd;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vhadd;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vhadd;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vhadd;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vhsub;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vhsub;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vhsub;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vhsub;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vhsub;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vhsub;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vmulh;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vmulh;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vmulh;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vmulh;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vmulh;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vmulh;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_n_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vqshl_imm;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_n_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vqshl_imm;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_n_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vqshl_imm;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_n_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vqshl_imm;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_n_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vqshl_imm;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_n_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vqshl_imm;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vrhadd;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vrhadd;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vrhadd;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vrhadd;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vrhadd;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vrhadd;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vrmulh;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vrmulh;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vrmulh;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vrmulh;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vrmulh;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vrmulh;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_n_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vrshr_imm;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_n_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vrshr_imm;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_n_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vrshr_imm;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_n_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vrshr_imm;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_n_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vrshr_imm;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_n_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vrshr_imm;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2}), {Val0, Val1, Val2});
}
case ARM::BI__builtin_arm_mve_vaddq_s16:
case ARM::BI__builtin_arm_mve_vaddq_s32:
case ARM::BI__builtin_arm_mve_vaddq_s8:
case ARM::BI__builtin_arm_mve_vaddq_u16:
case ARM::BI__builtin_arm_mve_vaddq_u32:
case ARM::BI__builtin_arm_mve_vaddq_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateAdd(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vandq_s16:
case ARM::BI__builtin_arm_mve_vandq_s32:
case ARM::BI__builtin_arm_mve_vandq_s8:
case ARM::BI__builtin_arm_mve_vandq_u16:
case ARM::BI__builtin_arm_mve_vandq_u32:
case ARM::BI__builtin_arm_mve_vandq_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateAnd(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vldrdq_gather_base_s64:
case ARM::BI__builtin_arm_mve_vldrdq_gather_base_u64:
case ARM::BI__builtin_arm_mve_vldrwq_gather_base_f32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_base_s32:
case ARM::BI__builtin_arm_mve_vldrwq_gather_base_u32: {
  llvm::Type * Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vldrdq_gather_base_s64:
    Param0 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vldrdq_gather_base_u64:
    Param0 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param1 = llvm::FixedVectorType::get(Int64Ty, 2);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_base_f32:
    Param0 = llvm::FixedVectorType::get(FloatTy, 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_base_s32:
    Param0 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vldrwq_gather_base_u32:
    Param0 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vldr_gather_base, {Param0, Param1}), {Val0, Val1});
}
case ARM::BI__builtin_arm_mve_sqrshr:
case ARM::BI__builtin_arm_mve_sqshl:
case ARM::BI__builtin_arm_mve_srshr:
case ARM::BI__builtin_arm_mve_uqrshl:
case ARM::BI__builtin_arm_mve_uqshl:
case ARM::BI__builtin_arm_mve_urshr: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_sqrshr:
    Param0 = Intrinsic::arm_mve_sqrshr;
    break;
  case ARM::BI__builtin_arm_mve_sqshl:
    Param0 = Intrinsic::arm_mve_sqshl;
    break;
  case ARM::BI__builtin_arm_mve_srshr:
    Param0 = Intrinsic::arm_mve_srshr;
    break;
  case ARM::BI__builtin_arm_mve_uqrshl:
    Param0 = Intrinsic::arm_mve_uqrshl;
    break;
  case ARM::BI__builtin_arm_mve_uqshl:
    Param0 = Intrinsic::arm_mve_uqshl;
    break;
  case ARM::BI__builtin_arm_mve_urshr:
    Param0 = Intrinsic::arm_mve_urshr;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateCall(CGM.getIntrinsic(Param0), {Val0, Val1});
}
case ARM::BI__builtin_arm_mve_vbrsrq_n_f16:
case ARM::BI__builtin_arm_mve_vbrsrq_n_f32:
case ARM::BI__builtin_arm_mve_vbrsrq_n_s16:
case ARM::BI__builtin_arm_mve_vbrsrq_n_s32:
case ARM::BI__builtin_arm_mve_vbrsrq_n_s8:
case ARM::BI__builtin_arm_mve_vbrsrq_n_u16:
case ARM::BI__builtin_arm_mve_vbrsrq_n_u32:
case ARM::BI__builtin_arm_mve_vbrsrq_n_u8:
case ARM::BI__builtin_arm_mve_vmaxnmq_f16:
case ARM::BI__builtin_arm_mve_vmaxnmq_f32:
case ARM::BI__builtin_arm_mve_vminnmq_f16:
case ARM::BI__builtin_arm_mve_vminnmq_f32:
case ARM::BI__builtin_arm_mve_vqaddq_s16:
case ARM::BI__builtin_arm_mve_vqaddq_s32:
case ARM::BI__builtin_arm_mve_vqaddq_s8:
case ARM::BI__builtin_arm_mve_vqaddq_u16:
case ARM::BI__builtin_arm_mve_vqaddq_u32:
case ARM::BI__builtin_arm_mve_vqaddq_u8:
case ARM::BI__builtin_arm_mve_vqdmulhq_s16:
case ARM::BI__builtin_arm_mve_vqdmulhq_s32:
case ARM::BI__builtin_arm_mve_vqdmulhq_s8:
case ARM::BI__builtin_arm_mve_vqrdmulhq_s16:
case ARM::BI__builtin_arm_mve_vqrdmulhq_s32:
case ARM::BI__builtin_arm_mve_vqrdmulhq_s8:
case ARM::BI__builtin_arm_mve_vqshluq_n_s16:
case ARM::BI__builtin_arm_mve_vqshluq_n_s32:
case ARM::BI__builtin_arm_mve_vqshluq_n_s8:
case ARM::BI__builtin_arm_mve_vqsubq_s16:
case ARM::BI__builtin_arm_mve_vqsubq_s32:
case ARM::BI__builtin_arm_mve_vqsubq_s8:
case ARM::BI__builtin_arm_mve_vqsubq_u16:
case ARM::BI__builtin_arm_mve_vqsubq_u32:
case ARM::BI__builtin_arm_mve_vqsubq_u8: {
  Intrinsic::ID  Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vbrsrq_n_f16:
    Param0 = Intrinsic::arm_mve_vbrsr;
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vbrsrq_n_f32:
    Param0 = Intrinsic::arm_mve_vbrsr;
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vbrsrq_n_s16:
    Param0 = Intrinsic::arm_mve_vbrsr;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vbrsrq_n_s32:
    Param0 = Intrinsic::arm_mve_vbrsr;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vbrsrq_n_s8:
    Param0 = Intrinsic::arm_mve_vbrsr;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vbrsrq_n_u16:
    Param0 = Intrinsic::arm_mve_vbrsr;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vbrsrq_n_u32:
    Param0 = Intrinsic::arm_mve_vbrsr;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vbrsrq_n_u8:
    Param0 = Intrinsic::arm_mve_vbrsr;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmaxnmq_f16:
    Param0 = Intrinsic::maxnum;
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmaxnmq_f32:
    Param0 = Intrinsic::maxnum;
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vminnmq_f16:
    Param0 = Intrinsic::minnum;
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vminnmq_f32:
    Param0 = Intrinsic::minnum;
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_s16:
    Param0 = Intrinsic::sadd_sat;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_s32:
    Param0 = Intrinsic::sadd_sat;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_s8:
    Param0 = Intrinsic::sadd_sat;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_u16:
    Param0 = Intrinsic::uadd_sat;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_u32:
    Param0 = Intrinsic::uadd_sat;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_u8:
    Param0 = Intrinsic::uadd_sat;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulhq_s16:
    Param0 = Intrinsic::arm_mve_vqdmulh;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulhq_s32:
    Param0 = Intrinsic::arm_mve_vqdmulh;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulhq_s8:
    Param0 = Intrinsic::arm_mve_vqdmulh;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmulhq_s16:
    Param0 = Intrinsic::arm_mve_vqrdmulh;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmulhq_s32:
    Param0 = Intrinsic::arm_mve_vqrdmulh;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmulhq_s8:
    Param0 = Intrinsic::arm_mve_vqrdmulh;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqshluq_n_s16:
    Param0 = Intrinsic::arm_mve_vqshlu_imm;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshluq_n_s32:
    Param0 = Intrinsic::arm_mve_vqshlu_imm;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshluq_n_s8:
    Param0 = Intrinsic::arm_mve_vqshlu_imm;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_s16:
    Param0 = Intrinsic::ssub_sat;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_s32:
    Param0 = Intrinsic::ssub_sat;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_s8:
    Param0 = Intrinsic::ssub_sat;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_u16:
    Param0 = Intrinsic::usub_sat;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_u32:
    Param0 = Intrinsic::usub_sat;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_u8:
    Param0 = Intrinsic::usub_sat;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {Param1}), {Val0, Val1});
}
case ARM::BI__builtin_arm_mve_vgetq_lane_f16:
case ARM::BI__builtin_arm_mve_vgetq_lane_f32:
case ARM::BI__builtin_arm_mve_vgetq_lane_s16:
case ARM::BI__builtin_arm_mve_vgetq_lane_s32:
case ARM::BI__builtin_arm_mve_vgetq_lane_s64:
case ARM::BI__builtin_arm_mve_vgetq_lane_s8:
case ARM::BI__builtin_arm_mve_vgetq_lane_u16:
case ARM::BI__builtin_arm_mve_vgetq_lane_u32:
case ARM::BI__builtin_arm_mve_vgetq_lane_u64:
case ARM::BI__builtin_arm_mve_vgetq_lane_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateExtractElement(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vaddq_f16:
case ARM::BI__builtin_arm_mve_vaddq_f32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateFAdd(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vmulq_f16:
case ARM::BI__builtin_arm_mve_vmulq_f32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateFMul(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vsubq_f16:
case ARM::BI__builtin_arm_mve_vsubq_f32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateFSub(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vmulq_s16:
case ARM::BI__builtin_arm_mve_vmulq_s32:
case ARM::BI__builtin_arm_mve_vmulq_s8:
case ARM::BI__builtin_arm_mve_vmulq_u16:
case ARM::BI__builtin_arm_mve_vmulq_u32:
case ARM::BI__builtin_arm_mve_vmulq_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateMul(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vorrq_s16:
case ARM::BI__builtin_arm_mve_vorrq_s32:
case ARM::BI__builtin_arm_mve_vorrq_s8:
case ARM::BI__builtin_arm_mve_vorrq_u16:
case ARM::BI__builtin_arm_mve_vorrq_u32:
case ARM::BI__builtin_arm_mve_vorrq_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateOr(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vsubq_s16:
case ARM::BI__builtin_arm_mve_vsubq_s32:
case ARM::BI__builtin_arm_mve_vsubq_s8:
case ARM::BI__builtin_arm_mve_vsubq_u16:
case ARM::BI__builtin_arm_mve_vsubq_u32:
case ARM::BI__builtin_arm_mve_vsubq_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateSub(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_veorq_s16:
case ARM::BI__builtin_arm_mve_veorq_s32:
case ARM::BI__builtin_arm_mve_veorq_s8:
case ARM::BI__builtin_arm_mve_veorq_u16:
case ARM::BI__builtin_arm_mve_veorq_u32:
case ARM::BI__builtin_arm_mve_veorq_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateXor(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vmovlbq_s8:
case ARM::BI__builtin_arm_mve_vmovlbq_u8:
case ARM::BI__builtin_arm_mve_vmovltq_s8:
case ARM::BI__builtin_arm_mve_vmovltq_u8: {
  bool  Param0;
  bool  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmovlbq_s8:
    Param0 = 0;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vmovlbq_u8:
    Param0 = 0;
    Param1 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vmovltq_s8:
    Param0 = 1;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vmovltq_u8:
    Param0 = 1;
    Param1 = 1;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = VectorUnzip(Builder, Val0, static_cast<bool>(Param0));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int16Ty, 8);
  return SignOrZeroExtend(Builder, Val1, Val2, static_cast<bool>(Param1));
}
case ARM::BI__builtin_arm_mve_vmovlbq_s16:
case ARM::BI__builtin_arm_mve_vmovlbq_u16:
case ARM::BI__builtin_arm_mve_vmovltq_s16:
case ARM::BI__builtin_arm_mve_vmovltq_u16: {
  bool  Param0;
  bool  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmovlbq_s16:
    Param0 = 0;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vmovlbq_u16:
    Param0 = 0;
    Param1 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vmovltq_s16:
    Param0 = 1;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vmovltq_u16:
    Param0 = 1;
    Param1 = 1;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = VectorUnzip(Builder, Val0, static_cast<bool>(Param0));
  llvm::Type *Val2 = llvm::FixedVectorType::get(Int32Ty, 4);
  return SignOrZeroExtend(Builder, Val1, Val2, static_cast<bool>(Param1));
}
case ARM::BI__builtin_arm_mve_vclzq_s16:
case ARM::BI__builtin_arm_mve_vclzq_s32:
case ARM::BI__builtin_arm_mve_vclzq_s8:
case ARM::BI__builtin_arm_mve_vclzq_u16:
case ARM::BI__builtin_arm_mve_vclzq_u32:
case ARM::BI__builtin_arm_mve_vclzq_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vclzq_s16:
    Param0 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vclzq_s32:
    Param0 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vclzq_s8:
    Param0 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vclzq_u16:
    Param0 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vclzq_u32:
    Param0 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vclzq_u8:
    Param0 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = llvm::ConstantInt::get(Builder.getInt1Ty(), static_cast<bool>(0));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::ctlz, {Param0}), {Val0, Val1});
}
case ARM::BI__builtin_arm_mve_vpnot: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = llvm::ConstantInt::get(Int16Ty, 65535);
  return Builder.CreateXor(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vaddlvq_p_s32:
case ARM::BI__builtin_arm_mve_vaddlvq_p_u32:
case ARM::BI__builtin_arm_mve_vaddvq_p_s16:
case ARM::BI__builtin_arm_mve_vaddvq_p_s32:
case ARM::BI__builtin_arm_mve_vaddvq_p_s8:
case ARM::BI__builtin_arm_mve_vaddvq_p_u16:
case ARM::BI__builtin_arm_mve_vaddvq_p_u32:
case ARM::BI__builtin_arm_mve_vaddvq_p_u8: {
  uint32_t  Param0;
  llvm::Type * Param1;
  Intrinsic::ID  Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vaddlvq_p_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_addlv_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddlvq_p_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_addlv_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddvq_p_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_addv_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vaddvq_p_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_addv_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddvq_p_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_addv_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vaddvq_p_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_addv_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vaddvq_p_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_addv_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddvq_p_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_addv_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param1}), {Val3});
  return Builder.CreateCall(CGM.getIntrinsic(Param2, {Param3, Param1}), {Val0, Val1, Val4});
}
case ARM::BI__builtin_arm_mve_vcvtbq_f32_f16:
case ARM::BI__builtin_arm_mve_vcvttq_f32_f16: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtbq_f32_f16:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vcvttq_f32_f16:
    Param0 = 1;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = llvm::ConstantInt::get(Int32Ty, Param0);
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcvt_widen), {Val0, Val1});
}
case ARM::BI__builtin_arm_mve_vaddlvq_s32:
case ARM::BI__builtin_arm_mve_vaddlvq_u32:
case ARM::BI__builtin_arm_mve_vaddvq_s16:
case ARM::BI__builtin_arm_mve_vaddvq_s32:
case ARM::BI__builtin_arm_mve_vaddvq_s8:
case ARM::BI__builtin_arm_mve_vaddvq_u16:
case ARM::BI__builtin_arm_mve_vaddvq_u32:
case ARM::BI__builtin_arm_mve_vaddvq_u8: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vaddlvq_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_addlv;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddlvq_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_addlv;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddvq_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_addv;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vaddvq_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_addv;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddvq_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_addv;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vaddvq_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_addv;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vaddvq_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_addv;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddvq_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_addv;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = llvm::ConstantInt::get(Int32Ty, Param0);
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2}), {Val0, Val1});
}
case ARM::BI__builtin_arm_mve_asrl:
case ARM::BI__builtin_arm_mve_lsll:
case ARM::BI__builtin_arm_mve_sqshll:
case ARM::BI__builtin_arm_mve_srshrl:
case ARM::BI__builtin_arm_mve_uqshll:
case ARM::BI__builtin_arm_mve_urshrl: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_asrl:
    Param0 = Intrinsic::arm_mve_asrl;
    break;
  case ARM::BI__builtin_arm_mve_lsll:
    Param0 = Intrinsic::arm_mve_lsll;
    break;
  case ARM::BI__builtin_arm_mve_sqshll:
    Param0 = Intrinsic::arm_mve_sqshll;
    break;
  case ARM::BI__builtin_arm_mve_srshrl:
    Param0 = Intrinsic::arm_mve_srshrl;
    break;
  case ARM::BI__builtin_arm_mve_uqshll:
    Param0 = Intrinsic::arm_mve_uqshll;
    break;
  case ARM::BI__builtin_arm_mve_urshrl:
    Param0 = Intrinsic::arm_mve_urshrl;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val2 = Builder.CreateLShr(Val0, Val1);
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val5 = EmitScalarExpr(E->getArg(1));
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Param0), {Val4, Val3, Val5});
  Value *Val7 = Builder.CreateExtractValue(Val6, static_cast<unsigned>(1));
  Value *Val8 = Builder.CreateIntCast(Val7, Int64Ty, false);
  Value *Val9 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val10 = Builder.CreateShl(Val8, Val9);
  Value *Val11 = Builder.CreateExtractValue(Val6, static_cast<unsigned>(0));
  Value *Val12 = Builder.CreateIntCast(Val11, Int64Ty, false);
  return Builder.CreateOr(Val10, Val12);
}
case ARM::BI__builtin_arm_mve_sqrshrl:
case ARM::BI__builtin_arm_mve_sqrshrl_sat48: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_sqrshrl:
    Param0 = 64;
    break;
  case ARM::BI__builtin_arm_mve_sqrshrl_sat48:
    Param0 = 48;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val2 = Builder.CreateLShr(Val0, Val1);
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val5 = EmitScalarExpr(E->getArg(1));
  Value *Val6 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_sqrshrl), {Val4, Val3, Val5, Val6});
  Value *Val8 = Builder.CreateExtractValue(Val7, static_cast<unsigned>(1));
  Value *Val9 = Builder.CreateIntCast(Val8, Int64Ty, false);
  Value *Val10 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val11 = Builder.CreateShl(Val9, Val10);
  Value *Val12 = Builder.CreateExtractValue(Val7, static_cast<unsigned>(0));
  Value *Val13 = Builder.CreateIntCast(Val12, Int64Ty, false);
  return Builder.CreateOr(Val11, Val13);
}
case ARM::BI__builtin_arm_mve_uqrshll:
case ARM::BI__builtin_arm_mve_uqrshll_sat48: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_uqrshll:
    Param0 = 64;
    break;
  case ARM::BI__builtin_arm_mve_uqrshll_sat48:
    Param0 = 48;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val2 = Builder.CreateLShr(Val0, Val1);
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val5 = EmitScalarExpr(E->getArg(1));
  Value *Val6 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_uqrshll), {Val4, Val3, Val5, Val6});
  Value *Val8 = Builder.CreateExtractValue(Val7, static_cast<unsigned>(1));
  Value *Val9 = Builder.CreateIntCast(Val8, Int64Ty, false);
  Value *Val10 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val11 = Builder.CreateShl(Val9, Val10);
  Value *Val12 = Builder.CreateExtractValue(Val7, static_cast<unsigned>(0));
  Value *Val13 = Builder.CreateIntCast(Val12, Int64Ty, false);
  return Builder.CreateOr(Val11, Val13);
}
case ARM::BI__builtin_arm_mve_vmlaldavaq_s16:
case ARM::BI__builtin_arm_mve_vmlaldavaq_s32:
case ARM::BI__builtin_arm_mve_vmlaldavaq_u16:
case ARM::BI__builtin_arm_mve_vmlaldavaq_u32:
case ARM::BI__builtin_arm_mve_vmlaldavaxq_s16:
case ARM::BI__builtin_arm_mve_vmlaldavaxq_s32:
case ARM::BI__builtin_arm_mve_vmlsldavaq_s16:
case ARM::BI__builtin_arm_mve_vmlsldavaq_s32:
case ARM::BI__builtin_arm_mve_vmlsldavaxq_s16:
case ARM::BI__builtin_arm_mve_vmlsldavaxq_s32:
case ARM::BI__builtin_arm_mve_vrmlaldavhaq_s32:
case ARM::BI__builtin_arm_mve_vrmlaldavhaq_u32:
case ARM::BI__builtin_arm_mve_vrmlaldavhaxq_s32:
case ARM::BI__builtin_arm_mve_vrmlsldavhaq_s32:
case ARM::BI__builtin_arm_mve_vrmlsldavhaxq_s32: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  Intrinsic::ID  Param3;
  llvm::Type * Param4;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmlaldavaq_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavaq_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavaq_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavaq_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavaxq_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavaxq_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlsldavaq_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlsldavaq_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlsldavaxq_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlsldavaxq_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlaldavhaq_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vrmlldavha;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlaldavhaq_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vrmlldavha;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlaldavhaxq_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vrmlldavha;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlsldavhaq_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vrmlldavha;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlsldavhaxq_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vrmlldavha;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val2 = Builder.CreateLShr(Val0, Val1);
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val6 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val7 = llvm::ConstantInt::get(Int32Ty, Param2);
  Value *Val8 = EmitScalarExpr(E->getArg(1));
  Value *Val9 = EmitScalarExpr(E->getArg(2));
  Value *Val10 = Builder.CreateCall(CGM.getIntrinsic(Param3, {Param4}), {Val5, Val6, Val7, Val4, Val3, Val8, Val9});
  Value *Val11 = Builder.CreateExtractValue(Val10, static_cast<unsigned>(1));
  Value *Val12 = Builder.CreateIntCast(Val11, Int64Ty, false);
  Value *Val13 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val14 = Builder.CreateShl(Val12, Val13);
  Value *Val15 = Builder.CreateExtractValue(Val10, static_cast<unsigned>(0));
  Value *Val16 = Builder.CreateIntCast(Val15, Int64Ty, false);
  return Builder.CreateOr(Val14, Val16);
}
case ARM::BI__builtin_arm_mve_vmlaldavaq_p_s16:
case ARM::BI__builtin_arm_mve_vmlaldavaq_p_s32:
case ARM::BI__builtin_arm_mve_vmlaldavaq_p_u16:
case ARM::BI__builtin_arm_mve_vmlaldavaq_p_u32:
case ARM::BI__builtin_arm_mve_vmlaldavaxq_p_s16:
case ARM::BI__builtin_arm_mve_vmlaldavaxq_p_s32:
case ARM::BI__builtin_arm_mve_vmlsldavaq_p_s16:
case ARM::BI__builtin_arm_mve_vmlsldavaq_p_s32:
case ARM::BI__builtin_arm_mve_vmlsldavaxq_p_s16:
case ARM::BI__builtin_arm_mve_vmlsldavaxq_p_s32:
case ARM::BI__builtin_arm_mve_vrmlaldavhaq_p_s32:
case ARM::BI__builtin_arm_mve_vrmlaldavhaq_p_u32:
case ARM::BI__builtin_arm_mve_vrmlaldavhaxq_p_s32:
case ARM::BI__builtin_arm_mve_vrmlsldavhaq_p_s32:
case ARM::BI__builtin_arm_mve_vrmlsldavhaxq_p_s32: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  llvm::Type * Param3;
  Intrinsic::ID  Param4;
  llvm::Type * Param5;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmlaldavaq_p_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavaq_p_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavaq_p_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavaq_p_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavaxq_p_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavaxq_p_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlsldavaq_p_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlsldavaq_p_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlsldavaxq_p_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlsldavaxq_p_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlaldavhaq_p_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vrmlldavha_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlaldavhaq_p_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vrmlldavha_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlaldavhaxq_p_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vrmlldavha_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlsldavhaq_p_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vrmlldavha_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlsldavhaxq_p_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vrmlldavha_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  Value *Val1 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val2 = Builder.CreateLShr(Val0, Val1);
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val6 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val7 = llvm::ConstantInt::get(Int32Ty, Param2);
  Value *Val8 = EmitScalarExpr(E->getArg(1));
  Value *Val9 = EmitScalarExpr(E->getArg(2));
  Value *Val10 = EmitScalarExpr(E->getArg(3));
  Value *Val11 = Builder.CreateIntCast(Val10, Int32Ty, false);
  Value *Val12 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param3}), {Val11});
  Value *Val13 = Builder.CreateCall(CGM.getIntrinsic(Param4, {Param5, Param3}), {Val5, Val6, Val7, Val4, Val3, Val8, Val9, Val12});
  Value *Val14 = Builder.CreateExtractValue(Val13, static_cast<unsigned>(1));
  Value *Val15 = Builder.CreateIntCast(Val14, Int64Ty, false);
  Value *Val16 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val17 = Builder.CreateShl(Val15, Val16);
  Value *Val18 = Builder.CreateExtractValue(Val13, static_cast<unsigned>(0));
  Value *Val19 = Builder.CreateIntCast(Val18, Int64Ty, false);
  return Builder.CreateOr(Val17, Val19);
}
case ARM::BI__builtin_arm_mve_vreinterpretq_f32_f16:
case ARM::BI__builtin_arm_mve_vreinterpretq_f32_s16:
case ARM::BI__builtin_arm_mve_vreinterpretq_f32_s32:
case ARM::BI__builtin_arm_mve_vreinterpretq_f32_s64:
case ARM::BI__builtin_arm_mve_vreinterpretq_f32_s8:
case ARM::BI__builtin_arm_mve_vreinterpretq_f32_u16:
case ARM::BI__builtin_arm_mve_vreinterpretq_f32_u32:
case ARM::BI__builtin_arm_mve_vreinterpretq_f32_u64:
case ARM::BI__builtin_arm_mve_vreinterpretq_f32_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(FloatTy, 4);
  return ARMMVEVectorReinterpret(Builder, this, Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vcvtq_f32_s32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(FloatTy, 4);
  return Builder.CreateSIToFP(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vcvtq_f32_u32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(FloatTy, 4);
  return Builder.CreateUIToFP(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vreinterpretq_f16_f32:
case ARM::BI__builtin_arm_mve_vreinterpretq_f16_s16:
case ARM::BI__builtin_arm_mve_vreinterpretq_f16_s32:
case ARM::BI__builtin_arm_mve_vreinterpretq_f16_s64:
case ARM::BI__builtin_arm_mve_vreinterpretq_f16_s8:
case ARM::BI__builtin_arm_mve_vreinterpretq_f16_u16:
case ARM::BI__builtin_arm_mve_vreinterpretq_f16_u32:
case ARM::BI__builtin_arm_mve_vreinterpretq_f16_u64:
case ARM::BI__builtin_arm_mve_vreinterpretq_f16_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(HalfTy, 8);
  return ARMMVEVectorReinterpret(Builder, this, Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vcvtq_f16_s16: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(HalfTy, 8);
  return Builder.CreateSIToFP(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vcvtq_f16_u16: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(HalfTy, 8);
  return Builder.CreateUIToFP(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vqnegq_s16: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val2 = ARMMVEConstantSplat<1,0>(Builder, Val1);
  Value *Val3 = Builder.CreateICmpEQ(Val0, Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val5 = ARMMVEConstantSplat<0,1>(Builder, Val4);
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val7 = llvm::Constant::getNullValue(Val6);
  Value *Val8 = Builder.CreateSub(Val7, Val0);
  return Builder.CreateSelect(Val3, Val5, Val8);
}
case ARM::BI__builtin_arm_mve_vmvnq_s16:
case ARM::BI__builtin_arm_mve_vmvnq_u16: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val2 = ARMMVEConstantSplat<1,1>(Builder, Val1);
  return Builder.CreateXor(Val0, Val2);
}
case ARM::BI__builtin_arm_mve_vmovntq_s16:
case ARM::BI__builtin_arm_mve_vmovntq_u16: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val2 = ARMMVEVectorReinterpret(Builder, this, Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = VectorZip(Builder, Val2, Val3);
  llvm::Type *Val5 = llvm::FixedVectorType::get(Int8Ty, 16);
  return Builder.CreateTrunc(Val4, Val5);
}
case ARM::BI__builtin_arm_mve_vandq_f16: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val2 = Builder.CreateBitCast(Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val5 = Builder.CreateBitCast(Val3, Val4);
  Value *Val6 = Builder.CreateAnd(Val2, Val5);
  llvm::Type *Val7 = llvm::FixedVectorType::get(HalfTy, 8);
  return Builder.CreateBitCast(Val6, Val7);
}
case ARM::BI__builtin_arm_mve_vbicq_f16: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val2 = Builder.CreateBitCast(Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val5 = Builder.CreateBitCast(Val3, Val4);
  Value *Val6 = Builder.CreateNot(Val5);
  Value *Val7 = Builder.CreateAnd(Val2, Val6);
  llvm::Type *Val8 = llvm::FixedVectorType::get(HalfTy, 8);
  return Builder.CreateBitCast(Val7, Val8);
}
case ARM::BI__builtin_arm_mve_vornq_f16: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val2 = Builder.CreateBitCast(Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val5 = Builder.CreateBitCast(Val3, Val4);
  Value *Val6 = Builder.CreateNot(Val5);
  Value *Val7 = Builder.CreateOr(Val2, Val6);
  llvm::Type *Val8 = llvm::FixedVectorType::get(HalfTy, 8);
  return Builder.CreateBitCast(Val7, Val8);
}
case ARM::BI__builtin_arm_mve_vorrq_f16: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val2 = Builder.CreateBitCast(Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val5 = Builder.CreateBitCast(Val3, Val4);
  Value *Val6 = Builder.CreateOr(Val2, Val5);
  llvm::Type *Val7 = llvm::FixedVectorType::get(HalfTy, 8);
  return Builder.CreateBitCast(Val6, Val7);
}
case ARM::BI__builtin_arm_mve_veorq_f16: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val2 = Builder.CreateBitCast(Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val5 = Builder.CreateBitCast(Val3, Val4);
  Value *Val6 = Builder.CreateXor(Val2, Val5);
  llvm::Type *Val7 = llvm::FixedVectorType::get(HalfTy, 8);
  return Builder.CreateBitCast(Val6, Val7);
}
case ARM::BI__builtin_arm_mve_vqabsq_s16: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val2 = llvm::Constant::getNullValue(Val1);
  Value *Val3 = Builder.CreateICmpSGT(Val0, Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val5 = ARMMVEConstantSplat<1,0>(Builder, Val4);
  Value *Val6 = Builder.CreateICmpEQ(Val0, Val5);
  llvm::Type *Val7 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val8 = ARMMVEConstantSplat<0,1>(Builder, Val7);
  llvm::Type *Val9 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val10 = llvm::Constant::getNullValue(Val9);
  Value *Val11 = Builder.CreateSub(Val10, Val0);
  Value *Val12 = Builder.CreateSelect(Val6, Val8, Val11);
  return Builder.CreateSelect(Val3, Val0, Val12);
}
case ARM::BI__builtin_arm_mve_vabsq_s16: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val2 = llvm::Constant::getNullValue(Val1);
  Value *Val3 = Builder.CreateICmpSLT(Val0, Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val5 = llvm::Constant::getNullValue(Val4);
  Value *Val6 = Builder.CreateSub(Val5, Val0);
  return Builder.CreateSelect(Val3, Val6, Val0);
}
case ARM::BI__builtin_arm_mve_vreinterpretq_s16_f16:
case ARM::BI__builtin_arm_mve_vreinterpretq_s16_f32:
case ARM::BI__builtin_arm_mve_vreinterpretq_s16_s32:
case ARM::BI__builtin_arm_mve_vreinterpretq_s16_s64:
case ARM::BI__builtin_arm_mve_vreinterpretq_s16_s8:
case ARM::BI__builtin_arm_mve_vreinterpretq_s16_u16:
case ARM::BI__builtin_arm_mve_vreinterpretq_s16_u32:
case ARM::BI__builtin_arm_mve_vreinterpretq_s16_u64:
case ARM::BI__builtin_arm_mve_vreinterpretq_s16_u8:
case ARM::BI__builtin_arm_mve_vreinterpretq_u16_f16:
case ARM::BI__builtin_arm_mve_vreinterpretq_u16_f32:
case ARM::BI__builtin_arm_mve_vreinterpretq_u16_s16:
case ARM::BI__builtin_arm_mve_vreinterpretq_u16_s32:
case ARM::BI__builtin_arm_mve_vreinterpretq_u16_s64:
case ARM::BI__builtin_arm_mve_vreinterpretq_u16_s8:
case ARM::BI__builtin_arm_mve_vreinterpretq_u16_u32:
case ARM::BI__builtin_arm_mve_vreinterpretq_u16_u64:
case ARM::BI__builtin_arm_mve_vreinterpretq_u16_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 8);
  return ARMMVEVectorReinterpret(Builder, this, Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vcvtq_s16_f16: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 8);
  return Builder.CreateFPToSI(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vcvtq_u16_f16: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 8);
  return Builder.CreateFPToUI(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vqnegq_s32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val2 = ARMMVEConstantSplat<1,0>(Builder, Val1);
  Value *Val3 = Builder.CreateICmpEQ(Val0, Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val5 = ARMMVEConstantSplat<0,1>(Builder, Val4);
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val7 = llvm::Constant::getNullValue(Val6);
  Value *Val8 = Builder.CreateSub(Val7, Val0);
  return Builder.CreateSelect(Val3, Val5, Val8);
}
case ARM::BI__builtin_arm_mve_vmvnq_s32:
case ARM::BI__builtin_arm_mve_vmvnq_u32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val2 = ARMMVEConstantSplat<1,1>(Builder, Val1);
  return Builder.CreateXor(Val0, Val2);
}
case ARM::BI__builtin_arm_mve_vmovntq_s32:
case ARM::BI__builtin_arm_mve_vmovntq_u32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val2 = ARMMVEVectorReinterpret(Builder, this, Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = VectorZip(Builder, Val2, Val3);
  llvm::Type *Val5 = llvm::FixedVectorType::get(Int16Ty, 8);
  return Builder.CreateTrunc(Val4, Val5);
}
case ARM::BI__builtin_arm_mve_vandq_f32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val2 = Builder.CreateBitCast(Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val5 = Builder.CreateBitCast(Val3, Val4);
  Value *Val6 = Builder.CreateAnd(Val2, Val5);
  llvm::Type *Val7 = llvm::FixedVectorType::get(FloatTy, 4);
  return Builder.CreateBitCast(Val6, Val7);
}
case ARM::BI__builtin_arm_mve_vbicq_f32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val2 = Builder.CreateBitCast(Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val5 = Builder.CreateBitCast(Val3, Val4);
  Value *Val6 = Builder.CreateNot(Val5);
  Value *Val7 = Builder.CreateAnd(Val2, Val6);
  llvm::Type *Val8 = llvm::FixedVectorType::get(FloatTy, 4);
  return Builder.CreateBitCast(Val7, Val8);
}
case ARM::BI__builtin_arm_mve_vornq_f32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val2 = Builder.CreateBitCast(Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val5 = Builder.CreateBitCast(Val3, Val4);
  Value *Val6 = Builder.CreateNot(Val5);
  Value *Val7 = Builder.CreateOr(Val2, Val6);
  llvm::Type *Val8 = llvm::FixedVectorType::get(FloatTy, 4);
  return Builder.CreateBitCast(Val7, Val8);
}
case ARM::BI__builtin_arm_mve_vorrq_f32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val2 = Builder.CreateBitCast(Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val5 = Builder.CreateBitCast(Val3, Val4);
  Value *Val6 = Builder.CreateOr(Val2, Val5);
  llvm::Type *Val7 = llvm::FixedVectorType::get(FloatTy, 4);
  return Builder.CreateBitCast(Val6, Val7);
}
case ARM::BI__builtin_arm_mve_veorq_f32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val2 = Builder.CreateBitCast(Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val5 = Builder.CreateBitCast(Val3, Val4);
  Value *Val6 = Builder.CreateXor(Val2, Val5);
  llvm::Type *Val7 = llvm::FixedVectorType::get(FloatTy, 4);
  return Builder.CreateBitCast(Val6, Val7);
}
case ARM::BI__builtin_arm_mve_vqabsq_s32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val2 = llvm::Constant::getNullValue(Val1);
  Value *Val3 = Builder.CreateICmpSGT(Val0, Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val5 = ARMMVEConstantSplat<1,0>(Builder, Val4);
  Value *Val6 = Builder.CreateICmpEQ(Val0, Val5);
  llvm::Type *Val7 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val8 = ARMMVEConstantSplat<0,1>(Builder, Val7);
  llvm::Type *Val9 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val10 = llvm::Constant::getNullValue(Val9);
  Value *Val11 = Builder.CreateSub(Val10, Val0);
  Value *Val12 = Builder.CreateSelect(Val6, Val8, Val11);
  return Builder.CreateSelect(Val3, Val0, Val12);
}
case ARM::BI__builtin_arm_mve_vabsq_s32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val2 = llvm::Constant::getNullValue(Val1);
  Value *Val3 = Builder.CreateICmpSLT(Val0, Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val5 = llvm::Constant::getNullValue(Val4);
  Value *Val6 = Builder.CreateSub(Val5, Val0);
  return Builder.CreateSelect(Val3, Val6, Val0);
}
case ARM::BI__builtin_arm_mve_vreinterpretq_s32_f16:
case ARM::BI__builtin_arm_mve_vreinterpretq_s32_f32:
case ARM::BI__builtin_arm_mve_vreinterpretq_s32_s16:
case ARM::BI__builtin_arm_mve_vreinterpretq_s32_s64:
case ARM::BI__builtin_arm_mve_vreinterpretq_s32_s8:
case ARM::BI__builtin_arm_mve_vreinterpretq_s32_u16:
case ARM::BI__builtin_arm_mve_vreinterpretq_s32_u32:
case ARM::BI__builtin_arm_mve_vreinterpretq_s32_u64:
case ARM::BI__builtin_arm_mve_vreinterpretq_s32_u8:
case ARM::BI__builtin_arm_mve_vreinterpretq_u32_f16:
case ARM::BI__builtin_arm_mve_vreinterpretq_u32_f32:
case ARM::BI__builtin_arm_mve_vreinterpretq_u32_s16:
case ARM::BI__builtin_arm_mve_vreinterpretq_u32_s32:
case ARM::BI__builtin_arm_mve_vreinterpretq_u32_s64:
case ARM::BI__builtin_arm_mve_vreinterpretq_u32_s8:
case ARM::BI__builtin_arm_mve_vreinterpretq_u32_u16:
case ARM::BI__builtin_arm_mve_vreinterpretq_u32_u64:
case ARM::BI__builtin_arm_mve_vreinterpretq_u32_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int32Ty, 4);
  return ARMMVEVectorReinterpret(Builder, this, Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vcvtq_s32_f32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int32Ty, 4);
  return Builder.CreateFPToSI(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vcvtq_u32_f32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int32Ty, 4);
  return Builder.CreateFPToUI(Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vreinterpretq_s64_f16:
case ARM::BI__builtin_arm_mve_vreinterpretq_s64_f32:
case ARM::BI__builtin_arm_mve_vreinterpretq_s64_s16:
case ARM::BI__builtin_arm_mve_vreinterpretq_s64_s32:
case ARM::BI__builtin_arm_mve_vreinterpretq_s64_s8:
case ARM::BI__builtin_arm_mve_vreinterpretq_s64_u16:
case ARM::BI__builtin_arm_mve_vreinterpretq_s64_u32:
case ARM::BI__builtin_arm_mve_vreinterpretq_s64_u64:
case ARM::BI__builtin_arm_mve_vreinterpretq_s64_u8:
case ARM::BI__builtin_arm_mve_vreinterpretq_u64_f16:
case ARM::BI__builtin_arm_mve_vreinterpretq_u64_f32:
case ARM::BI__builtin_arm_mve_vreinterpretq_u64_s16:
case ARM::BI__builtin_arm_mve_vreinterpretq_u64_s32:
case ARM::BI__builtin_arm_mve_vreinterpretq_u64_s64:
case ARM::BI__builtin_arm_mve_vreinterpretq_u64_s8:
case ARM::BI__builtin_arm_mve_vreinterpretq_u64_u16:
case ARM::BI__builtin_arm_mve_vreinterpretq_u64_u32:
case ARM::BI__builtin_arm_mve_vreinterpretq_u64_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int64Ty, 2);
  return ARMMVEVectorReinterpret(Builder, this, Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vqnegq_s8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val2 = ARMMVEConstantSplat<1,0>(Builder, Val1);
  Value *Val3 = Builder.CreateICmpEQ(Val0, Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val5 = ARMMVEConstantSplat<0,1>(Builder, Val4);
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val7 = llvm::Constant::getNullValue(Val6);
  Value *Val8 = Builder.CreateSub(Val7, Val0);
  return Builder.CreateSelect(Val3, Val5, Val8);
}
case ARM::BI__builtin_arm_mve_vmvnq_s8:
case ARM::BI__builtin_arm_mve_vmvnq_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val2 = ARMMVEConstantSplat<1,1>(Builder, Val1);
  return Builder.CreateXor(Val0, Val2);
}
case ARM::BI__builtin_arm_mve_vqabsq_s8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val2 = llvm::Constant::getNullValue(Val1);
  Value *Val3 = Builder.CreateICmpSGT(Val0, Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val5 = ARMMVEConstantSplat<1,0>(Builder, Val4);
  Value *Val6 = Builder.CreateICmpEQ(Val0, Val5);
  llvm::Type *Val7 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val8 = ARMMVEConstantSplat<0,1>(Builder, Val7);
  llvm::Type *Val9 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val10 = llvm::Constant::getNullValue(Val9);
  Value *Val11 = Builder.CreateSub(Val10, Val0);
  Value *Val12 = Builder.CreateSelect(Val6, Val8, Val11);
  return Builder.CreateSelect(Val3, Val0, Val12);
}
case ARM::BI__builtin_arm_mve_vabsq_s8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val2 = llvm::Constant::getNullValue(Val1);
  Value *Val3 = Builder.CreateICmpSLT(Val0, Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val5 = llvm::Constant::getNullValue(Val4);
  Value *Val6 = Builder.CreateSub(Val5, Val0);
  return Builder.CreateSelect(Val3, Val6, Val0);
}
case ARM::BI__builtin_arm_mve_vreinterpretq_s8_f16:
case ARM::BI__builtin_arm_mve_vreinterpretq_s8_f32:
case ARM::BI__builtin_arm_mve_vreinterpretq_s8_s16:
case ARM::BI__builtin_arm_mve_vreinterpretq_s8_s32:
case ARM::BI__builtin_arm_mve_vreinterpretq_s8_s64:
case ARM::BI__builtin_arm_mve_vreinterpretq_s8_u16:
case ARM::BI__builtin_arm_mve_vreinterpretq_s8_u32:
case ARM::BI__builtin_arm_mve_vreinterpretq_s8_u64:
case ARM::BI__builtin_arm_mve_vreinterpretq_s8_u8:
case ARM::BI__builtin_arm_mve_vreinterpretq_u8_f16:
case ARM::BI__builtin_arm_mve_vreinterpretq_u8_f32:
case ARM::BI__builtin_arm_mve_vreinterpretq_u8_s16:
case ARM::BI__builtin_arm_mve_vreinterpretq_u8_s32:
case ARM::BI__builtin_arm_mve_vreinterpretq_u8_s64:
case ARM::BI__builtin_arm_mve_vreinterpretq_u8_s8:
case ARM::BI__builtin_arm_mve_vreinterpretq_u8_u16:
case ARM::BI__builtin_arm_mve_vreinterpretq_u8_u32:
case ARM::BI__builtin_arm_mve_vreinterpretq_u8_u64: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int8Ty, 16);
  return ARMMVEVectorReinterpret(Builder, this, Val0, Val1);
}
case ARM::BI__builtin_arm_mve_vrev16q_s8:
case ARM::BI__builtin_arm_mve_vrev16q_u8:
case ARM::BI__builtin_arm_mve_vrev32q_f16:
case ARM::BI__builtin_arm_mve_vrev32q_s16:
case ARM::BI__builtin_arm_mve_vrev32q_s8:
case ARM::BI__builtin_arm_mve_vrev32q_u16:
case ARM::BI__builtin_arm_mve_vrev32q_u8:
case ARM::BI__builtin_arm_mve_vrev64q_f16:
case ARM::BI__builtin_arm_mve_vrev64q_f32:
case ARM::BI__builtin_arm_mve_vrev64q_s16:
case ARM::BI__builtin_arm_mve_vrev64q_s32:
case ARM::BI__builtin_arm_mve_vrev64q_s8:
case ARM::BI__builtin_arm_mve_vrev64q_u16:
case ARM::BI__builtin_arm_mve_vrev64q_u32:
case ARM::BI__builtin_arm_mve_vrev64q_u8: {
  unsigned  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vrev16q_s8:
    Param0 = 16;
    break;
  case ARM::BI__builtin_arm_mve_vrev16q_u8:
    Param0 = 16;
    break;
  case ARM::BI__builtin_arm_mve_vrev32q_f16:
    Param0 = 32;
    break;
  case ARM::BI__builtin_arm_mve_vrev32q_s16:
    Param0 = 32;
    break;
  case ARM::BI__builtin_arm_mve_vrev32q_s8:
    Param0 = 32;
    break;
  case ARM::BI__builtin_arm_mve_vrev32q_u16:
    Param0 = 32;
    break;
  case ARM::BI__builtin_arm_mve_vrev32q_u8:
    Param0 = 32;
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_f16:
    Param0 = 64;
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_f32:
    Param0 = 64;
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_s16:
    Param0 = 64;
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_s32:
    Param0 = 64;
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_s8:
    Param0 = 64;
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_u16:
    Param0 = 64;
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_u32:
    Param0 = 64;
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_u8:
    Param0 = 64;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  return ARMMVEVectorElementReverse(Builder, Val0, static_cast<unsigned>(Param0));
}
case ARM::BI__builtin_arm_mve_vdupq_n_f16:
case ARM::BI__builtin_arm_mve_vdupq_n_f32:
case ARM::BI__builtin_arm_mve_vdupq_n_s16:
case ARM::BI__builtin_arm_mve_vdupq_n_s32:
case ARM::BI__builtin_arm_mve_vdupq_n_s8:
case ARM::BI__builtin_arm_mve_vdupq_n_u16:
case ARM::BI__builtin_arm_mve_vdupq_n_u32:
case ARM::BI__builtin_arm_mve_vdupq_n_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  return ARMMVEVectorSplat(Builder, Val0);
}
case ARM::BI__builtin_arm_mve_vabsq_f16:
case ARM::BI__builtin_arm_mve_vabsq_f32:
case ARM::BI__builtin_arm_mve_vclsq_s16:
case ARM::BI__builtin_arm_mve_vclsq_s32:
case ARM::BI__builtin_arm_mve_vclsq_s8:
case ARM::BI__builtin_arm_mve_vrndaq_f16:
case ARM::BI__builtin_arm_mve_vrndaq_f32:
case ARM::BI__builtin_arm_mve_vrndmq_f16:
case ARM::BI__builtin_arm_mve_vrndmq_f32:
case ARM::BI__builtin_arm_mve_vrndnq_f16:
case ARM::BI__builtin_arm_mve_vrndnq_f32:
case ARM::BI__builtin_arm_mve_vrndpq_f16:
case ARM::BI__builtin_arm_mve_vrndpq_f32:
case ARM::BI__builtin_arm_mve_vrndq_f16:
case ARM::BI__builtin_arm_mve_vrndq_f32:
case ARM::BI__builtin_arm_mve_vrndxq_f16:
case ARM::BI__builtin_arm_mve_vrndxq_f32: {
  Intrinsic::ID  Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vabsq_f16:
    Param0 = Intrinsic::fabs;
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vabsq_f32:
    Param0 = Intrinsic::fabs;
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vclsq_s16:
    Param0 = Intrinsic::arm_mve_vcls;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vclsq_s32:
    Param0 = Intrinsic::arm_mve_vcls;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vclsq_s8:
    Param0 = Intrinsic::arm_mve_vcls;
    Param1 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrndaq_f16:
    Param0 = Intrinsic::round;
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrndaq_f32:
    Param0 = Intrinsic::round;
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrndmq_f16:
    Param0 = Intrinsic::floor;
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrndmq_f32:
    Param0 = Intrinsic::floor;
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrndnq_f16:
    Param0 = Intrinsic::arm_mve_vrintn;
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrndnq_f32:
    Param0 = Intrinsic::arm_mve_vrintn;
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrndpq_f16:
    Param0 = Intrinsic::ceil;
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrndpq_f32:
    Param0 = Intrinsic::ceil;
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrndq_f16:
    Param0 = Intrinsic::trunc;
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrndq_f32:
    Param0 = Intrinsic::trunc;
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrndxq_f16:
    Param0 = Intrinsic::rint;
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrndxq_f32:
    Param0 = Intrinsic::rint;
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {Param1}), {Val0});
}
case ARM::BI__builtin_arm_mve_vnegq_f16:
case ARM::BI__builtin_arm_mve_vnegq_f32: {
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateFNeg(Val0);
}
case ARM::BI__builtin_arm_mve_vshrq_n_s16:
case ARM::BI__builtin_arm_mve_vshrq_n_s32:
case ARM::BI__builtin_arm_mve_vshrq_n_s8:
case ARM::BI__builtin_arm_mve_vshrq_n_u16:
case ARM::BI__builtin_arm_mve_vshrq_n_u32:
case ARM::BI__builtin_arm_mve_vshrq_n_u8: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vshrq_n_s16:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vshrq_n_s32:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vshrq_n_s8:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vshrq_n_u16:
    Param0 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vshrq_n_u32:
    Param0 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vshrq_n_u8:
    Param0 = 1;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(0));
  return MVEImmediateShr(Builder, Val0, GetIntegerConstantValue<unsigned>(E->getArg(1), getContext()), static_cast<bool>(Param0));
}
case ARM::BI__builtin_arm_mve_vst1q_f32:
case ARM::BI__builtin_arm_mve_vstrwq_f32: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Address Val1 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val2 = Builder.CreatePointerCast((Val1.emitRawPointer(*this)), Builder.getPtrTy());
  Address Val3 = Address(Val2, llvm::FixedVectorType::get(FloatTy, 4), CharUnits::fromQuantity(4));
  return Builder.CreateStore(Val0, Val3);
}
case ARM::BI__builtin_arm_mve_vst1q_f16:
case ARM::BI__builtin_arm_mve_vstrhq_f16: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Address Val1 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val2 = Builder.CreatePointerCast((Val1.emitRawPointer(*this)), Builder.getPtrTy());
  Address Val3 = Address(Val2, llvm::FixedVectorType::get(HalfTy, 8), CharUnits::fromQuantity(2));
  return Builder.CreateStore(Val0, Val3);
}
case ARM::BI__builtin_arm_mve_vst1q_s16:
case ARM::BI__builtin_arm_mve_vst1q_u16:
case ARM::BI__builtin_arm_mve_vstrhq_s16:
case ARM::BI__builtin_arm_mve_vstrhq_u16: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Address Val1 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val2 = Builder.CreatePointerCast((Val1.emitRawPointer(*this)), Builder.getPtrTy());
  Address Val3 = Address(Val2, llvm::FixedVectorType::get(Int16Ty, 8), CharUnits::fromQuantity(2));
  return Builder.CreateStore(Val0, Val3);
}
case ARM::BI__builtin_arm_mve_vst1q_s32:
case ARM::BI__builtin_arm_mve_vst1q_u32:
case ARM::BI__builtin_arm_mve_vstrwq_s32:
case ARM::BI__builtin_arm_mve_vstrwq_u32: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Address Val1 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val2 = Builder.CreatePointerCast((Val1.emitRawPointer(*this)), Builder.getPtrTy());
  Address Val3 = Address(Val2, llvm::FixedVectorType::get(Int32Ty, 4), CharUnits::fromQuantity(4));
  return Builder.CreateStore(Val0, Val3);
}
case ARM::BI__builtin_arm_mve_vst1q_s8:
case ARM::BI__builtin_arm_mve_vst1q_u8:
case ARM::BI__builtin_arm_mve_vstrbq_s8:
case ARM::BI__builtin_arm_mve_vstrbq_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Address Val1 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val2 = Builder.CreatePointerCast((Val1.emitRawPointer(*this)), Builder.getPtrTy());
  Address Val3 = Address(Val2, llvm::FixedVectorType::get(Int8Ty, 16), CharUnits::fromQuantity(1));
  return Builder.CreateStore(Val0, Val3);
}
case ARM::BI__builtin_arm_mve_vst1q_p_f16:
case ARM::BI__builtin_arm_mve_vst1q_p_f32:
case ARM::BI__builtin_arm_mve_vst1q_p_s16:
case ARM::BI__builtin_arm_mve_vst1q_p_s32:
case ARM::BI__builtin_arm_mve_vst1q_p_s8:
case ARM::BI__builtin_arm_mve_vst1q_p_u16:
case ARM::BI__builtin_arm_mve_vst1q_p_u32:
case ARM::BI__builtin_arm_mve_vst1q_p_u8:
case ARM::BI__builtin_arm_mve_vstrbq_p_s8:
case ARM::BI__builtin_arm_mve_vstrbq_p_u8:
case ARM::BI__builtin_arm_mve_vstrhq_p_f16:
case ARM::BI__builtin_arm_mve_vstrhq_p_s16:
case ARM::BI__builtin_arm_mve_vstrhq_p_u16:
case ARM::BI__builtin_arm_mve_vstrwq_p_f32:
case ARM::BI__builtin_arm_mve_vstrwq_p_s32:
case ARM::BI__builtin_arm_mve_vstrwq_p_u32: {
  uint32_t  Param0;
  llvm::Type * Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vst1q_p_f16:
    Param0 = 2;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vst1q_p_f32:
    Param0 = 4;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vst1q_p_s16:
    Param0 = 2;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vst1q_p_s32:
    Param0 = 4;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vst1q_p_s8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vst1q_p_u16:
    Param0 = 2;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vst1q_p_u32:
    Param0 = 4;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vst1q_p_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vstrbq_p_s8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vstrbq_p_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_p_f16:
    Param0 = 2;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_p_s16:
    Param0 = 2;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrhq_p_u16:
    Param0 = 2;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_p_f32:
    Param0 = 4;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_p_s32:
    Param0 = 4;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vstrwq_p_u32:
    Param0 = 4;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Address Val1 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val2 = Builder.CreatePointerCast((Val1.emitRawPointer(*this)), Builder.getPtrTy());
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param1}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::masked_store, {Param2, Builder.getPtrTy()}), {Val0, Val2, Val3, Val6});
}
case ARM::BI__builtin_arm_mve_vctp16q_m:
case ARM::BI__builtin_arm_mve_vctp32q_m:
case ARM::BI__builtin_arm_mve_vctp64q_m:
case ARM::BI__builtin_arm_mve_vctp8q_m: {
  llvm::Type * Param0;
  Intrinsic::ID  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vctp16q_m:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vctp16;
    break;
  case ARM::BI__builtin_arm_mve_vctp32q_m:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vctp32;
    break;
  case ARM::BI__builtin_arm_mve_vctp64q_m:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param1 = Intrinsic::arm_mve_vctp64;
    break;
  case ARM::BI__builtin_arm_mve_vctp8q_m:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vctp8;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Param1), {Val3});
  Value *Val5 = Builder.CreateAnd(Val2, Val4);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val5});
  return Builder.CreateIntCast(Val6, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vmovnbq_s16:
case ARM::BI__builtin_arm_mve_vmovnbq_u16: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = EmitScalarExpr(E->getArg(0));
  Value *Val2 = ARMMVEVectorElementReverse(Builder, Val1, static_cast<unsigned>(16));
  llvm::Type *Val3 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val4 = ARMMVEVectorReinterpret(Builder, this, Val2, Val3);
  Value *Val5 = VectorZip(Builder, Val0, Val4);
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int8Ty, 16);
  return Builder.CreateTrunc(Val5, Val6);
}
case ARM::BI__builtin_arm_mve_vmovnbq_s32:
case ARM::BI__builtin_arm_mve_vmovnbq_u32: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = EmitScalarExpr(E->getArg(0));
  Value *Val2 = ARMMVEVectorElementReverse(Builder, Val1, static_cast<unsigned>(32));
  llvm::Type *Val3 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val4 = ARMMVEVectorReinterpret(Builder, this, Val2, Val3);
  Value *Val5 = VectorZip(Builder, Val0, Val4);
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int16Ty, 8);
  return Builder.CreateTrunc(Val5, Val6);
}
case ARM::BI__builtin_arm_mve_vsetq_lane_f16:
case ARM::BI__builtin_arm_mve_vsetq_lane_f32:
case ARM::BI__builtin_arm_mve_vsetq_lane_s16:
case ARM::BI__builtin_arm_mve_vsetq_lane_s32:
case ARM::BI__builtin_arm_mve_vsetq_lane_s64:
case ARM::BI__builtin_arm_mve_vsetq_lane_s8:
case ARM::BI__builtin_arm_mve_vsetq_lane_u16:
case ARM::BI__builtin_arm_mve_vsetq_lane_u32:
case ARM::BI__builtin_arm_mve_vsetq_lane_u64:
case ARM::BI__builtin_arm_mve_vsetq_lane_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = EmitScalarExpr(E->getArg(0));
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  return Builder.CreateInsertElement(Val0, Val1, Val2);
}
case ARM::BI__builtin_arm_mve_vfmaq_m_f16:
case ARM::BI__builtin_arm_mve_vfmaq_m_f32: {
  llvm::Type * Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vfmaq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vfmaq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = EmitScalarExpr(E->getArg(2));
  Value *Val2 = (Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(3));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_fma_predicated, {Param1, Param0}), {Val0, Val2, Val3, Val6});
}
case ARM::BI__builtin_arm_mve_vfmaq_f16:
case ARM::BI__builtin_arm_mve_vfmaq_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vfmaq_f16:
    Param0 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vfmaq_f32:
    Param0 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = EmitScalarExpr(E->getArg(2));
  Value *Val2 = (Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::fma, {Param0}), {Val0, Val2, Val3});
}
case ARM::BI__builtin_arm_mve_vmlaq_n_s16:
case ARM::BI__builtin_arm_mve_vmlaq_n_s32:
case ARM::BI__builtin_arm_mve_vmlaq_n_s8:
case ARM::BI__builtin_arm_mve_vmlaq_n_u16:
case ARM::BI__builtin_arm_mve_vmlaq_n_u32:
case ARM::BI__builtin_arm_mve_vmlaq_n_u8: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = EmitScalarExpr(E->getArg(2));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = Builder.CreateMul(Val0, Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateAdd(Val3, Val4);
}
case ARM::BI__builtin_arm_mve_vaddq_m_n_f16:
case ARM::BI__builtin_arm_mve_vaddq_m_n_f32:
case ARM::BI__builtin_arm_mve_vaddq_m_n_s16:
case ARM::BI__builtin_arm_mve_vaddq_m_n_s32:
case ARM::BI__builtin_arm_mve_vaddq_m_n_s8:
case ARM::BI__builtin_arm_mve_vaddq_m_n_u16:
case ARM::BI__builtin_arm_mve_vaddq_m_n_u32:
case ARM::BI__builtin_arm_mve_vaddq_m_n_u8:
case ARM::BI__builtin_arm_mve_vmulq_m_n_f16:
case ARM::BI__builtin_arm_mve_vmulq_m_n_f32:
case ARM::BI__builtin_arm_mve_vmulq_m_n_s16:
case ARM::BI__builtin_arm_mve_vmulq_m_n_s32:
case ARM::BI__builtin_arm_mve_vmulq_m_n_s8:
case ARM::BI__builtin_arm_mve_vmulq_m_n_u16:
case ARM::BI__builtin_arm_mve_vmulq_m_n_u32:
case ARM::BI__builtin_arm_mve_vmulq_m_n_u8:
case ARM::BI__builtin_arm_mve_vqdmulhq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqdmulhq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqdmulhq_m_n_s8:
case ARM::BI__builtin_arm_mve_vqrdmulhq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqrdmulhq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqrdmulhq_m_n_s8:
case ARM::BI__builtin_arm_mve_vsubq_m_n_f16:
case ARM::BI__builtin_arm_mve_vsubq_m_n_f32:
case ARM::BI__builtin_arm_mve_vsubq_m_n_s16:
case ARM::BI__builtin_arm_mve_vsubq_m_n_s32:
case ARM::BI__builtin_arm_mve_vsubq_m_n_s8:
case ARM::BI__builtin_arm_mve_vsubq_m_n_u16:
case ARM::BI__builtin_arm_mve_vsubq_m_n_u32:
case ARM::BI__builtin_arm_mve_vsubq_m_n_u8: {
  llvm::Type * Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vaddq_m_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_add_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vaddq_m_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_add_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_add_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vaddq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_add_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_add_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vaddq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_add_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vaddq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_add_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_add_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmulq_m_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_mul_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmulq_m_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_mul_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmulq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_mul_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmulq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_mul_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmulq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_mul_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmulq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_mul_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmulq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_mul_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmulq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_mul_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulhq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_qdmulh_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulhq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_qdmulh_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulhq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_qdmulh_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmulhq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_qrdmulh_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmulhq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_qrdmulh_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmulhq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_qrdmulh_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vsubq_m_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_sub_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vsubq_m_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_sub_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vsubq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_sub_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vsubq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_sub_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vsubq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_sub_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vsubq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_sub_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vsubq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_sub_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vsubq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_sub_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = EmitScalarExpr(E->getArg(2));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(3));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val4});
  Value *Val6 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2, Param0}), {Val0, Val2, Val5, Val6});
}
case ARM::BI__builtin_arm_mve_vqdmullbq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqdmullbq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqdmulltq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqdmulltq_m_n_s32: {
  uint32_t  Param0;
  llvm::Type * Param1;
  llvm::Type * Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vqdmullbq_m_n_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmullbq_m_n_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulltq_m_n_s16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulltq_m_n_s32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param2 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = EmitScalarExpr(E->getArg(2));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = EmitScalarExpr(E->getArg(3));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param1}), {Val5});
  Value *Val7 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vqdmull_predicated, {Param2, Param3, Param1}), {Val0, Val2, Val3, Val6, Val7});
}
case ARM::BI__builtin_arm_mve_vhaddq_m_n_s16:
case ARM::BI__builtin_arm_mve_vhaddq_m_n_s32:
case ARM::BI__builtin_arm_mve_vhaddq_m_n_s8:
case ARM::BI__builtin_arm_mve_vhaddq_m_n_u16:
case ARM::BI__builtin_arm_mve_vhaddq_m_n_u32:
case ARM::BI__builtin_arm_mve_vhaddq_m_n_u8:
case ARM::BI__builtin_arm_mve_vhsubq_m_n_s16:
case ARM::BI__builtin_arm_mve_vhsubq_m_n_s32:
case ARM::BI__builtin_arm_mve_vhsubq_m_n_s8:
case ARM::BI__builtin_arm_mve_vhsubq_m_n_u16:
case ARM::BI__builtin_arm_mve_vhsubq_m_n_u32:
case ARM::BI__builtin_arm_mve_vhsubq_m_n_u8:
case ARM::BI__builtin_arm_mve_vqaddq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqaddq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqaddq_m_n_s8:
case ARM::BI__builtin_arm_mve_vqaddq_m_n_u16:
case ARM::BI__builtin_arm_mve_vqaddq_m_n_u32:
case ARM::BI__builtin_arm_mve_vqaddq_m_n_u8:
case ARM::BI__builtin_arm_mve_vqsubq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqsubq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqsubq_m_n_s8:
case ARM::BI__builtin_arm_mve_vqsubq_m_n_u16:
case ARM::BI__builtin_arm_mve_vqsubq_m_n_u32:
case ARM::BI__builtin_arm_mve_vqsubq_m_n_u8: {
  uint32_t  Param0;
  llvm::Type * Param1;
  Intrinsic::ID  Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vhaddq_m_n_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_hadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_m_n_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_hadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_m_n_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_hadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_m_n_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_hadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_m_n_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_hadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_m_n_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_hadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_m_n_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_hsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_m_n_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_hsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_m_n_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_hsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_m_n_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_hsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_m_n_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_hsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_m_n_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_hsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_m_n_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_qadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_m_n_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_qadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_m_n_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_qadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_m_n_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_qadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_m_n_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_qadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_m_n_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_qadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_m_n_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_qsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_m_n_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_qsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_m_n_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_qsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_m_n_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_qsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_m_n_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_qsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_m_n_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_qsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = EmitScalarExpr(E->getArg(2));
  Value *Val2 = ARMMVEVectorSplat(Builder, Val1);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = EmitScalarExpr(E->getArg(3));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param1}), {Val5});
  Value *Val7 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateCall(CGM.getIntrinsic(Param2, {Param3, Param1}), {Val0, Val2, Val3, Val6, Val7});
}
case ARM::BI__builtin_arm_mve_vfmsq_m_f16:
case ARM::BI__builtin_arm_mve_vfmsq_m_f32: {
  llvm::Type * Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vfmsq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vfmsq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = EmitScalarExpr(E->getArg(2));
  Value *Val2 = Builder.CreateFNeg(Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(3));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_fma_predicated, {Param1, Param0}), {Val0, Val2, Val3, Val6});
}
case ARM::BI__builtin_arm_mve_vfmsq_f16:
case ARM::BI__builtin_arm_mve_vfmsq_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vfmsq_f16:
    Param0 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vfmsq_f32:
    Param0 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = EmitScalarExpr(E->getArg(2));
  Value *Val2 = Builder.CreateFNeg(Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::fma, {Param0}), {Val0, Val2, Val3});
}
case ARM::BI__builtin_arm_mve_vabsq_m_f16:
case ARM::BI__builtin_arm_mve_vabsq_m_f32:
case ARM::BI__builtin_arm_mve_vabsq_m_s16:
case ARM::BI__builtin_arm_mve_vabsq_m_s32:
case ARM::BI__builtin_arm_mve_vabsq_m_s8:
case ARM::BI__builtin_arm_mve_vclsq_m_s16:
case ARM::BI__builtin_arm_mve_vclsq_m_s32:
case ARM::BI__builtin_arm_mve_vclsq_m_s8:
case ARM::BI__builtin_arm_mve_vclzq_m_s16:
case ARM::BI__builtin_arm_mve_vclzq_m_s32:
case ARM::BI__builtin_arm_mve_vclzq_m_s8:
case ARM::BI__builtin_arm_mve_vclzq_m_u16:
case ARM::BI__builtin_arm_mve_vclzq_m_u32:
case ARM::BI__builtin_arm_mve_vclzq_m_u8:
case ARM::BI__builtin_arm_mve_vmvnq_m_s16:
case ARM::BI__builtin_arm_mve_vmvnq_m_s32:
case ARM::BI__builtin_arm_mve_vmvnq_m_s8:
case ARM::BI__builtin_arm_mve_vmvnq_m_u16:
case ARM::BI__builtin_arm_mve_vmvnq_m_u32:
case ARM::BI__builtin_arm_mve_vmvnq_m_u8:
case ARM::BI__builtin_arm_mve_vnegq_m_f16:
case ARM::BI__builtin_arm_mve_vnegq_m_f32:
case ARM::BI__builtin_arm_mve_vnegq_m_s16:
case ARM::BI__builtin_arm_mve_vnegq_m_s32:
case ARM::BI__builtin_arm_mve_vnegq_m_s8:
case ARM::BI__builtin_arm_mve_vqabsq_m_s16:
case ARM::BI__builtin_arm_mve_vqabsq_m_s32:
case ARM::BI__builtin_arm_mve_vqabsq_m_s8:
case ARM::BI__builtin_arm_mve_vqnegq_m_s16:
case ARM::BI__builtin_arm_mve_vqnegq_m_s32:
case ARM::BI__builtin_arm_mve_vqnegq_m_s8:
case ARM::BI__builtin_arm_mve_vrndaq_m_f16:
case ARM::BI__builtin_arm_mve_vrndaq_m_f32:
case ARM::BI__builtin_arm_mve_vrndmq_m_f16:
case ARM::BI__builtin_arm_mve_vrndmq_m_f32:
case ARM::BI__builtin_arm_mve_vrndnq_m_f16:
case ARM::BI__builtin_arm_mve_vrndnq_m_f32:
case ARM::BI__builtin_arm_mve_vrndpq_m_f16:
case ARM::BI__builtin_arm_mve_vrndpq_m_f32:
case ARM::BI__builtin_arm_mve_vrndq_m_f16:
case ARM::BI__builtin_arm_mve_vrndq_m_f32:
case ARM::BI__builtin_arm_mve_vrndxq_m_f16:
case ARM::BI__builtin_arm_mve_vrndxq_m_f32: {
  llvm::Type * Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vabsq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_abs_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vabsq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_abs_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vabsq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_abs_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vabsq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_abs_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vabsq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_abs_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vclsq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_cls_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vclsq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_cls_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vclsq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_cls_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vclzq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_clz_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vclzq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_clz_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vclzq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_clz_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vclzq_m_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_clz_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vclzq_m_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_clz_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vclzq_m_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_clz_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_mvn_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_mvn_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_mvn_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_m_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_mvn_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_m_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_mvn_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_m_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_mvn_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vnegq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_neg_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vnegq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_neg_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vnegq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_neg_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vnegq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_neg_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vnegq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_neg_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqabsq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_qabs_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqabsq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_qabs_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqabsq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_qabs_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqnegq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_qneg_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqnegq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_qneg_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqnegq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_qneg_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrndaq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vrinta_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrndaq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vrinta_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrndmq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vrintm_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrndmq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vrintm_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrndnq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vrintn_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrndnq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vrintn_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrndpq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vrintp_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrndpq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vrintp_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrndq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vrintz_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrndq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vrintz_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrndxq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vrintx_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrndxq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vrintx_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = EmitScalarExpr(E->getArg(2));
  Value *Val2 = Builder.CreateIntCast(Val1, Int32Ty, false);
  Value *Val3 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val2});
  Value *Val4 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2, Param0}), {Val0, Val3, Val4});
}
case ARM::BI__builtin_arm_mve_vaddq_m_f16:
case ARM::BI__builtin_arm_mve_vaddq_m_f32:
case ARM::BI__builtin_arm_mve_vaddq_m_s16:
case ARM::BI__builtin_arm_mve_vaddq_m_s32:
case ARM::BI__builtin_arm_mve_vaddq_m_s8:
case ARM::BI__builtin_arm_mve_vaddq_m_u16:
case ARM::BI__builtin_arm_mve_vaddq_m_u32:
case ARM::BI__builtin_arm_mve_vaddq_m_u8:
case ARM::BI__builtin_arm_mve_vmulq_m_f16:
case ARM::BI__builtin_arm_mve_vmulq_m_f32:
case ARM::BI__builtin_arm_mve_vmulq_m_s16:
case ARM::BI__builtin_arm_mve_vmulq_m_s32:
case ARM::BI__builtin_arm_mve_vmulq_m_s8:
case ARM::BI__builtin_arm_mve_vmulq_m_u16:
case ARM::BI__builtin_arm_mve_vmulq_m_u32:
case ARM::BI__builtin_arm_mve_vmulq_m_u8:
case ARM::BI__builtin_arm_mve_vqdmulhq_m_s16:
case ARM::BI__builtin_arm_mve_vqdmulhq_m_s32:
case ARM::BI__builtin_arm_mve_vqdmulhq_m_s8:
case ARM::BI__builtin_arm_mve_vqrdmulhq_m_s16:
case ARM::BI__builtin_arm_mve_vqrdmulhq_m_s32:
case ARM::BI__builtin_arm_mve_vqrdmulhq_m_s8:
case ARM::BI__builtin_arm_mve_vqshluq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqshluq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqshluq_m_n_s8:
case ARM::BI__builtin_arm_mve_vshlq_m_n_s16:
case ARM::BI__builtin_arm_mve_vshlq_m_n_s32:
case ARM::BI__builtin_arm_mve_vshlq_m_n_s8:
case ARM::BI__builtin_arm_mve_vshlq_m_n_u16:
case ARM::BI__builtin_arm_mve_vshlq_m_n_u32:
case ARM::BI__builtin_arm_mve_vshlq_m_n_u8:
case ARM::BI__builtin_arm_mve_vsubq_m_f16:
case ARM::BI__builtin_arm_mve_vsubq_m_f32:
case ARM::BI__builtin_arm_mve_vsubq_m_s16:
case ARM::BI__builtin_arm_mve_vsubq_m_s32:
case ARM::BI__builtin_arm_mve_vsubq_m_s8:
case ARM::BI__builtin_arm_mve_vsubq_m_u16:
case ARM::BI__builtin_arm_mve_vsubq_m_u32:
case ARM::BI__builtin_arm_mve_vsubq_m_u8: {
  llvm::Type * Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vaddq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_add_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vaddq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_add_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_add_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vaddq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_add_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_add_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vaddq_m_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_add_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vaddq_m_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_add_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddq_m_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_add_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmulq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_mul_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmulq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_mul_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmulq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_mul_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmulq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_mul_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmulq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_mul_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmulq_m_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_mul_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmulq_m_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_mul_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmulq_m_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_mul_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulhq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_qdmulh_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulhq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_qdmulh_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulhq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_qdmulh_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmulhq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_qrdmulh_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmulhq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_qrdmulh_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrdmulhq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_qrdmulh_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqshluq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_vqshlu_imm_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshluq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_vqshlu_imm_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshluq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_vqshlu_imm_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_shl_imm_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_shl_imm_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_shl_imm_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_shl_imm_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_shl_imm_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_shl_imm_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vsubq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_sub_predicated;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vsubq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_sub_predicated;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vsubq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_sub_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vsubq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_sub_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vsubq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_sub_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vsubq_m_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = Intrinsic::arm_mve_sub_predicated;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vsubq_m_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = Intrinsic::arm_mve_sub_predicated;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vsubq_m_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param1 = Intrinsic::arm_mve_sub_predicated;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = EmitScalarExpr(E->getArg(2));
  Value *Val2 = EmitScalarExpr(E->getArg(3));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val3});
  Value *Val5 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2, Param0}), {Val0, Val1, Val4, Val5});
}
case ARM::BI__builtin_arm_mve_vabdq_m_f16:
case ARM::BI__builtin_arm_mve_vabdq_m_f32:
case ARM::BI__builtin_arm_mve_vabdq_m_s16:
case ARM::BI__builtin_arm_mve_vabdq_m_s32:
case ARM::BI__builtin_arm_mve_vabdq_m_s8:
case ARM::BI__builtin_arm_mve_vabdq_m_u16:
case ARM::BI__builtin_arm_mve_vabdq_m_u32:
case ARM::BI__builtin_arm_mve_vabdq_m_u8:
case ARM::BI__builtin_arm_mve_vhaddq_m_s16:
case ARM::BI__builtin_arm_mve_vhaddq_m_s32:
case ARM::BI__builtin_arm_mve_vhaddq_m_s8:
case ARM::BI__builtin_arm_mve_vhaddq_m_u16:
case ARM::BI__builtin_arm_mve_vhaddq_m_u32:
case ARM::BI__builtin_arm_mve_vhaddq_m_u8:
case ARM::BI__builtin_arm_mve_vhsubq_m_s16:
case ARM::BI__builtin_arm_mve_vhsubq_m_s32:
case ARM::BI__builtin_arm_mve_vhsubq_m_s8:
case ARM::BI__builtin_arm_mve_vhsubq_m_u16:
case ARM::BI__builtin_arm_mve_vhsubq_m_u32:
case ARM::BI__builtin_arm_mve_vhsubq_m_u8:
case ARM::BI__builtin_arm_mve_vmaxnmq_m_f16:
case ARM::BI__builtin_arm_mve_vmaxnmq_m_f32:
case ARM::BI__builtin_arm_mve_vmaxq_m_s16:
case ARM::BI__builtin_arm_mve_vmaxq_m_s32:
case ARM::BI__builtin_arm_mve_vmaxq_m_s8:
case ARM::BI__builtin_arm_mve_vmaxq_m_u16:
case ARM::BI__builtin_arm_mve_vmaxq_m_u32:
case ARM::BI__builtin_arm_mve_vmaxq_m_u8:
case ARM::BI__builtin_arm_mve_vminnmq_m_f16:
case ARM::BI__builtin_arm_mve_vminnmq_m_f32:
case ARM::BI__builtin_arm_mve_vminq_m_s16:
case ARM::BI__builtin_arm_mve_vminq_m_s32:
case ARM::BI__builtin_arm_mve_vminq_m_s8:
case ARM::BI__builtin_arm_mve_vminq_m_u16:
case ARM::BI__builtin_arm_mve_vminq_m_u32:
case ARM::BI__builtin_arm_mve_vminq_m_u8:
case ARM::BI__builtin_arm_mve_vmulhq_m_s16:
case ARM::BI__builtin_arm_mve_vmulhq_m_s32:
case ARM::BI__builtin_arm_mve_vmulhq_m_s8:
case ARM::BI__builtin_arm_mve_vmulhq_m_u16:
case ARM::BI__builtin_arm_mve_vmulhq_m_u32:
case ARM::BI__builtin_arm_mve_vmulhq_m_u8:
case ARM::BI__builtin_arm_mve_vqaddq_m_s16:
case ARM::BI__builtin_arm_mve_vqaddq_m_s32:
case ARM::BI__builtin_arm_mve_vqaddq_m_s8:
case ARM::BI__builtin_arm_mve_vqaddq_m_u16:
case ARM::BI__builtin_arm_mve_vqaddq_m_u32:
case ARM::BI__builtin_arm_mve_vqaddq_m_u8:
case ARM::BI__builtin_arm_mve_vqshlq_m_n_s16:
case ARM::BI__builtin_arm_mve_vqshlq_m_n_s32:
case ARM::BI__builtin_arm_mve_vqshlq_m_n_s8:
case ARM::BI__builtin_arm_mve_vqshlq_m_n_u16:
case ARM::BI__builtin_arm_mve_vqshlq_m_n_u32:
case ARM::BI__builtin_arm_mve_vqshlq_m_n_u8:
case ARM::BI__builtin_arm_mve_vqsubq_m_s16:
case ARM::BI__builtin_arm_mve_vqsubq_m_s32:
case ARM::BI__builtin_arm_mve_vqsubq_m_s8:
case ARM::BI__builtin_arm_mve_vqsubq_m_u16:
case ARM::BI__builtin_arm_mve_vqsubq_m_u32:
case ARM::BI__builtin_arm_mve_vqsubq_m_u8:
case ARM::BI__builtin_arm_mve_vrhaddq_m_s16:
case ARM::BI__builtin_arm_mve_vrhaddq_m_s32:
case ARM::BI__builtin_arm_mve_vrhaddq_m_s8:
case ARM::BI__builtin_arm_mve_vrhaddq_m_u16:
case ARM::BI__builtin_arm_mve_vrhaddq_m_u32:
case ARM::BI__builtin_arm_mve_vrhaddq_m_u8:
case ARM::BI__builtin_arm_mve_vrmulhq_m_s16:
case ARM::BI__builtin_arm_mve_vrmulhq_m_s32:
case ARM::BI__builtin_arm_mve_vrmulhq_m_s8:
case ARM::BI__builtin_arm_mve_vrmulhq_m_u16:
case ARM::BI__builtin_arm_mve_vrmulhq_m_u32:
case ARM::BI__builtin_arm_mve_vrmulhq_m_u8:
case ARM::BI__builtin_arm_mve_vrshrq_m_n_s16:
case ARM::BI__builtin_arm_mve_vrshrq_m_n_s32:
case ARM::BI__builtin_arm_mve_vrshrq_m_n_s8:
case ARM::BI__builtin_arm_mve_vrshrq_m_n_u16:
case ARM::BI__builtin_arm_mve_vrshrq_m_n_u32:
case ARM::BI__builtin_arm_mve_vrshrq_m_n_u8:
case ARM::BI__builtin_arm_mve_vshrq_m_n_s16:
case ARM::BI__builtin_arm_mve_vshrq_m_n_s32:
case ARM::BI__builtin_arm_mve_vshrq_m_n_s8:
case ARM::BI__builtin_arm_mve_vshrq_m_n_u16:
case ARM::BI__builtin_arm_mve_vshrq_m_n_u32:
case ARM::BI__builtin_arm_mve_vshrq_m_n_u8: {
  uint32_t  Param0;
  llvm::Type * Param1;
  Intrinsic::ID  Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vabdq_m_f16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_abd_predicated;
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vabdq_m_f32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_abd_predicated;
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vabdq_m_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_abd_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vabdq_m_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_abd_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vabdq_m_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_abd_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vabdq_m_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_abd_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vabdq_m_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_abd_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vabdq_m_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_abd_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_m_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_hadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_m_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_hadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_m_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_hadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_m_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_hadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_m_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_hadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_m_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_hadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_m_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_hsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_m_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_hsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_m_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_hsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_m_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_hsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_m_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_hsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_m_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_hsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmaxnmq_m_f16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_max_predicated;
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmaxnmq_m_f32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_max_predicated;
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmaxq_m_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_max_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmaxq_m_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_max_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmaxq_m_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_max_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmaxq_m_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_max_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmaxq_m_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_max_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmaxq_m_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_max_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vminnmq_m_f16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_min_predicated;
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vminnmq_m_f32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_min_predicated;
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vminq_m_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_min_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vminq_m_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_min_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vminq_m_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_min_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vminq_m_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_min_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vminq_m_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_min_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vminq_m_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_min_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_m_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_mulh_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_m_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_mulh_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_m_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_mulh_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_m_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_mulh_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_m_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_mulh_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_m_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_mulh_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_m_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_qadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_m_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_qadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_m_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_qadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_m_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_qadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_m_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_qadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqaddq_m_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_qadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_n_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vqshl_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_n_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vqshl_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_n_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_vqshl_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_n_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vqshl_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_n_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vqshl_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_n_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_vqshl_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_m_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_qsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_m_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_qsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_m_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_qsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_m_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_qsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_m_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_qsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqsubq_m_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_qsub_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_m_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_rhadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_m_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_rhadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_m_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_rhadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_m_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_rhadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_m_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_rhadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_m_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_rhadd_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_m_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_rmulh_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_m_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_rmulh_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_m_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_rmulh_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_m_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_rmulh_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_m_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_rmulh_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_m_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_rmulh_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_m_n_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vrshr_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_m_n_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vrshr_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_m_n_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_vrshr_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_m_n_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vrshr_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_m_n_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vrshr_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_m_n_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_vrshr_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshrq_m_n_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_shr_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshrq_m_n_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_shr_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshrq_m_n_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_shr_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshrq_m_n_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_shr_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshrq_m_n_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_shr_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshrq_m_n_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_shr_imm_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = EmitScalarExpr(E->getArg(2));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = EmitScalarExpr(E->getArg(3));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param1}), {Val4});
  Value *Val6 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateCall(CGM.getIntrinsic(Param2, {Param3, Param1}), {Val0, Val1, Val2, Val5, Val6});
}
case ARM::BI__builtin_arm_mve_vmullbq_poly_m_p16:
case ARM::BI__builtin_arm_mve_vmullbq_poly_m_p8:
case ARM::BI__builtin_arm_mve_vmulltq_poly_m_p16:
case ARM::BI__builtin_arm_mve_vmulltq_poly_m_p8:
case ARM::BI__builtin_arm_mve_vqdmullbq_m_s16:
case ARM::BI__builtin_arm_mve_vqdmullbq_m_s32:
case ARM::BI__builtin_arm_mve_vqdmulltq_m_s16:
case ARM::BI__builtin_arm_mve_vqdmulltq_m_s32: {
  uint32_t  Param0;
  llvm::Type * Param1;
  Intrinsic::ID  Param2;
  llvm::Type * Param3;
  llvm::Type * Param4;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmullbq_poly_m_p16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_mull_poly_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmullbq_poly_m_p8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_mull_poly_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_poly_m_p16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_mull_poly_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_poly_m_p8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_mull_poly_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqdmullbq_m_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vqdmull_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmullbq_m_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param2 = Intrinsic::arm_mve_vqdmull_predicated;
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulltq_m_s16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vqdmull_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqdmulltq_m_s32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param2 = Intrinsic::arm_mve_vqdmull_predicated;
    Param3 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = EmitScalarExpr(E->getArg(2));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = EmitScalarExpr(E->getArg(3));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param1}), {Val4});
  Value *Val6 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateCall(CGM.getIntrinsic(Param2, {Param3, Param4, Param1}), {Val0, Val1, Val2, Val5, Val6});
}
case ARM::BI__builtin_arm_mve_vmullbq_int_m_s16:
case ARM::BI__builtin_arm_mve_vmullbq_int_m_s32:
case ARM::BI__builtin_arm_mve_vmullbq_int_m_s8:
case ARM::BI__builtin_arm_mve_vmullbq_int_m_u16:
case ARM::BI__builtin_arm_mve_vmullbq_int_m_u32:
case ARM::BI__builtin_arm_mve_vmullbq_int_m_u8:
case ARM::BI__builtin_arm_mve_vmulltq_int_m_s16:
case ARM::BI__builtin_arm_mve_vmulltq_int_m_s32:
case ARM::BI__builtin_arm_mve_vmulltq_int_m_s8:
case ARM::BI__builtin_arm_mve_vmulltq_int_m_u16:
case ARM::BI__builtin_arm_mve_vmulltq_int_m_u32:
case ARM::BI__builtin_arm_mve_vmulltq_int_m_u8:
case ARM::BI__builtin_arm_mve_vshllbq_m_n_s16:
case ARM::BI__builtin_arm_mve_vshllbq_m_n_s8:
case ARM::BI__builtin_arm_mve_vshllbq_m_n_u16:
case ARM::BI__builtin_arm_mve_vshllbq_m_n_u8:
case ARM::BI__builtin_arm_mve_vshlltq_m_n_s16:
case ARM::BI__builtin_arm_mve_vshlltq_m_n_s8:
case ARM::BI__builtin_arm_mve_vshlltq_m_n_u16:
case ARM::BI__builtin_arm_mve_vshlltq_m_n_u8: {
  uint32_t  Param0;
  uint32_t  Param1;
  llvm::Type * Param2;
  Intrinsic::ID  Param3;
  llvm::Type * Param4;
  llvm::Type * Param5;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmullbq_int_m_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = Intrinsic::arm_mve_mull_int_predicated;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmullbq_int_m_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param3 = Intrinsic::arm_mve_mull_int_predicated;
    Param4 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmullbq_int_m_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = Intrinsic::arm_mve_mull_int_predicated;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmullbq_int_m_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = Intrinsic::arm_mve_mull_int_predicated;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmullbq_int_m_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param3 = Intrinsic::arm_mve_mull_int_predicated;
    Param4 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmullbq_int_m_u8:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = Intrinsic::arm_mve_mull_int_predicated;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_m_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = Intrinsic::arm_mve_mull_int_predicated;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_m_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param3 = Intrinsic::arm_mve_mull_int_predicated;
    Param4 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_m_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = Intrinsic::arm_mve_mull_int_predicated;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_m_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = Intrinsic::arm_mve_mull_int_predicated;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_m_u32:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 2);
    Param3 = Intrinsic::arm_mve_mull_int_predicated;
    Param4 = llvm::FixedVectorType::get(Int64Ty, 2);
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_m_u8:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = Intrinsic::arm_mve_mull_int_predicated;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshllbq_m_n_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = Intrinsic::arm_mve_vshll_imm_predicated;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshllbq_m_n_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = Intrinsic::arm_mve_vshll_imm_predicated;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshllbq_m_n_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = Intrinsic::arm_mve_vshll_imm_predicated;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshllbq_m_n_u8:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = Intrinsic::arm_mve_vshll_imm_predicated;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshlltq_m_n_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = Intrinsic::arm_mve_vshll_imm_predicated;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshlltq_m_n_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = Intrinsic::arm_mve_vshll_imm_predicated;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshlltq_m_n_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = Intrinsic::arm_mve_vshll_imm_predicated;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshlltq_m_n_u8:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = Intrinsic::arm_mve_vshll_imm_predicated;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param5 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = EmitScalarExpr(E->getArg(2));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val4 = EmitScalarExpr(E->getArg(3));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param2}), {Val5});
  Value *Val7 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateCall(CGM.getIntrinsic(Param3, {Param4, Param5, Param2}), {Val0, Val1, Val2, Val3, Val6, Val7});
}
case ARM::BI__builtin_arm_mve_vqrshlq_m_s16:
case ARM::BI__builtin_arm_mve_vqrshlq_m_s32:
case ARM::BI__builtin_arm_mve_vqrshlq_m_s8:
case ARM::BI__builtin_arm_mve_vqrshlq_m_u16:
case ARM::BI__builtin_arm_mve_vqrshlq_m_u32:
case ARM::BI__builtin_arm_mve_vqrshlq_m_u8:
case ARM::BI__builtin_arm_mve_vqshlq_m_s16:
case ARM::BI__builtin_arm_mve_vqshlq_m_s32:
case ARM::BI__builtin_arm_mve_vqshlq_m_s8:
case ARM::BI__builtin_arm_mve_vqshlq_m_u16:
case ARM::BI__builtin_arm_mve_vqshlq_m_u32:
case ARM::BI__builtin_arm_mve_vqshlq_m_u8:
case ARM::BI__builtin_arm_mve_vrshlq_m_s16:
case ARM::BI__builtin_arm_mve_vrshlq_m_s32:
case ARM::BI__builtin_arm_mve_vrshlq_m_s8:
case ARM::BI__builtin_arm_mve_vrshlq_m_u16:
case ARM::BI__builtin_arm_mve_vrshlq_m_u32:
case ARM::BI__builtin_arm_mve_vrshlq_m_u8:
case ARM::BI__builtin_arm_mve_vshlq_m_s16:
case ARM::BI__builtin_arm_mve_vshlq_m_s32:
case ARM::BI__builtin_arm_mve_vshlq_m_s8:
case ARM::BI__builtin_arm_mve_vshlq_m_u16:
case ARM::BI__builtin_arm_mve_vshlq_m_u32:
case ARM::BI__builtin_arm_mve_vshlq_m_u8: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  llvm::Type * Param3;
  llvm::Type * Param4;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vqrshlq_m_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_m_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_m_s8:
    Param0 = 1;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_m_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_m_u32:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqrshlq_m_u8:
    Param0 = 1;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_s8:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vqshlq_m_u8:
    Param0 = 1;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_m_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_m_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_m_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_m_u16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_m_u32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_m_u8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_u16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_u32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vshlq_m_u8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = EmitScalarExpr(E->getArg(2));
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param2);
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param3}), {Val6});
  Value *Val8 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vshl_vector_predicated, {Param4, Param4, Param3}), {Val0, Val1, Val2, Val3, Val4, Val7, Val8});
}
case ARM::BI__builtin_arm_mve_vaddlvaq_s32:
case ARM::BI__builtin_arm_mve_vaddlvaq_u32:
case ARM::BI__builtin_arm_mve_vaddvaq_s16:
case ARM::BI__builtin_arm_mve_vaddvaq_s32:
case ARM::BI__builtin_arm_mve_vaddvaq_s8:
case ARM::BI__builtin_arm_mve_vaddvaq_u16:
case ARM::BI__builtin_arm_mve_vaddvaq_u32:
case ARM::BI__builtin_arm_mve_vaddvaq_u8: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vaddlvaq_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_addlv;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddlvaq_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_addlv;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddvaq_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_addv;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vaddvaq_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_addv;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddvaq_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_addv;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vaddvaq_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_addv;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vaddvaq_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_addv;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddvaq_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_addv;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2}), {Val0, Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateAdd(Val2, Val3);
}
case ARM::BI__builtin_arm_mve_vaddlvaq_p_s32:
case ARM::BI__builtin_arm_mve_vaddlvaq_p_u32:
case ARM::BI__builtin_arm_mve_vaddvaq_p_s16:
case ARM::BI__builtin_arm_mve_vaddvaq_p_s32:
case ARM::BI__builtin_arm_mve_vaddvaq_p_s8:
case ARM::BI__builtin_arm_mve_vaddvaq_p_u16:
case ARM::BI__builtin_arm_mve_vaddvaq_p_u32:
case ARM::BI__builtin_arm_mve_vaddvaq_p_u8: {
  uint32_t  Param0;
  llvm::Type * Param1;
  Intrinsic::ID  Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vaddlvaq_p_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_addlv_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddlvaq_p_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_addlv_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddvaq_p_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_addv_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vaddvaq_p_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_addv_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddvaq_p_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_addv_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vaddvaq_p_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_addv_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vaddvaq_p_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_addv_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vaddvaq_p_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_addv_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param1}), {Val3});
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Param2, {Param3, Param1}), {Val0, Val1, Val4});
  Value *Val6 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateAdd(Val5, Val6);
}
case ARM::BI__builtin_arm_mve_vcvtq_m_f16_s16:
case ARM::BI__builtin_arm_mve_vcvtq_m_f16_u16:
case ARM::BI__builtin_arm_mve_vcvtq_m_f32_s32:
case ARM::BI__builtin_arm_mve_vcvtq_m_f32_u32:
case ARM::BI__builtin_arm_mve_vcvtq_m_s16_f16:
case ARM::BI__builtin_arm_mve_vcvtq_m_s32_f32:
case ARM::BI__builtin_arm_mve_vcvtq_m_u16_f16:
case ARM::BI__builtin_arm_mve_vcvtq_m_u32_f32: {
  uint32_t  Param0;
  llvm::Type * Param1;
  llvm::Type * Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtq_m_f16_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_m_f16_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_m_f32_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_m_f32_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_m_s16_f16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_m_s32_f32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_m_u16_f16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_m_u32_f32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param1}), {Val3});
  Value *Val5 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcvt_fp_int_predicated, {Param2, Param3, Param1}), {Val0, Val1, Val4, Val5});
}
case ARM::BI__builtin_arm_mve_vrev16q_m_s8:
case ARM::BI__builtin_arm_mve_vrev16q_m_u8:
case ARM::BI__builtin_arm_mve_vrev32q_m_f16:
case ARM::BI__builtin_arm_mve_vrev32q_m_s16:
case ARM::BI__builtin_arm_mve_vrev32q_m_s8:
case ARM::BI__builtin_arm_mve_vrev32q_m_u16:
case ARM::BI__builtin_arm_mve_vrev32q_m_u8:
case ARM::BI__builtin_arm_mve_vrev64q_m_f16:
case ARM::BI__builtin_arm_mve_vrev64q_m_f32:
case ARM::BI__builtin_arm_mve_vrev64q_m_s16:
case ARM::BI__builtin_arm_mve_vrev64q_m_s32:
case ARM::BI__builtin_arm_mve_vrev64q_m_s8:
case ARM::BI__builtin_arm_mve_vrev64q_m_u16:
case ARM::BI__builtin_arm_mve_vrev64q_m_u32:
case ARM::BI__builtin_arm_mve_vrev64q_m_u8: {
  uint32_t  Param0;
  llvm::Type * Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vrev16q_m_s8:
    Param0 = 16;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrev16q_m_u8:
    Param0 = 16;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrev32q_m_f16:
    Param0 = 32;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrev32q_m_s16:
    Param0 = 32;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrev32q_m_s8:
    Param0 = 32;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrev32q_m_u16:
    Param0 = 32;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrev32q_m_u8:
    Param0 = 32;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_m_f16:
    Param0 = 64;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_m_f32:
    Param0 = 64;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_m_s16:
    Param0 = 64;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_m_s32:
    Param0 = 64;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_m_s8:
    Param0 = 64;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_m_u16:
    Param0 = 64;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_m_u32:
    Param0 = 64;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_m_u8:
    Param0 = 64;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val2 = EmitScalarExpr(E->getArg(2));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param1}), {Val3});
  Value *Val5 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vrev_predicated, {Param2, Param1}), {Val0, Val1, Val4, Val5});
}
case ARM::BI__builtin_arm_mve_vmovlbq_m_s16:
case ARM::BI__builtin_arm_mve_vmovlbq_m_s8:
case ARM::BI__builtin_arm_mve_vmovlbq_m_u16:
case ARM::BI__builtin_arm_mve_vmovlbq_m_u8:
case ARM::BI__builtin_arm_mve_vmovltq_m_s16:
case ARM::BI__builtin_arm_mve_vmovltq_m_s8:
case ARM::BI__builtin_arm_mve_vmovltq_m_u16:
case ARM::BI__builtin_arm_mve_vmovltq_m_u8: {
  uint32_t  Param0;
  uint32_t  Param1;
  llvm::Type * Param2;
  llvm::Type * Param3;
  llvm::Type * Param4;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmovlbq_m_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmovlbq_m_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmovlbq_m_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmovlbq_m_u8:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmovltq_m_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmovltq_m_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmovltq_m_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmovltq_m_u8:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  Value *Val1 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param2}), {Val4});
  Value *Val6 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vmovl_predicated, {Param3, Param4, Param2}), {Val0, Val1, Val2, Val5, Val6});
}
case ARM::BI__builtin_arm_mve_vstrhq_s32:
case ARM::BI__builtin_arm_mve_vstrhq_u32: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 4);
  Value *Val2 = Builder.CreateTrunc(Val0, Val1);
  Address Val3 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val4 = Builder.CreatePointerCast((Val3.emitRawPointer(*this)), Builder.getPtrTy());
  Address Val5 = Address(Val4, llvm::FixedVectorType::get(Int16Ty, 4), CharUnits::fromQuantity(2));
  return Builder.CreateStore(Val2, Val5);
}
case ARM::BI__builtin_arm_mve_vstrhq_p_s32:
case ARM::BI__builtin_arm_mve_vstrhq_p_u32: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 4);
  Value *Val2 = Builder.CreateTrunc(Val0, Val1);
  Address Val3 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val4 = Builder.CreatePointerCast((Val3.emitRawPointer(*this)), Builder.getPtrTy());
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, 2);
  Value *Val6 = EmitScalarExpr(E->getArg(2));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val7});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::masked_store, {llvm::FixedVectorType::get(Int16Ty, 4), Builder.getPtrTy()}), {Val2, Val4, Val5, Val8});
}
case ARM::BI__builtin_arm_mve_vandq_m_f16:
case ARM::BI__builtin_arm_mve_vbicq_m_f16:
case ARM::BI__builtin_arm_mve_veorq_m_f16:
case ARM::BI__builtin_arm_mve_vornq_m_f16:
case ARM::BI__builtin_arm_mve_vorrq_m_f16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vandq_m_f16:
    Param0 = Intrinsic::arm_mve_and_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_m_f16:
    Param0 = Intrinsic::arm_mve_bic_predicated;
    break;
  case ARM::BI__builtin_arm_mve_veorq_m_f16:
    Param0 = Intrinsic::arm_mve_eor_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vornq_m_f16:
    Param0 = Intrinsic::arm_mve_orn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_m_f16:
    Param0 = Intrinsic::arm_mve_orr_predicated;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val2 = Builder.CreateBitCast(Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val5 = Builder.CreateBitCast(Val3, Val4);
  Value *Val6 = EmitScalarExpr(E->getArg(3));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val7});
  Value *Val9 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val10 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val11 = Builder.CreateBitCast(Val9, Val10);
  Value *Val12 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val5, Val8, Val11});
  llvm::Type *Val13 = llvm::FixedVectorType::get(HalfTy, 8);
  return Builder.CreateBitCast(Val12, Val13);
}
case ARM::BI__builtin_arm_mve_vandq_m_s16:
case ARM::BI__builtin_arm_mve_vandq_m_u16:
case ARM::BI__builtin_arm_mve_vbicq_m_s16:
case ARM::BI__builtin_arm_mve_vbicq_m_u16:
case ARM::BI__builtin_arm_mve_veorq_m_s16:
case ARM::BI__builtin_arm_mve_veorq_m_u16:
case ARM::BI__builtin_arm_mve_vornq_m_s16:
case ARM::BI__builtin_arm_mve_vornq_m_u16:
case ARM::BI__builtin_arm_mve_vorrq_m_s16:
case ARM::BI__builtin_arm_mve_vorrq_m_u16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vandq_m_s16:
    Param0 = Intrinsic::arm_mve_and_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vandq_m_u16:
    Param0 = Intrinsic::arm_mve_and_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_m_s16:
    Param0 = Intrinsic::arm_mve_bic_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_m_u16:
    Param0 = Intrinsic::arm_mve_bic_predicated;
    break;
  case ARM::BI__builtin_arm_mve_veorq_m_s16:
    Param0 = Intrinsic::arm_mve_eor_predicated;
    break;
  case ARM::BI__builtin_arm_mve_veorq_m_u16:
    Param0 = Intrinsic::arm_mve_eor_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vornq_m_s16:
    Param0 = Intrinsic::arm_mve_orn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vornq_m_u16:
    Param0 = Intrinsic::arm_mve_orn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_m_s16:
    Param0 = Intrinsic::arm_mve_orr_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_m_u16:
    Param0 = Intrinsic::arm_mve_orr_predicated;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val2 = Builder.CreateBitCast(Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val5 = Builder.CreateBitCast(Val3, Val4);
  Value *Val6 = EmitScalarExpr(E->getArg(3));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val7});
  Value *Val9 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val10 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val11 = Builder.CreateBitCast(Val9, Val10);
  Value *Val12 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val5, Val8, Val11});
  llvm::Type *Val13 = llvm::FixedVectorType::get(Int16Ty, 8);
  return Builder.CreateBitCast(Val12, Val13);
}
case ARM::BI__builtin_arm_mve_vmaxaq_s16: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val2 = llvm::Constant::getNullValue(Val1);
  Value *Val3 = Builder.CreateICmpSLT(Val0, Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val5 = llvm::Constant::getNullValue(Val4);
  Value *Val6 = Builder.CreateSub(Val5, Val0);
  Value *Val7 = Builder.CreateSelect(Val3, Val6, Val0);
  Value *Val8 = EmitScalarExpr(E->getArg(0));
  Value *Val9 = Builder.CreateICmpUGE(Val8, Val7);
  return Builder.CreateSelect(Val9, Val8, Val7);
}
case ARM::BI__builtin_arm_mve_vminaq_s16: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val2 = llvm::Constant::getNullValue(Val1);
  Value *Val3 = Builder.CreateICmpSLT(Val0, Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val5 = llvm::Constant::getNullValue(Val4);
  Value *Val6 = Builder.CreateSub(Val5, Val0);
  Value *Val7 = Builder.CreateSelect(Val3, Val6, Val0);
  Value *Val8 = EmitScalarExpr(E->getArg(0));
  Value *Val9 = Builder.CreateICmpULE(Val8, Val7);
  return Builder.CreateSelect(Val9, Val8, Val7);
}
case ARM::BI__builtin_arm_mve_vandq_m_f32:
case ARM::BI__builtin_arm_mve_vbicq_m_f32:
case ARM::BI__builtin_arm_mve_veorq_m_f32:
case ARM::BI__builtin_arm_mve_vornq_m_f32:
case ARM::BI__builtin_arm_mve_vorrq_m_f32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vandq_m_f32:
    Param0 = Intrinsic::arm_mve_and_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_m_f32:
    Param0 = Intrinsic::arm_mve_bic_predicated;
    break;
  case ARM::BI__builtin_arm_mve_veorq_m_f32:
    Param0 = Intrinsic::arm_mve_eor_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vornq_m_f32:
    Param0 = Intrinsic::arm_mve_orn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_m_f32:
    Param0 = Intrinsic::arm_mve_orr_predicated;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val2 = Builder.CreateBitCast(Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val5 = Builder.CreateBitCast(Val3, Val4);
  Value *Val6 = EmitScalarExpr(E->getArg(3));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val7});
  Value *Val9 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val10 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val11 = Builder.CreateBitCast(Val9, Val10);
  Value *Val12 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val5, Val8, Val11});
  llvm::Type *Val13 = llvm::FixedVectorType::get(FloatTy, 4);
  return Builder.CreateBitCast(Val12, Val13);
}
case ARM::BI__builtin_arm_mve_vandq_m_s32:
case ARM::BI__builtin_arm_mve_vandq_m_u32:
case ARM::BI__builtin_arm_mve_vbicq_m_s32:
case ARM::BI__builtin_arm_mve_vbicq_m_u32:
case ARM::BI__builtin_arm_mve_veorq_m_s32:
case ARM::BI__builtin_arm_mve_veorq_m_u32:
case ARM::BI__builtin_arm_mve_vornq_m_s32:
case ARM::BI__builtin_arm_mve_vornq_m_u32:
case ARM::BI__builtin_arm_mve_vorrq_m_s32:
case ARM::BI__builtin_arm_mve_vorrq_m_u32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vandq_m_s32:
    Param0 = Intrinsic::arm_mve_and_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vandq_m_u32:
    Param0 = Intrinsic::arm_mve_and_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_m_s32:
    Param0 = Intrinsic::arm_mve_bic_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_m_u32:
    Param0 = Intrinsic::arm_mve_bic_predicated;
    break;
  case ARM::BI__builtin_arm_mve_veorq_m_s32:
    Param0 = Intrinsic::arm_mve_eor_predicated;
    break;
  case ARM::BI__builtin_arm_mve_veorq_m_u32:
    Param0 = Intrinsic::arm_mve_eor_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vornq_m_s32:
    Param0 = Intrinsic::arm_mve_orn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vornq_m_u32:
    Param0 = Intrinsic::arm_mve_orn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_m_s32:
    Param0 = Intrinsic::arm_mve_orr_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_m_u32:
    Param0 = Intrinsic::arm_mve_orr_predicated;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val2 = Builder.CreateBitCast(Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val5 = Builder.CreateBitCast(Val3, Val4);
  Value *Val6 = EmitScalarExpr(E->getArg(3));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val7});
  Value *Val9 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val10 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val11 = Builder.CreateBitCast(Val9, Val10);
  Value *Val12 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val5, Val8, Val11});
  llvm::Type *Val13 = llvm::FixedVectorType::get(Int32Ty, 4);
  return Builder.CreateBitCast(Val12, Val13);
}
case ARM::BI__builtin_arm_mve_vmaxaq_s32: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val2 = llvm::Constant::getNullValue(Val1);
  Value *Val3 = Builder.CreateICmpSLT(Val0, Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val5 = llvm::Constant::getNullValue(Val4);
  Value *Val6 = Builder.CreateSub(Val5, Val0);
  Value *Val7 = Builder.CreateSelect(Val3, Val6, Val0);
  Value *Val8 = EmitScalarExpr(E->getArg(0));
  Value *Val9 = Builder.CreateICmpUGE(Val8, Val7);
  return Builder.CreateSelect(Val9, Val8, Val7);
}
case ARM::BI__builtin_arm_mve_vminaq_s32: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val2 = llvm::Constant::getNullValue(Val1);
  Value *Val3 = Builder.CreateICmpSLT(Val0, Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val5 = llvm::Constant::getNullValue(Val4);
  Value *Val6 = Builder.CreateSub(Val5, Val0);
  Value *Val7 = Builder.CreateSelect(Val3, Val6, Val0);
  Value *Val8 = EmitScalarExpr(E->getArg(0));
  Value *Val9 = Builder.CreateICmpULE(Val8, Val7);
  return Builder.CreateSelect(Val9, Val8, Val7);
}
case ARM::BI__builtin_arm_mve_vandq_m_s8:
case ARM::BI__builtin_arm_mve_vandq_m_u8:
case ARM::BI__builtin_arm_mve_vbicq_m_s8:
case ARM::BI__builtin_arm_mve_vbicq_m_u8:
case ARM::BI__builtin_arm_mve_veorq_m_s8:
case ARM::BI__builtin_arm_mve_veorq_m_u8:
case ARM::BI__builtin_arm_mve_vornq_m_s8:
case ARM::BI__builtin_arm_mve_vornq_m_u8:
case ARM::BI__builtin_arm_mve_vorrq_m_s8:
case ARM::BI__builtin_arm_mve_vorrq_m_u8: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vandq_m_s8:
    Param0 = Intrinsic::arm_mve_and_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vandq_m_u8:
    Param0 = Intrinsic::arm_mve_and_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_m_s8:
    Param0 = Intrinsic::arm_mve_bic_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_m_u8:
    Param0 = Intrinsic::arm_mve_bic_predicated;
    break;
  case ARM::BI__builtin_arm_mve_veorq_m_s8:
    Param0 = Intrinsic::arm_mve_eor_predicated;
    break;
  case ARM::BI__builtin_arm_mve_veorq_m_u8:
    Param0 = Intrinsic::arm_mve_eor_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vornq_m_s8:
    Param0 = Intrinsic::arm_mve_orn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vornq_m_u8:
    Param0 = Intrinsic::arm_mve_orn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_m_s8:
    Param0 = Intrinsic::arm_mve_orr_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_m_u8:
    Param0 = Intrinsic::arm_mve_orr_predicated;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val2 = Builder.CreateBitCast(Val0, Val1);
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val5 = Builder.CreateBitCast(Val3, Val4);
  Value *Val6 = EmitScalarExpr(E->getArg(3));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val7});
  Value *Val9 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val10 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val11 = Builder.CreateBitCast(Val9, Val10);
  Value *Val12 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val2, Val5, Val8, Val11});
  llvm::Type *Val13 = llvm::FixedVectorType::get(Int8Ty, 16);
  return Builder.CreateBitCast(Val12, Val13);
}
case ARM::BI__builtin_arm_mve_vmaxaq_s8: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val2 = llvm::Constant::getNullValue(Val1);
  Value *Val3 = Builder.CreateICmpSLT(Val0, Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val5 = llvm::Constant::getNullValue(Val4);
  Value *Val6 = Builder.CreateSub(Val5, Val0);
  Value *Val7 = Builder.CreateSelect(Val3, Val6, Val0);
  Value *Val8 = EmitScalarExpr(E->getArg(0));
  Value *Val9 = Builder.CreateICmpUGE(Val8, Val7);
  return Builder.CreateSelect(Val9, Val8, Val7);
}
case ARM::BI__builtin_arm_mve_vminaq_s8: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val2 = llvm::Constant::getNullValue(Val1);
  Value *Val3 = Builder.CreateICmpSLT(Val0, Val2);
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val5 = llvm::Constant::getNullValue(Val4);
  Value *Val6 = Builder.CreateSub(Val5, Val0);
  Value *Val7 = Builder.CreateSelect(Val3, Val6, Val0);
  Value *Val8 = EmitScalarExpr(E->getArg(0));
  Value *Val9 = Builder.CreateICmpULE(Val8, Val7);
  return Builder.CreateSelect(Val9, Val8, Val7);
}
case ARM::BI__builtin_arm_mve_vstrbq_s32:
case ARM::BI__builtin_arm_mve_vstrbq_u32: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int8Ty, 4);
  Value *Val2 = Builder.CreateTrunc(Val0, Val1);
  Address Val3 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val4 = Builder.CreatePointerCast((Val3.emitRawPointer(*this)), Builder.getPtrTy());
  Address Val5 = Address(Val4, llvm::FixedVectorType::get(Int8Ty, 4), CharUnits::fromQuantity(1));
  return Builder.CreateStore(Val2, Val5);
}
case ARM::BI__builtin_arm_mve_vstrbq_p_s32:
case ARM::BI__builtin_arm_mve_vstrbq_p_u32: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int8Ty, 4);
  Value *Val2 = Builder.CreateTrunc(Val0, Val1);
  Address Val3 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val4 = Builder.CreatePointerCast((Val3.emitRawPointer(*this)), Builder.getPtrTy());
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, 1);
  Value *Val6 = EmitScalarExpr(E->getArg(2));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val7});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::masked_store, {llvm::FixedVectorType::get(Int8Ty, 4), Builder.getPtrTy()}), {Val2, Val4, Val5, Val8});
}
case ARM::BI__builtin_arm_mve_vstrbq_s16:
case ARM::BI__builtin_arm_mve_vstrbq_u16: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int8Ty, 8);
  Value *Val2 = Builder.CreateTrunc(Val0, Val1);
  Address Val3 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val4 = Builder.CreatePointerCast((Val3.emitRawPointer(*this)), Builder.getPtrTy());
  Address Val5 = Address(Val4, llvm::FixedVectorType::get(Int8Ty, 8), CharUnits::fromQuantity(1));
  return Builder.CreateStore(Val2, Val5);
}
case ARM::BI__builtin_arm_mve_vstrbq_p_s16:
case ARM::BI__builtin_arm_mve_vstrbq_p_u16: {
  Value *Val0 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val1 = llvm::FixedVectorType::get(Int8Ty, 8);
  Value *Val2 = Builder.CreateTrunc(Val0, Val1);
  Address Val3 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val4 = Builder.CreatePointerCast((Val3.emitRawPointer(*this)), Builder.getPtrTy());
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, 1);
  Value *Val6 = EmitScalarExpr(E->getArg(2));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val7});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::masked_store, {llvm::FixedVectorType::get(Int8Ty, 8), Builder.getPtrTy()}), {Val2, Val4, Val5, Val8});
}
case ARM::BI__builtin_arm_mve_vfmasq_n_f16:
case ARM::BI__builtin_arm_mve_vfmasq_n_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vfmasq_n_f16:
    Param0 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vfmasq_n_f32:
    Param0 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = ARMMVEVectorSplat(Builder, Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = (Val3);
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::fma, {Param0}), {Val2, Val4, Val1});
}
case ARM::BI__builtin_arm_mve_vfmaq_m_n_f16:
case ARM::BI__builtin_arm_mve_vfmaq_m_n_f32: {
  llvm::Type * Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vfmaq_m_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vfmaq_m_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = ARMMVEVectorSplat(Builder, Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = (Val1);
  Value *Val4 = EmitScalarExpr(E->getArg(0));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_fma_predicated, {Param1, Param0}), {Val2, Val3, Val4, Val7});
}
case ARM::BI__builtin_arm_mve_vfmaq_n_f16:
case ARM::BI__builtin_arm_mve_vfmaq_n_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vfmaq_n_f16:
    Param0 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vfmaq_n_f32:
    Param0 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = ARMMVEVectorSplat(Builder, Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = (Val1);
  Value *Val4 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::fma, {Param0}), {Val2, Val3, Val4});
}
case ARM::BI__builtin_arm_mve_vcmpeqq_m_n_f16:
case ARM::BI__builtin_arm_mve_vcmpeqq_m_n_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpeqq_m_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_m_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = ARMMVEVectorSplat(Builder, Val4);
  Value *Val6 = Builder.CreateFCmpOEQ(Val3, Val5);
  Value *Val7 = Builder.CreateAnd(Val2, Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val7});
  return Builder.CreateIntCast(Val8, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpgeq_m_n_f16:
case ARM::BI__builtin_arm_mve_vcmpgeq_m_n_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpgeq_m_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgeq_m_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = ARMMVEVectorSplat(Builder, Val4);
  Value *Val6 = Builder.CreateFCmpOGE(Val3, Val5);
  Value *Val7 = Builder.CreateAnd(Val2, Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val7});
  return Builder.CreateIntCast(Val8, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpgtq_m_n_f16:
case ARM::BI__builtin_arm_mve_vcmpgtq_m_n_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpgtq_m_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgtq_m_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = ARMMVEVectorSplat(Builder, Val4);
  Value *Val6 = Builder.CreateFCmpOGT(Val3, Val5);
  Value *Val7 = Builder.CreateAnd(Val2, Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val7});
  return Builder.CreateIntCast(Val8, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpleq_m_n_f16:
case ARM::BI__builtin_arm_mve_vcmpleq_m_n_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpleq_m_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpleq_m_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = ARMMVEVectorSplat(Builder, Val4);
  Value *Val6 = Builder.CreateFCmpULE(Val3, Val5);
  Value *Val7 = Builder.CreateAnd(Val2, Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val7});
  return Builder.CreateIntCast(Val8, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpltq_m_n_f16:
case ARM::BI__builtin_arm_mve_vcmpltq_m_n_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpltq_m_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpltq_m_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = ARMMVEVectorSplat(Builder, Val4);
  Value *Val6 = Builder.CreateFCmpULT(Val3, Val5);
  Value *Val7 = Builder.CreateAnd(Val2, Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val7});
  return Builder.CreateIntCast(Val8, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpneq_m_n_f16:
case ARM::BI__builtin_arm_mve_vcmpneq_m_n_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpneq_m_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_m_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = ARMMVEVectorSplat(Builder, Val4);
  Value *Val6 = Builder.CreateFCmpUNE(Val3, Val5);
  Value *Val7 = Builder.CreateAnd(Val2, Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val7});
  return Builder.CreateIntCast(Val8, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpeqq_m_n_s16:
case ARM::BI__builtin_arm_mve_vcmpeqq_m_n_s32:
case ARM::BI__builtin_arm_mve_vcmpeqq_m_n_s8:
case ARM::BI__builtin_arm_mve_vcmpeqq_m_n_u16:
case ARM::BI__builtin_arm_mve_vcmpeqq_m_n_u32:
case ARM::BI__builtin_arm_mve_vcmpeqq_m_n_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpeqq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = ARMMVEVectorSplat(Builder, Val4);
  Value *Val6 = Builder.CreateICmpEQ(Val3, Val5);
  Value *Val7 = Builder.CreateAnd(Val2, Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val7});
  return Builder.CreateIntCast(Val8, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpneq_m_n_s16:
case ARM::BI__builtin_arm_mve_vcmpneq_m_n_s32:
case ARM::BI__builtin_arm_mve_vcmpneq_m_n_s8:
case ARM::BI__builtin_arm_mve_vcmpneq_m_n_u16:
case ARM::BI__builtin_arm_mve_vcmpneq_m_n_u32:
case ARM::BI__builtin_arm_mve_vcmpneq_m_n_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpneq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = ARMMVEVectorSplat(Builder, Val4);
  Value *Val6 = Builder.CreateICmpNE(Val3, Val5);
  Value *Val7 = Builder.CreateAnd(Val2, Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val7});
  return Builder.CreateIntCast(Val8, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpgeq_m_n_s16:
case ARM::BI__builtin_arm_mve_vcmpgeq_m_n_s32:
case ARM::BI__builtin_arm_mve_vcmpgeq_m_n_s8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpgeq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgeq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgeq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = ARMMVEVectorSplat(Builder, Val4);
  Value *Val6 = Builder.CreateICmpSGE(Val3, Val5);
  Value *Val7 = Builder.CreateAnd(Val2, Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val7});
  return Builder.CreateIntCast(Val8, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpgtq_m_n_s16:
case ARM::BI__builtin_arm_mve_vcmpgtq_m_n_s32:
case ARM::BI__builtin_arm_mve_vcmpgtq_m_n_s8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpgtq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgtq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgtq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = ARMMVEVectorSplat(Builder, Val4);
  Value *Val6 = Builder.CreateICmpSGT(Val3, Val5);
  Value *Val7 = Builder.CreateAnd(Val2, Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val7});
  return Builder.CreateIntCast(Val8, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpleq_m_n_s16:
case ARM::BI__builtin_arm_mve_vcmpleq_m_n_s32:
case ARM::BI__builtin_arm_mve_vcmpleq_m_n_s8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpleq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpleq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpleq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = ARMMVEVectorSplat(Builder, Val4);
  Value *Val6 = Builder.CreateICmpSLE(Val3, Val5);
  Value *Val7 = Builder.CreateAnd(Val2, Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val7});
  return Builder.CreateIntCast(Val8, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpltq_m_n_s16:
case ARM::BI__builtin_arm_mve_vcmpltq_m_n_s32:
case ARM::BI__builtin_arm_mve_vcmpltq_m_n_s8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpltq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpltq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpltq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = ARMMVEVectorSplat(Builder, Val4);
  Value *Val6 = Builder.CreateICmpSLT(Val3, Val5);
  Value *Val7 = Builder.CreateAnd(Val2, Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val7});
  return Builder.CreateIntCast(Val8, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpcsq_m_n_u16:
case ARM::BI__builtin_arm_mve_vcmpcsq_m_n_u32:
case ARM::BI__builtin_arm_mve_vcmpcsq_m_n_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpcsq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpcsq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpcsq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = ARMMVEVectorSplat(Builder, Val4);
  Value *Val6 = Builder.CreateICmpUGE(Val3, Val5);
  Value *Val7 = Builder.CreateAnd(Val2, Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val7});
  return Builder.CreateIntCast(Val8, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmphiq_m_n_u16:
case ARM::BI__builtin_arm_mve_vcmphiq_m_n_u32:
case ARM::BI__builtin_arm_mve_vcmphiq_m_n_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmphiq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmphiq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmphiq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = ARMMVEVectorSplat(Builder, Val4);
  Value *Val6 = Builder.CreateICmpUGT(Val3, Val5);
  Value *Val7 = Builder.CreateAnd(Val2, Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val7});
  return Builder.CreateIntCast(Val8, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpeqq_m_f16:
case ARM::BI__builtin_arm_mve_vcmpeqq_m_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpeqq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateFCmpOEQ(Val3, Val4);
  Value *Val6 = Builder.CreateAnd(Val2, Val5);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val6});
  return Builder.CreateIntCast(Val7, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpgeq_m_f16:
case ARM::BI__builtin_arm_mve_vcmpgeq_m_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpgeq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgeq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateFCmpOGE(Val3, Val4);
  Value *Val6 = Builder.CreateAnd(Val2, Val5);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val6});
  return Builder.CreateIntCast(Val7, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpgtq_m_f16:
case ARM::BI__builtin_arm_mve_vcmpgtq_m_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpgtq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgtq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateFCmpOGT(Val3, Val4);
  Value *Val6 = Builder.CreateAnd(Val2, Val5);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val6});
  return Builder.CreateIntCast(Val7, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpleq_m_f16:
case ARM::BI__builtin_arm_mve_vcmpleq_m_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpleq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpleq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateFCmpULE(Val3, Val4);
  Value *Val6 = Builder.CreateAnd(Val2, Val5);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val6});
  return Builder.CreateIntCast(Val7, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpltq_m_f16:
case ARM::BI__builtin_arm_mve_vcmpltq_m_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpltq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpltq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateFCmpULT(Val3, Val4);
  Value *Val6 = Builder.CreateAnd(Val2, Val5);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val6});
  return Builder.CreateIntCast(Val7, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpneq_m_f16:
case ARM::BI__builtin_arm_mve_vcmpneq_m_f32: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpneq_m_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_m_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateFCmpUNE(Val3, Val4);
  Value *Val6 = Builder.CreateAnd(Val2, Val5);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val6});
  return Builder.CreateIntCast(Val7, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpeqq_m_s16:
case ARM::BI__builtin_arm_mve_vcmpeqq_m_s32:
case ARM::BI__builtin_arm_mve_vcmpeqq_m_s8:
case ARM::BI__builtin_arm_mve_vcmpeqq_m_u16:
case ARM::BI__builtin_arm_mve_vcmpeqq_m_u32:
case ARM::BI__builtin_arm_mve_vcmpeqq_m_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpeqq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_m_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_m_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpeqq_m_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateICmpEQ(Val3, Val4);
  Value *Val6 = Builder.CreateAnd(Val2, Val5);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val6});
  return Builder.CreateIntCast(Val7, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpneq_m_s16:
case ARM::BI__builtin_arm_mve_vcmpneq_m_s32:
case ARM::BI__builtin_arm_mve_vcmpneq_m_s8:
case ARM::BI__builtin_arm_mve_vcmpneq_m_u16:
case ARM::BI__builtin_arm_mve_vcmpneq_m_u32:
case ARM::BI__builtin_arm_mve_vcmpneq_m_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpneq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_m_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_m_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpneq_m_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateICmpNE(Val3, Val4);
  Value *Val6 = Builder.CreateAnd(Val2, Val5);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val6});
  return Builder.CreateIntCast(Val7, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpgeq_m_s16:
case ARM::BI__builtin_arm_mve_vcmpgeq_m_s32:
case ARM::BI__builtin_arm_mve_vcmpgeq_m_s8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpgeq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgeq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgeq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateICmpSGE(Val3, Val4);
  Value *Val6 = Builder.CreateAnd(Val2, Val5);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val6});
  return Builder.CreateIntCast(Val7, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpgtq_m_s16:
case ARM::BI__builtin_arm_mve_vcmpgtq_m_s32:
case ARM::BI__builtin_arm_mve_vcmpgtq_m_s8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpgtq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgtq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpgtq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateICmpSGT(Val3, Val4);
  Value *Val6 = Builder.CreateAnd(Val2, Val5);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val6});
  return Builder.CreateIntCast(Val7, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpleq_m_s16:
case ARM::BI__builtin_arm_mve_vcmpleq_m_s32:
case ARM::BI__builtin_arm_mve_vcmpleq_m_s8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpleq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpleq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpleq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateICmpSLE(Val3, Val4);
  Value *Val6 = Builder.CreateAnd(Val2, Val5);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val6});
  return Builder.CreateIntCast(Val7, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpltq_m_s16:
case ARM::BI__builtin_arm_mve_vcmpltq_m_s32:
case ARM::BI__builtin_arm_mve_vcmpltq_m_s8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpltq_m_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpltq_m_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpltq_m_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateICmpSLT(Val3, Val4);
  Value *Val6 = Builder.CreateAnd(Val2, Val5);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val6});
  return Builder.CreateIntCast(Val7, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmpcsq_m_u16:
case ARM::BI__builtin_arm_mve_vcmpcsq_m_u32:
case ARM::BI__builtin_arm_mve_vcmpcsq_m_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmpcsq_m_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmpcsq_m_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmpcsq_m_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateICmpUGE(Val3, Val4);
  Value *Val6 = Builder.CreateAnd(Val2, Val5);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val6});
  return Builder.CreateIntCast(Val7, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vcmphiq_m_u16:
case ARM::BI__builtin_arm_mve_vcmphiq_m_u32:
case ARM::BI__builtin_arm_mve_vcmphiq_m_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmphiq_m_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmphiq_m_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmphiq_m_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateICmpUGT(Val3, Val4);
  Value *Val6 = Builder.CreateAnd(Val2, Val5);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_v2i, {Param0}), {Val6});
  return Builder.CreateIntCast(Val7, Int16Ty, false);
}
case ARM::BI__builtin_arm_mve_vpselq_f16:
case ARM::BI__builtin_arm_mve_vpselq_f32:
case ARM::BI__builtin_arm_mve_vpselq_s16:
case ARM::BI__builtin_arm_mve_vpselq_s32:
case ARM::BI__builtin_arm_mve_vpselq_s8:
case ARM::BI__builtin_arm_mve_vpselq_u16:
case ARM::BI__builtin_arm_mve_vpselq_u32:
case ARM::BI__builtin_arm_mve_vpselq_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vpselq_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vpselq_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vpselq_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vpselq_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vpselq_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  case ARM::BI__builtin_arm_mve_vpselq_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vpselq_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vpselq_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateSelect(Val2, Val3, Val4);
}
case ARM::BI__builtin_arm_mve_vdupq_m_n_f16:
case ARM::BI__builtin_arm_mve_vdupq_m_n_f32:
case ARM::BI__builtin_arm_mve_vdupq_m_n_s16:
case ARM::BI__builtin_arm_mve_vdupq_m_n_s32:
case ARM::BI__builtin_arm_mve_vdupq_m_n_s8:
case ARM::BI__builtin_arm_mve_vdupq_m_n_u16:
case ARM::BI__builtin_arm_mve_vdupq_m_n_u32:
case ARM::BI__builtin_arm_mve_vdupq_m_n_u8: {
  llvm::Type * Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vdupq_m_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vdupq_m_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vdupq_m_n_s16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vdupq_m_n_s32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vdupq_m_n_s8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  case ARM::BI__builtin_arm_mve_vdupq_m_n_u16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    break;
  case ARM::BI__builtin_arm_mve_vdupq_m_n_u32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    break;
  case ARM::BI__builtin_arm_mve_vdupq_m_n_u8:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = ARMMVEVectorSplat(Builder, Val3);
  Value *Val5 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateSelect(Val2, Val4, Val5);
}
case ARM::BI__builtin_arm_mve_vbicq_m_n_s32:
case ARM::BI__builtin_arm_mve_vbicq_m_n_u32: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vbicq_m_n_s32:
    Param0 = true;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_m_n_u32:
    Param0 = false;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, Param0);
  Value *Val6 = ARMMVEVectorSplat(Builder, Val5);
  Value *Val7 = Builder.CreateNot(Val6);
  Value *Val8 = Builder.CreateAnd(Val3, Val7);
  return Builder.CreateSelect(Val2, Val8, Val3);
}
case ARM::BI__builtin_arm_mve_vorrq_m_n_s32:
case ARM::BI__builtin_arm_mve_vorrq_m_n_u32: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vorrq_m_n_s32:
    Param0 = true;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_m_n_u32:
    Param0 = false;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, Param0);
  Value *Val6 = ARMMVEVectorSplat(Builder, Val5);
  Value *Val7 = Builder.CreateOr(Val3, Val6);
  return Builder.CreateSelect(Val2, Val7, Val3);
}
case ARM::BI__builtin_arm_mve_vpselq_s64:
case ARM::BI__builtin_arm_mve_vpselq_u64: {
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val4 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val5 = Builder.CreateBitCast(Val3, Val4);
  Value *Val6 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val7 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val8 = Builder.CreateBitCast(Val6, Val7);
  Value *Val9 = Builder.CreateSelect(Val2, Val5, Val8);
  llvm::Type *Val10 = llvm::FixedVectorType::get(Int64Ty, 2);
  return Builder.CreateBitCast(Val9, Val10);
}
case ARM::BI__builtin_arm_mve_vmvnq_m_n_s32:
case ARM::BI__builtin_arm_mve_vmvnq_m_n_u32: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmvnq_m_n_s32:
    Param0 = true;
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_m_n_u32:
    Param0 = false;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, Param0);
  Value *Val5 = ARMMVEVectorSplat(Builder, Val4);
  Value *Val6 = Builder.CreateNot(Val5);
  Value *Val7 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateSelect(Val2, Val6, Val7);
}
case ARM::BI__builtin_arm_mve_vbicq_m_n_s16:
case ARM::BI__builtin_arm_mve_vbicq_m_n_u16: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vbicq_m_n_s16:
    Param0 = true;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_m_n_u16:
    Param0 = false;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateIntCast(Val5, Int16Ty, Param0);
  Value *Val7 = ARMMVEVectorSplat(Builder, Val6);
  Value *Val8 = Builder.CreateNot(Val7);
  Value *Val9 = Builder.CreateAnd(Val3, Val8);
  return Builder.CreateSelect(Val2, Val9, Val3);
}
case ARM::BI__builtin_arm_mve_vorrq_m_n_s16:
case ARM::BI__builtin_arm_mve_vorrq_m_n_u16: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vorrq_m_n_s16:
    Param0 = true;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_m_n_u16:
    Param0 = false;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateIntCast(Val5, Int16Ty, Param0);
  Value *Val7 = ARMMVEVectorSplat(Builder, Val6);
  Value *Val8 = Builder.CreateOr(Val3, Val7);
  return Builder.CreateSelect(Val2, Val8, Val3);
}
case ARM::BI__builtin_arm_mve_vmvnq_m_n_s16:
case ARM::BI__builtin_arm_mve_vmvnq_m_n_u16: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmvnq_m_n_s16:
    Param0 = true;
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_m_n_u16:
    Param0 = false;
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(2));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateIntCast(Val4, Int16Ty, Param0);
  Value *Val6 = ARMMVEVectorSplat(Builder, Val5);
  Value *Val7 = Builder.CreateNot(Val6);
  Value *Val8 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateSelect(Val2, Val7, Val8);
}
case ARM::BI__builtin_arm_mve_vfmasq_m_n_f16:
case ARM::BI__builtin_arm_mve_vfmasq_m_n_f32: {
  llvm::Type * Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vfmasq_m_n_f16:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vfmasq_m_n_f32:
    Param0 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = EmitScalarExpr(E->getArg(3));
  Value *Val1 = Builder.CreateIntCast(Val0, Int32Ty, false);
  Value *Val2 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param0}), {Val1});
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  Value *Val4 = ARMMVEVectorSplat(Builder, Val3);
  Value *Val5 = EmitScalarExpr(E->getArg(0));
  Value *Val6 = EmitScalarExpr(E->getArg(1));
  Value *Val7 = (Val6);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::fma, {Param1}), {Val5, Val7, Val4});
  return Builder.CreateSelect(Val2, Val8, Val5);
}
case ARM::BI__builtin_arm_mve_vcvtaq_m_s16_f16:
case ARM::BI__builtin_arm_mve_vcvtaq_m_s32_f32:
case ARM::BI__builtin_arm_mve_vcvtaq_m_u16_f16:
case ARM::BI__builtin_arm_mve_vcvtaq_m_u32_f32:
case ARM::BI__builtin_arm_mve_vcvtmq_m_s16_f16:
case ARM::BI__builtin_arm_mve_vcvtmq_m_s32_f32:
case ARM::BI__builtin_arm_mve_vcvtmq_m_u16_f16:
case ARM::BI__builtin_arm_mve_vcvtmq_m_u32_f32:
case ARM::BI__builtin_arm_mve_vcvtnq_m_s16_f16:
case ARM::BI__builtin_arm_mve_vcvtnq_m_s32_f32:
case ARM::BI__builtin_arm_mve_vcvtnq_m_u16_f16:
case ARM::BI__builtin_arm_mve_vcvtnq_m_u32_f32:
case ARM::BI__builtin_arm_mve_vcvtpq_m_s16_f16:
case ARM::BI__builtin_arm_mve_vcvtpq_m_s32_f32:
case ARM::BI__builtin_arm_mve_vcvtpq_m_u16_f16:
case ARM::BI__builtin_arm_mve_vcvtpq_m_u32_f32: {
  uint32_t  Param0;
  llvm::Type * Param1;
  Intrinsic::ID  Param2;
  llvm::Type * Param3;
  llvm::Type * Param4;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtaq_m_s16_f16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vcvta_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtaq_m_s32_f32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vcvta_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtaq_m_u16_f16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vcvta_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtaq_m_u32_f32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vcvta_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtmq_m_s16_f16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vcvtm_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtmq_m_s32_f32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vcvtm_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtmq_m_u16_f16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vcvtm_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtmq_m_u32_f32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vcvtm_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtnq_m_s16_f16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vcvtn_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtnq_m_s32_f32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vcvtn_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtnq_m_u16_f16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vcvtn_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtnq_m_u32_f32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vcvtn_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtpq_m_s16_f16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vcvtp_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtpq_m_s32_f32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vcvtp_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtpq_m_u16_f16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vcvtp_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param4 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtpq_m_u32_f32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vcvtp_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param4 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val1 = EmitScalarExpr(E->getArg(0));
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param1}), {Val4});
  return Builder.CreateCall(CGM.getIntrinsic(Param2, {Param3, Param4, Param1}), {Val0, Val1, Val2, Val5});
}
case ARM::BI__builtin_arm_mve_vcvtq_m_n_f16_s16:
case ARM::BI__builtin_arm_mve_vcvtq_m_n_f16_u16:
case ARM::BI__builtin_arm_mve_vcvtq_m_n_f32_s32:
case ARM::BI__builtin_arm_mve_vcvtq_m_n_f32_u32:
case ARM::BI__builtin_arm_mve_vcvtq_m_n_s16_f16:
case ARM::BI__builtin_arm_mve_vcvtq_m_n_s32_f32:
case ARM::BI__builtin_arm_mve_vcvtq_m_n_u16_f16:
case ARM::BI__builtin_arm_mve_vcvtq_m_n_u32_f32: {
  uint32_t  Param0;
  llvm::Type * Param1;
  llvm::Type * Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtq_m_n_f16_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_m_n_f16_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_m_n_f32_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_m_n_f32_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_m_n_s16_f16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_m_n_s32_f32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_m_n_u16_f16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_m_n_u32_f32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val1 = EmitScalarExpr(E->getArg(0));
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  Value *Val4 = EmitScalarExpr(E->getArg(3));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param1}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcvt_fix_predicated, {Param2, Param3, Param1}), {Val0, Val1, Val2, Val3, Val6});
}
case ARM::BI__builtin_arm_mve_vabavq_p_s16:
case ARM::BI__builtin_arm_mve_vabavq_p_s32:
case ARM::BI__builtin_arm_mve_vabavq_p_s8:
case ARM::BI__builtin_arm_mve_vabavq_p_u16:
case ARM::BI__builtin_arm_mve_vabavq_p_u32:
case ARM::BI__builtin_arm_mve_vabavq_p_u8:
case ARM::BI__builtin_arm_mve_vcmlaq_m_f16:
case ARM::BI__builtin_arm_mve_vcmlaq_m_f32:
case ARM::BI__builtin_arm_mve_vcmlaq_rot180_m_f16:
case ARM::BI__builtin_arm_mve_vcmlaq_rot180_m_f32:
case ARM::BI__builtin_arm_mve_vcmlaq_rot270_m_f16:
case ARM::BI__builtin_arm_mve_vcmlaq_rot270_m_f32:
case ARM::BI__builtin_arm_mve_vcmlaq_rot90_m_f16:
case ARM::BI__builtin_arm_mve_vcmlaq_rot90_m_f32:
case ARM::BI__builtin_arm_mve_vcmulq_m_f16:
case ARM::BI__builtin_arm_mve_vcmulq_m_f32:
case ARM::BI__builtin_arm_mve_vcmulq_rot180_m_f16:
case ARM::BI__builtin_arm_mve_vcmulq_rot180_m_f32:
case ARM::BI__builtin_arm_mve_vcmulq_rot270_m_f16:
case ARM::BI__builtin_arm_mve_vcmulq_rot270_m_f32:
case ARM::BI__builtin_arm_mve_vcmulq_rot90_m_f16:
case ARM::BI__builtin_arm_mve_vcmulq_rot90_m_f32: {
  uint32_t  Param0;
  llvm::Type * Param1;
  Intrinsic::ID  Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vabavq_p_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vabav_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vabavq_p_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vabav_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vabavq_p_s8:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_vabav_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vabavq_p_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vabav_predicated;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vabavq_p_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vabav_predicated;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vabavq_p_u8:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param2 = Intrinsic::arm_mve_vabav_predicated;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vcmlaq_m_f16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vcmlaq_predicated;
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmlaq_m_f32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vcmlaq_predicated;
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmlaq_rot180_m_f16:
    Param0 = 2;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vcmlaq_predicated;
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmlaq_rot180_m_f32:
    Param0 = 2;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vcmlaq_predicated;
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmlaq_rot270_m_f16:
    Param0 = 3;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vcmlaq_predicated;
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmlaq_rot270_m_f32:
    Param0 = 3;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vcmlaq_predicated;
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmlaq_rot90_m_f16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vcmlaq_predicated;
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmlaq_rot90_m_f32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vcmlaq_predicated;
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_m_f16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vcmulq_predicated;
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_m_f32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vcmulq_predicated;
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_rot180_m_f16:
    Param0 = 2;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vcmulq_predicated;
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_rot180_m_f32:
    Param0 = 2;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vcmulq_predicated;
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_rot270_m_f16:
    Param0 = 3;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vcmulq_predicated;
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_rot270_m_f32:
    Param0 = 3;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vcmulq_predicated;
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_rot90_m_f16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param2 = Intrinsic::arm_mve_vcmulq_predicated;
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_rot90_m_f32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param2 = Intrinsic::arm_mve_vcmulq_predicated;
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val1 = EmitScalarExpr(E->getArg(0));
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  Value *Val4 = EmitScalarExpr(E->getArg(3));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param1}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Param2, {Param3, Param1}), {Val0, Val1, Val2, Val3, Val6});
}
case ARM::BI__builtin_arm_mve_vabavq_s16:
case ARM::BI__builtin_arm_mve_vabavq_s32:
case ARM::BI__builtin_arm_mve_vabavq_s8:
case ARM::BI__builtin_arm_mve_vabavq_u16:
case ARM::BI__builtin_arm_mve_vabavq_u32:
case ARM::BI__builtin_arm_mve_vabavq_u8:
case ARM::BI__builtin_arm_mve_vcmlaq_f16:
case ARM::BI__builtin_arm_mve_vcmlaq_f32:
case ARM::BI__builtin_arm_mve_vcmlaq_rot180_f16:
case ARM::BI__builtin_arm_mve_vcmlaq_rot180_f32:
case ARM::BI__builtin_arm_mve_vcmlaq_rot270_f16:
case ARM::BI__builtin_arm_mve_vcmlaq_rot270_f32:
case ARM::BI__builtin_arm_mve_vcmlaq_rot90_f16:
case ARM::BI__builtin_arm_mve_vcmlaq_rot90_f32: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vabavq_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vabav;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vabavq_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vabav;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vabavq_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vabav;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vabavq_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vabav;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vabavq_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vabav;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vabavq_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vabav;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vcmlaq_f16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcmlaq;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmlaq_f32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcmlaq;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmlaq_rot180_f16:
    Param0 = 2;
    Param1 = Intrinsic::arm_mve_vcmlaq;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmlaq_rot180_f32:
    Param0 = 2;
    Param1 = Intrinsic::arm_mve_vcmlaq;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmlaq_rot270_f16:
    Param0 = 3;
    Param1 = Intrinsic::arm_mve_vcmlaq;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmlaq_rot270_f32:
    Param0 = 3;
    Param1 = Intrinsic::arm_mve_vcmlaq;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmlaq_rot90_f16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcmlaq;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmlaq_rot90_f32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcmlaq;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val1 = EmitScalarExpr(E->getArg(0));
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = EmitScalarExpr(E->getArg(2));
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2}), {Val0, Val1, Val2, Val3});
}
case ARM::BI__builtin_arm_mve_vcmulq_f16:
case ARM::BI__builtin_arm_mve_vcmulq_f32:
case ARM::BI__builtin_arm_mve_vcmulq_rot180_f16:
case ARM::BI__builtin_arm_mve_vcmulq_rot180_f32:
case ARM::BI__builtin_arm_mve_vcmulq_rot270_f16:
case ARM::BI__builtin_arm_mve_vcmulq_rot270_f32:
case ARM::BI__builtin_arm_mve_vcmulq_rot90_f16:
case ARM::BI__builtin_arm_mve_vcmulq_rot90_f32: {
  uint32_t  Param0;
  llvm::Type * Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmulq_f16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_f32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_rot180_f16:
    Param0 = 2;
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_rot180_f32:
    Param0 = 2;
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_rot270_f16:
    Param0 = 3;
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_rot270_f32:
    Param0 = 3;
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_rot90_f16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_rot90_f32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val1 = EmitScalarExpr(E->getArg(0));
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcmulq, {Param1}), {Val0, Val1, Val2});
}
case ARM::BI__builtin_arm_mve_vcvtq_n_f16_s16:
case ARM::BI__builtin_arm_mve_vcvtq_n_f16_u16:
case ARM::BI__builtin_arm_mve_vcvtq_n_f32_s32:
case ARM::BI__builtin_arm_mve_vcvtq_n_f32_u32:
case ARM::BI__builtin_arm_mve_vcvtq_n_s16_f16:
case ARM::BI__builtin_arm_mve_vcvtq_n_s32_f32:
case ARM::BI__builtin_arm_mve_vcvtq_n_u16_f16:
case ARM::BI__builtin_arm_mve_vcvtq_n_u32_f32: {
  uint32_t  Param0;
  llvm::Type * Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtq_n_f16_s16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_n_f16_u16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(HalfTy, 8);
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_n_f32_s32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_n_f32_u32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(FloatTy, 4);
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_n_s16_f16:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_n_s32_f32:
    Param0 = 0;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_n_u16_f16:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_n_u32_f32:
    Param0 = 1;
    Param1 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val1 = EmitScalarExpr(E->getArg(0));
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcvt_fix, {Param1, Param2}), {Val0, Val1, Val2});
}
case ARM::BI__builtin_arm_mve_vcvtaq_s16_f16:
case ARM::BI__builtin_arm_mve_vcvtaq_s32_f32:
case ARM::BI__builtin_arm_mve_vcvtaq_u16_f16:
case ARM::BI__builtin_arm_mve_vcvtaq_u32_f32:
case ARM::BI__builtin_arm_mve_vcvtmq_s16_f16:
case ARM::BI__builtin_arm_mve_vcvtmq_s32_f32:
case ARM::BI__builtin_arm_mve_vcvtmq_u16_f16:
case ARM::BI__builtin_arm_mve_vcvtmq_u32_f32:
case ARM::BI__builtin_arm_mve_vcvtnq_s16_f16:
case ARM::BI__builtin_arm_mve_vcvtnq_s32_f32:
case ARM::BI__builtin_arm_mve_vcvtnq_u16_f16:
case ARM::BI__builtin_arm_mve_vcvtnq_u32_f32:
case ARM::BI__builtin_arm_mve_vcvtpq_s16_f16:
case ARM::BI__builtin_arm_mve_vcvtpq_s32_f32:
case ARM::BI__builtin_arm_mve_vcvtpq_u16_f16:
case ARM::BI__builtin_arm_mve_vcvtpq_u32_f32: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  llvm::Type * Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtaq_s16_f16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvta;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtaq_s32_f32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvta;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtaq_u16_f16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvta;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtaq_u32_f32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvta;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtmq_s16_f16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvtm;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtmq_s32_f32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvtm;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtmq_u16_f16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvtm;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtmq_u32_f32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvtm;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtnq_s16_f16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvtn;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtnq_s32_f32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvtn;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtnq_u16_f16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvtn;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtnq_u32_f32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvtn;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtpq_s16_f16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvtp;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtpq_s32_f32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvtp;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcvtpq_u16_f16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvtp;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcvtpq_u32_f32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvtp;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  }
  Value *Val0 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val1 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {Param2, Param3}), {Val0, Val1});
}
case ARM::BI__builtin_arm_mve_vcaddq_rot270_m_f16:
case ARM::BI__builtin_arm_mve_vcaddq_rot270_m_f32:
case ARM::BI__builtin_arm_mve_vcaddq_rot270_m_s16:
case ARM::BI__builtin_arm_mve_vcaddq_rot270_m_s32:
case ARM::BI__builtin_arm_mve_vcaddq_rot270_m_s8:
case ARM::BI__builtin_arm_mve_vcaddq_rot270_m_u16:
case ARM::BI__builtin_arm_mve_vcaddq_rot270_m_u32:
case ARM::BI__builtin_arm_mve_vcaddq_rot270_m_u8:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_m_f16:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_m_f32:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_m_s16:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_m_s32:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_m_s8:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_m_u16:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_m_u32:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_m_u8:
case ARM::BI__builtin_arm_mve_vhcaddq_rot270_m_s16:
case ARM::BI__builtin_arm_mve_vhcaddq_rot270_m_s32:
case ARM::BI__builtin_arm_mve_vhcaddq_rot270_m_s8:
case ARM::BI__builtin_arm_mve_vhcaddq_rot90_m_s16:
case ARM::BI__builtin_arm_mve_vhcaddq_rot90_m_s32:
case ARM::BI__builtin_arm_mve_vhcaddq_rot90_m_s8: {
  uint32_t  Param0;
  uint32_t  Param1;
  llvm::Type * Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_m_f16:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_m_f32:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_m_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_m_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_m_s8:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_m_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_m_u32:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_m_u8:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_m_f16:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_m_f32:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_m_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_m_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_m_s8:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_m_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_m_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_m_u8:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot270_m_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot270_m_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot270_m_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot90_m_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot90_m_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot90_m_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val1 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param2}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcaddq_predicated, {Param3, Param2}), {Val0, Val1, Val2, Val3, Val4, Val7});
}
case ARM::BI__builtin_arm_mve_vcaddq_rot270_f16:
case ARM::BI__builtin_arm_mve_vcaddq_rot270_f32:
case ARM::BI__builtin_arm_mve_vcaddq_rot270_s16:
case ARM::BI__builtin_arm_mve_vcaddq_rot270_s32:
case ARM::BI__builtin_arm_mve_vcaddq_rot270_s8:
case ARM::BI__builtin_arm_mve_vcaddq_rot270_u16:
case ARM::BI__builtin_arm_mve_vcaddq_rot270_u32:
case ARM::BI__builtin_arm_mve_vcaddq_rot270_u8:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_f16:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_f32:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_s16:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_s32:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_s8:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_u16:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_u32:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_u8:
case ARM::BI__builtin_arm_mve_vhcaddq_rot270_s16:
case ARM::BI__builtin_arm_mve_vhcaddq_rot270_s32:
case ARM::BI__builtin_arm_mve_vhcaddq_rot270_s8:
case ARM::BI__builtin_arm_mve_vhcaddq_rot90_s16:
case ARM::BI__builtin_arm_mve_vhcaddq_rot90_s32:
case ARM::BI__builtin_arm_mve_vhcaddq_rot90_s8: {
  uint32_t  Param0;
  uint32_t  Param1;
  llvm::Type * Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_f16:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_f32:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_s16:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_s32:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_s8:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_u32:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_u8:
    Param0 = 1;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_f16:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(HalfTy, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_f32:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(FloatTy, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_s16:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_s32:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_s8:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_u8:
    Param0 = 1;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot270_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot270_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot270_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot90_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot90_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot90_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val1 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcaddq, {Param2}), {Val0, Val1, Val2, Val3});
}
case ARM::BI__builtin_arm_mve_vmladavaq_p_s16:
case ARM::BI__builtin_arm_mve_vmladavaq_p_s32:
case ARM::BI__builtin_arm_mve_vmladavaq_p_s8:
case ARM::BI__builtin_arm_mve_vmladavaq_p_u16:
case ARM::BI__builtin_arm_mve_vmladavaq_p_u32:
case ARM::BI__builtin_arm_mve_vmladavaq_p_u8:
case ARM::BI__builtin_arm_mve_vmladavaxq_p_s16:
case ARM::BI__builtin_arm_mve_vmladavaxq_p_s32:
case ARM::BI__builtin_arm_mve_vmladavaxq_p_s8:
case ARM::BI__builtin_arm_mve_vmlsdavaq_p_s16:
case ARM::BI__builtin_arm_mve_vmlsdavaq_p_s32:
case ARM::BI__builtin_arm_mve_vmlsdavaq_p_s8:
case ARM::BI__builtin_arm_mve_vmlsdavaxq_p_s16:
case ARM::BI__builtin_arm_mve_vmlsdavaxq_p_s32:
case ARM::BI__builtin_arm_mve_vmlsdavaxq_p_s8: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  llvm::Type * Param3;
  llvm::Type * Param4;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmladavaq_p_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmladavaq_p_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmladavaq_p_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmladavaq_p_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmladavaq_p_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmladavaq_p_u8:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmladavaxq_p_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmladavaxq_p_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmladavaxq_p_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavaq_p_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavaq_p_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavaq_p_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavaxq_p_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavaxq_p_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavaxq_p_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val1 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param2);
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = EmitScalarExpr(E->getArg(3));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param3}), {Val7});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vmldava_predicated, {Param4, Param3}), {Val0, Val1, Val2, Val3, Val4, Val5, Val8});
}
case ARM::BI__builtin_arm_mve_vmladavaq_s16:
case ARM::BI__builtin_arm_mve_vmladavaq_s32:
case ARM::BI__builtin_arm_mve_vmladavaq_s8:
case ARM::BI__builtin_arm_mve_vmladavaq_u16:
case ARM::BI__builtin_arm_mve_vmladavaq_u32:
case ARM::BI__builtin_arm_mve_vmladavaq_u8:
case ARM::BI__builtin_arm_mve_vmladavaxq_s16:
case ARM::BI__builtin_arm_mve_vmladavaxq_s32:
case ARM::BI__builtin_arm_mve_vmladavaxq_s8:
case ARM::BI__builtin_arm_mve_vmlsdavaq_s16:
case ARM::BI__builtin_arm_mve_vmlsdavaq_s32:
case ARM::BI__builtin_arm_mve_vmlsdavaq_s8:
case ARM::BI__builtin_arm_mve_vmlsdavaxq_s16:
case ARM::BI__builtin_arm_mve_vmlsdavaxq_s32:
case ARM::BI__builtin_arm_mve_vmlsdavaxq_s8: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmladavaq_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmladavaq_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmladavaq_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmladavaq_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmladavaq_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmladavaq_u8:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmladavaxq_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmladavaxq_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmladavaxq_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavaq_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavaq_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavaq_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavaxq_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavaxq_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavaxq_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val1 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param2);
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vmldava, {Param3}), {Val0, Val1, Val2, Val3, Val4, Val5});
}
case ARM::BI__builtin_arm_mve_vmladavq_p_s16:
case ARM::BI__builtin_arm_mve_vmladavq_p_s32:
case ARM::BI__builtin_arm_mve_vmladavq_p_s8:
case ARM::BI__builtin_arm_mve_vmladavq_p_u16:
case ARM::BI__builtin_arm_mve_vmladavq_p_u32:
case ARM::BI__builtin_arm_mve_vmladavq_p_u8:
case ARM::BI__builtin_arm_mve_vmladavxq_p_s16:
case ARM::BI__builtin_arm_mve_vmladavxq_p_s32:
case ARM::BI__builtin_arm_mve_vmladavxq_p_s8:
case ARM::BI__builtin_arm_mve_vmlsdavq_p_s16:
case ARM::BI__builtin_arm_mve_vmlsdavq_p_s32:
case ARM::BI__builtin_arm_mve_vmlsdavq_p_s8:
case ARM::BI__builtin_arm_mve_vmlsdavxq_p_s16:
case ARM::BI__builtin_arm_mve_vmlsdavxq_p_s32:
case ARM::BI__builtin_arm_mve_vmlsdavxq_p_s8: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  llvm::Type * Param3;
  llvm::Type * Param4;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmladavq_p_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmladavq_p_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmladavq_p_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmladavq_p_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmladavq_p_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmladavq_p_u8:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmladavxq_p_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmladavxq_p_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmladavxq_p_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavq_p_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavq_p_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavq_p_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavxq_p_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavxq_p_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavxq_p_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 16);
    Param4 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val1 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param2);
  llvm::Type *Val3 = Int32Ty;
  Value *Val4 = llvm::Constant::getNullValue(Val3);
  Value *Val5 = EmitScalarExpr(E->getArg(0));
  Value *Val6 = EmitScalarExpr(E->getArg(1));
  Value *Val7 = EmitScalarExpr(E->getArg(2));
  Value *Val8 = Builder.CreateIntCast(Val7, Int32Ty, false);
  Value *Val9 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param3}), {Val8});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vmldava_predicated, {Param4, Param3}), {Val0, Val1, Val2, Val4, Val5, Val6, Val9});
}
case ARM::BI__builtin_arm_mve_vmladavq_s16:
case ARM::BI__builtin_arm_mve_vmladavq_s32:
case ARM::BI__builtin_arm_mve_vmladavq_s8:
case ARM::BI__builtin_arm_mve_vmladavq_u16:
case ARM::BI__builtin_arm_mve_vmladavq_u32:
case ARM::BI__builtin_arm_mve_vmladavq_u8:
case ARM::BI__builtin_arm_mve_vmladavxq_s16:
case ARM::BI__builtin_arm_mve_vmladavxq_s32:
case ARM::BI__builtin_arm_mve_vmladavxq_s8:
case ARM::BI__builtin_arm_mve_vmlsdavq_s16:
case ARM::BI__builtin_arm_mve_vmlsdavq_s32:
case ARM::BI__builtin_arm_mve_vmlsdavq_s8:
case ARM::BI__builtin_arm_mve_vmlsdavxq_s16:
case ARM::BI__builtin_arm_mve_vmlsdavxq_s32:
case ARM::BI__builtin_arm_mve_vmlsdavxq_s8: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  llvm::Type * Param3;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmladavq_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmladavq_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmladavq_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmladavq_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmladavq_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmladavq_u8:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmladavxq_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmladavxq_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmladavxq_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavq_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavq_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavq_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavxq_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavxq_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlsdavxq_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Int8Ty, 16);
    break;
  }
  Value *Val0 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val1 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param2);
  llvm::Type *Val3 = Int32Ty;
  Value *Val4 = llvm::Constant::getNullValue(Val3);
  Value *Val5 = EmitScalarExpr(E->getArg(0));
  Value *Val6 = EmitScalarExpr(E->getArg(1));
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vmldava, {Param3}), {Val0, Val1, Val2, Val4, Val5, Val6});
}
case ARM::BI__builtin_arm_mve_vmlaldavq_s16:
case ARM::BI__builtin_arm_mve_vmlaldavq_s32:
case ARM::BI__builtin_arm_mve_vmlaldavq_u16:
case ARM::BI__builtin_arm_mve_vmlaldavq_u32:
case ARM::BI__builtin_arm_mve_vmlaldavxq_s16:
case ARM::BI__builtin_arm_mve_vmlaldavxq_s32:
case ARM::BI__builtin_arm_mve_vmlsldavq_s16:
case ARM::BI__builtin_arm_mve_vmlsldavq_s32:
case ARM::BI__builtin_arm_mve_vmlsldavxq_s16:
case ARM::BI__builtin_arm_mve_vmlsldavxq_s32:
case ARM::BI__builtin_arm_mve_vrmlaldavhq_s32:
case ARM::BI__builtin_arm_mve_vrmlaldavhq_u32:
case ARM::BI__builtin_arm_mve_vrmlaldavhxq_s32:
case ARM::BI__builtin_arm_mve_vrmlsldavhq_s32:
case ARM::BI__builtin_arm_mve_vrmlsldavhxq_s32: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  Intrinsic::ID  Param3;
  llvm::Type * Param4;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmlaldavq_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavq_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavq_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavq_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavxq_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavxq_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlsldavq_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlsldavq_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlsldavxq_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlsldavxq_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vmlldava;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlaldavhq_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vrmlldavha;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlaldavhq_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vrmlldavha;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlaldavhxq_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vrmlldavha;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlsldavhq_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = Intrinsic::arm_mve_vrmlldavha;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlsldavhxq_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = Intrinsic::arm_mve_vrmlldavha;
    Param4 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  llvm::Type *Val0 = Int32Ty;
  Value *Val1 = llvm::Constant::getNullValue(Val0);
  llvm::Type *Val2 = Int32Ty;
  Value *Val3 = llvm::Constant::getNullValue(Val2);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val6 = llvm::ConstantInt::get(Int32Ty, Param2);
  Value *Val7 = EmitScalarExpr(E->getArg(0));
  Value *Val8 = EmitScalarExpr(E->getArg(1));
  Value *Val9 = Builder.CreateCall(CGM.getIntrinsic(Param3, {Param4}), {Val4, Val5, Val6, Val3, Val1, Val7, Val8});
  Value *Val10 = Builder.CreateExtractValue(Val9, static_cast<unsigned>(1));
  Value *Val11 = Builder.CreateIntCast(Val10, Int64Ty, false);
  Value *Val12 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val13 = Builder.CreateShl(Val11, Val12);
  Value *Val14 = Builder.CreateExtractValue(Val9, static_cast<unsigned>(0));
  Value *Val15 = Builder.CreateIntCast(Val14, Int64Ty, false);
  return Builder.CreateOr(Val13, Val15);
}
case ARM::BI__builtin_arm_mve_vmlaldavq_p_s16:
case ARM::BI__builtin_arm_mve_vmlaldavq_p_s32:
case ARM::BI__builtin_arm_mve_vmlaldavq_p_u16:
case ARM::BI__builtin_arm_mve_vmlaldavq_p_u32:
case ARM::BI__builtin_arm_mve_vmlaldavxq_p_s16:
case ARM::BI__builtin_arm_mve_vmlaldavxq_p_s32:
case ARM::BI__builtin_arm_mve_vmlsldavq_p_s16:
case ARM::BI__builtin_arm_mve_vmlsldavq_p_s32:
case ARM::BI__builtin_arm_mve_vmlsldavxq_p_s16:
case ARM::BI__builtin_arm_mve_vmlsldavxq_p_s32:
case ARM::BI__builtin_arm_mve_vrmlaldavhq_p_s32:
case ARM::BI__builtin_arm_mve_vrmlaldavhq_p_u32:
case ARM::BI__builtin_arm_mve_vrmlaldavhxq_p_s32:
case ARM::BI__builtin_arm_mve_vrmlsldavhq_p_s32:
case ARM::BI__builtin_arm_mve_vrmlsldavhxq_p_s32: {
  uint32_t  Param0;
  uint32_t  Param1;
  uint32_t  Param2;
  llvm::Type * Param3;
  Intrinsic::ID  Param4;
  llvm::Type * Param5;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmlaldavq_p_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavq_p_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavq_p_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavq_p_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavxq_p_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlaldavxq_p_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlsldavq_p_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlsldavq_p_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vmlsldavxq_p_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 8);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int16Ty, 8);
    break;
  case ARM::BI__builtin_arm_mve_vmlsldavxq_p_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vmlldava_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlaldavhq_p_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vrmlldavha_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlaldavhq_p_u32:
    Param0 = 1;
    Param1 = 0;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vrmlldavha_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlaldavhxq_p_s32:
    Param0 = 0;
    Param1 = 0;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vrmlldavha_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlsldavhq_p_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 0;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vrmlldavha_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  case ARM::BI__builtin_arm_mve_vrmlsldavhxq_p_s32:
    Param0 = 0;
    Param1 = 1;
    Param2 = 1;
    Param3 = llvm::FixedVectorType::get(Builder.getInt1Ty(), 4);
    Param4 = Intrinsic::arm_mve_vrmlldavha_predicated;
    Param5 = llvm::FixedVectorType::get(Int32Ty, 4);
    break;
  }
  llvm::Type *Val0 = Int32Ty;
  Value *Val1 = llvm::Constant::getNullValue(Val0);
  llvm::Type *Val2 = Int32Ty;
  Value *Val3 = llvm::Constant::getNullValue(Val2);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val6 = llvm::ConstantInt::get(Int32Ty, Param2);
  Value *Val7 = EmitScalarExpr(E->getArg(0));
  Value *Val8 = EmitScalarExpr(E->getArg(1));
  Value *Val9 = EmitScalarExpr(E->getArg(2));
  Value *Val10 = Builder.CreateIntCast(Val9, Int32Ty, false);
  Value *Val11 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {Param3}), {Val10});
  Value *Val12 = Builder.CreateCall(CGM.getIntrinsic(Param4, {Param5, Param3}), {Val4, Val5, Val6, Val3, Val1, Val7, Val8, Val11});
  Value *Val13 = Builder.CreateExtractValue(Val12, static_cast<unsigned>(1));
  Value *Val14 = Builder.CreateIntCast(Val13, Int64Ty, false);
  Value *Val15 = llvm::ConstantInt::get(Int64Ty, 32);
  Value *Val16 = Builder.CreateShl(Val14, Val15);
  Value *Val17 = Builder.CreateExtractValue(Val12, static_cast<unsigned>(0));
  Value *Val18 = Builder.CreateIntCast(Val17, Int64Ty, false);
  return Builder.CreateOr(Val16, Val18);
}
case ARM::BI__builtin_arm_mve_vaddq_x_n_f32:
case ARM::BI__builtin_arm_mve_vmulq_x_n_f32:
case ARM::BI__builtin_arm_mve_vsubq_x_n_f32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vaddq_x_n_f32:
    Param0 = Intrinsic::arm_mve_add_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulq_x_n_f32:
    Param0 = Intrinsic::arm_mve_mul_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsubq_x_n_f32:
    Param0 = Intrinsic::arm_mve_sub_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(FloatTy, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = ARMMVEVectorSplat(Builder, Val3);
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(FloatTy, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val4, Val7, Val1});
}
case ARM::BI__builtin_arm_mve_vabsq_x_f32:
case ARM::BI__builtin_arm_mve_vnegq_x_f32:
case ARM::BI__builtin_arm_mve_vrndaq_x_f32:
case ARM::BI__builtin_arm_mve_vrndmq_x_f32:
case ARM::BI__builtin_arm_mve_vrndnq_x_f32:
case ARM::BI__builtin_arm_mve_vrndpq_x_f32:
case ARM::BI__builtin_arm_mve_vrndq_x_f32:
case ARM::BI__builtin_arm_mve_vrndxq_x_f32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vabsq_x_f32:
    Param0 = Intrinsic::arm_mve_abs_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vnegq_x_f32:
    Param0 = Intrinsic::arm_mve_neg_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrndaq_x_f32:
    Param0 = Intrinsic::arm_mve_vrinta_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrndmq_x_f32:
    Param0 = Intrinsic::arm_mve_vrintm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrndnq_x_f32:
    Param0 = Intrinsic::arm_mve_vrintn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrndpq_x_f32:
    Param0 = Intrinsic::arm_mve_vrintp_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrndq_x_f32:
    Param0 = Intrinsic::arm_mve_vrintz_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrndxq_x_f32:
    Param0 = Intrinsic::arm_mve_vrintx_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(FloatTy, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val4});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(FloatTy, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val5, Val1});
}
case ARM::BI__builtin_arm_mve_vbrsrq_x_n_f32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vbrsrq_x_n_f32:
    Param0 = Intrinsic::arm_mve_vbrsr_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(FloatTy, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(FloatTy, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val1, Val2, Val3, Val6});
}
case ARM::BI__builtin_arm_mve_vaddq_x_f32:
case ARM::BI__builtin_arm_mve_vmulq_x_f32:
case ARM::BI__builtin_arm_mve_vsubq_x_f32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vaddq_x_f32:
    Param0 = Intrinsic::arm_mve_add_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulq_x_f32:
    Param0 = Intrinsic::arm_mve_mul_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsubq_x_f32:
    Param0 = Intrinsic::arm_mve_sub_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(FloatTy, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(FloatTy, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val3, Val6, Val1});
}
case ARM::BI__builtin_arm_mve_vabdq_x_f32:
case ARM::BI__builtin_arm_mve_vmaxnmq_x_f32:
case ARM::BI__builtin_arm_mve_vminnmq_x_f32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vabdq_x_f32:
    Param0 = Intrinsic::arm_mve_abd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmaxnmq_x_f32:
    Param0 = Intrinsic::arm_mve_max_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vminnmq_x_f32:
    Param0 = Intrinsic::arm_mve_min_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(FloatTy, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, 0);
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(FloatTy, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val3, Val4, Val7, Val1});
}
case ARM::BI__builtin_arm_mve_vrev64q_x_f32: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(FloatTy, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, 64);
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vrev_predicated, {llvm::FixedVectorType::get(FloatTy, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val3, Val6, Val1});
}
case ARM::BI__builtin_arm_mve_vcvtq_x_f32_s32:
case ARM::BI__builtin_arm_mve_vcvtq_x_f32_u32: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtq_x_f32_s32:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_x_f32_u32:
    Param0 = 1;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(FloatTy, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcvt_fp_int_predicated, {llvm::FixedVectorType::get(FloatTy, 4), llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val3, Val6, Val1});
}
case ARM::BI__builtin_arm_mve_vcvtbq_x_f32_f16:
case ARM::BI__builtin_arm_mve_vcvttq_x_f32_f16: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtbq_x_f32_f16:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vcvttq_x_f32_f16:
    Param0 = 1;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(FloatTy, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcvt_widen_predicated), {Val1, Val2, Val3, Val6});
}
case ARM::BI__builtin_arm_mve_vandq_x_f32:
case ARM::BI__builtin_arm_mve_vbicq_x_f32:
case ARM::BI__builtin_arm_mve_veorq_x_f32:
case ARM::BI__builtin_arm_mve_vornq_x_f32:
case ARM::BI__builtin_arm_mve_vorrq_x_f32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vandq_x_f32:
    Param0 = Intrinsic::arm_mve_and_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_x_f32:
    Param0 = Intrinsic::arm_mve_bic_predicated;
    break;
  case ARM::BI__builtin_arm_mve_veorq_x_f32:
    Param0 = Intrinsic::arm_mve_eor_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vornq_x_f32:
    Param0 = Intrinsic::arm_mve_orn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_x_f32:
    Param0 = Intrinsic::arm_mve_orr_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(FloatTy, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val3 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val4 = Builder.CreateBitCast(Val2, Val3);
  Value *Val5 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val7 = Builder.CreateBitCast(Val5, Val6);
  Value *Val8 = EmitScalarExpr(E->getArg(2));
  Value *Val9 = Builder.CreateIntCast(Val8, Int32Ty, false);
  Value *Val10 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val9});
  llvm::Type *Val11 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val12 = Builder.CreateBitCast(Val1, Val11);
  Value *Val13 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val4, Val7, Val10, Val12});
  llvm::Type *Val14 = llvm::FixedVectorType::get(FloatTy, 4);
  return Builder.CreateBitCast(Val13, Val14);
}
case ARM::BI__builtin_arm_mve_vdupq_x_n_f32: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(FloatTy, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val3});
  Value *Val5 = EmitScalarExpr(E->getArg(0));
  Value *Val6 = ARMMVEVectorSplat(Builder, Val5);
  return Builder.CreateSelect(Val4, Val6, Val1);
}
case ARM::BI__builtin_arm_mve_vcaddq_rot270_x_f32:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_x_f32: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_x_f32:
    Param0 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_x_f32:
    Param0 = 0;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(FloatTy, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, 1);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = EmitScalarExpr(E->getArg(0));
  Value *Val5 = EmitScalarExpr(E->getArg(1));
  Value *Val6 = EmitScalarExpr(E->getArg(2));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val7});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcaddq_predicated, {llvm::FixedVectorType::get(FloatTy, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val3, Val1, Val4, Val5, Val8});
}
case ARM::BI__builtin_arm_mve_vcmulq_rot180_x_f32:
case ARM::BI__builtin_arm_mve_vcmulq_rot270_x_f32:
case ARM::BI__builtin_arm_mve_vcmulq_rot90_x_f32:
case ARM::BI__builtin_arm_mve_vcmulq_x_f32: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmulq_rot180_x_f32:
    Param0 = 2;
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_rot270_x_f32:
    Param0 = 3;
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_rot90_x_f32:
    Param0 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_x_f32:
    Param0 = 0;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(FloatTy, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcmulq_predicated, {llvm::FixedVectorType::get(FloatTy, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val1, Val3, Val4, Val7});
}
case ARM::BI__builtin_arm_mve_vcvtq_x_n_f32_s32:
case ARM::BI__builtin_arm_mve_vcvtq_x_n_f32_u32: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtq_x_n_f32_s32:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_x_n_f32_u32:
    Param0 = 1;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(FloatTy, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcvt_fix_predicated, {llvm::FixedVectorType::get(FloatTy, 4), llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val1, Val3, Val4, Val7});
}
case ARM::BI__builtin_arm_mve_vuninitializedq_f32:
case ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_f32: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(FloatTy, 4);
  return UndefValue::get(Val0);
}
case ARM::BI__builtin_arm_mve_vaddq_x_n_f16:
case ARM::BI__builtin_arm_mve_vmulq_x_n_f16:
case ARM::BI__builtin_arm_mve_vsubq_x_n_f16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vaddq_x_n_f16:
    Param0 = Intrinsic::arm_mve_add_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulq_x_n_f16:
    Param0 = Intrinsic::arm_mve_mul_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsubq_x_n_f16:
    Param0 = Intrinsic::arm_mve_sub_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(HalfTy, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = ARMMVEVectorSplat(Builder, Val3);
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(HalfTy, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val4, Val7, Val1});
}
case ARM::BI__builtin_arm_mve_vabsq_x_f16:
case ARM::BI__builtin_arm_mve_vnegq_x_f16:
case ARM::BI__builtin_arm_mve_vrndaq_x_f16:
case ARM::BI__builtin_arm_mve_vrndmq_x_f16:
case ARM::BI__builtin_arm_mve_vrndnq_x_f16:
case ARM::BI__builtin_arm_mve_vrndpq_x_f16:
case ARM::BI__builtin_arm_mve_vrndq_x_f16:
case ARM::BI__builtin_arm_mve_vrndxq_x_f16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vabsq_x_f16:
    Param0 = Intrinsic::arm_mve_abs_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vnegq_x_f16:
    Param0 = Intrinsic::arm_mve_neg_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrndaq_x_f16:
    Param0 = Intrinsic::arm_mve_vrinta_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrndmq_x_f16:
    Param0 = Intrinsic::arm_mve_vrintm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrndnq_x_f16:
    Param0 = Intrinsic::arm_mve_vrintn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrndpq_x_f16:
    Param0 = Intrinsic::arm_mve_vrintp_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrndq_x_f16:
    Param0 = Intrinsic::arm_mve_vrintz_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrndxq_x_f16:
    Param0 = Intrinsic::arm_mve_vrintx_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(HalfTy, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val4});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(HalfTy, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val5, Val1});
}
case ARM::BI__builtin_arm_mve_vbrsrq_x_n_f16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vbrsrq_x_n_f16:
    Param0 = Intrinsic::arm_mve_vbrsr_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(HalfTy, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(HalfTy, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val1, Val2, Val3, Val6});
}
case ARM::BI__builtin_arm_mve_vaddq_x_f16:
case ARM::BI__builtin_arm_mve_vmulq_x_f16:
case ARM::BI__builtin_arm_mve_vsubq_x_f16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vaddq_x_f16:
    Param0 = Intrinsic::arm_mve_add_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulq_x_f16:
    Param0 = Intrinsic::arm_mve_mul_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsubq_x_f16:
    Param0 = Intrinsic::arm_mve_sub_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(HalfTy, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(HalfTy, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val3, Val6, Val1});
}
case ARM::BI__builtin_arm_mve_vabdq_x_f16:
case ARM::BI__builtin_arm_mve_vmaxnmq_x_f16:
case ARM::BI__builtin_arm_mve_vminnmq_x_f16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vabdq_x_f16:
    Param0 = Intrinsic::arm_mve_abd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmaxnmq_x_f16:
    Param0 = Intrinsic::arm_mve_max_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vminnmq_x_f16:
    Param0 = Intrinsic::arm_mve_min_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(HalfTy, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, 0);
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(HalfTy, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val3, Val4, Val7, Val1});
}
case ARM::BI__builtin_arm_mve_vcvtq_x_f16_s16:
case ARM::BI__builtin_arm_mve_vcvtq_x_f16_u16: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtq_x_f16_s16:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_x_f16_u16:
    Param0 = 1;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(HalfTy, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcvt_fp_int_predicated, {llvm::FixedVectorType::get(HalfTy, 8), llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val3, Val6, Val1});
}
case ARM::BI__builtin_arm_mve_vrev32q_x_f16:
case ARM::BI__builtin_arm_mve_vrev64q_x_f16: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vrev32q_x_f16:
    Param0 = 32;
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_x_f16:
    Param0 = 64;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(HalfTy, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vrev_predicated, {llvm::FixedVectorType::get(HalfTy, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val3, Val6, Val1});
}
case ARM::BI__builtin_arm_mve_vandq_x_f16:
case ARM::BI__builtin_arm_mve_vbicq_x_f16:
case ARM::BI__builtin_arm_mve_veorq_x_f16:
case ARM::BI__builtin_arm_mve_vornq_x_f16:
case ARM::BI__builtin_arm_mve_vorrq_x_f16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vandq_x_f16:
    Param0 = Intrinsic::arm_mve_and_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_x_f16:
    Param0 = Intrinsic::arm_mve_bic_predicated;
    break;
  case ARM::BI__builtin_arm_mve_veorq_x_f16:
    Param0 = Intrinsic::arm_mve_eor_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vornq_x_f16:
    Param0 = Intrinsic::arm_mve_orn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_x_f16:
    Param0 = Intrinsic::arm_mve_orr_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(HalfTy, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val3 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val4 = Builder.CreateBitCast(Val2, Val3);
  Value *Val5 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val7 = Builder.CreateBitCast(Val5, Val6);
  Value *Val8 = EmitScalarExpr(E->getArg(2));
  Value *Val9 = Builder.CreateIntCast(Val8, Int32Ty, false);
  Value *Val10 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val9});
  llvm::Type *Val11 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val12 = Builder.CreateBitCast(Val1, Val11);
  Value *Val13 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val4, Val7, Val10, Val12});
  llvm::Type *Val14 = llvm::FixedVectorType::get(HalfTy, 8);
  return Builder.CreateBitCast(Val13, Val14);
}
case ARM::BI__builtin_arm_mve_vdupq_x_n_f16: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(HalfTy, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val3});
  Value *Val5 = EmitScalarExpr(E->getArg(0));
  Value *Val6 = ARMMVEVectorSplat(Builder, Val5);
  return Builder.CreateSelect(Val4, Val6, Val1);
}
case ARM::BI__builtin_arm_mve_vcaddq_rot270_x_f16:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_x_f16: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_x_f16:
    Param0 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_x_f16:
    Param0 = 0;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(HalfTy, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, 1);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = EmitScalarExpr(E->getArg(0));
  Value *Val5 = EmitScalarExpr(E->getArg(1));
  Value *Val6 = EmitScalarExpr(E->getArg(2));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val7});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcaddq_predicated, {llvm::FixedVectorType::get(HalfTy, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val3, Val1, Val4, Val5, Val8});
}
case ARM::BI__builtin_arm_mve_vcmulq_rot180_x_f16:
case ARM::BI__builtin_arm_mve_vcmulq_rot270_x_f16:
case ARM::BI__builtin_arm_mve_vcmulq_rot90_x_f16:
case ARM::BI__builtin_arm_mve_vcmulq_x_f16: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcmulq_rot180_x_f16:
    Param0 = 2;
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_rot270_x_f16:
    Param0 = 3;
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_rot90_x_f16:
    Param0 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vcmulq_x_f16:
    Param0 = 0;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(HalfTy, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcmulq_predicated, {llvm::FixedVectorType::get(HalfTy, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val1, Val3, Val4, Val7});
}
case ARM::BI__builtin_arm_mve_vcvtq_x_n_f16_s16:
case ARM::BI__builtin_arm_mve_vcvtq_x_n_f16_u16: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtq_x_n_f16_s16:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_x_n_f16_u16:
    Param0 = 1;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(HalfTy, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcvt_fix_predicated, {llvm::FixedVectorType::get(HalfTy, 8), llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val1, Val3, Val4, Val7});
}
case ARM::BI__builtin_arm_mve_vuninitializedq_f16:
case ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_f16: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(HalfTy, 8);
  return UndefValue::get(Val0);
}
case ARM::BI__builtin_arm_mve_vddupq_x_wb_u16:
case ARM::BI__builtin_arm_mve_vidupq_x_wb_u16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vddupq_x_wb_u16:
    Param0 = Intrinsic::arm_mve_vddup_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vidupq_x_wb_u16:
    Param0 = Intrinsic::arm_mve_vidup_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Address Val2 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val3 = Builder.CreateLoad(Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val6});
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val1, Val3, Val4, Val7});
  Value *Val9 = Builder.CreateExtractValue(Val8, static_cast<unsigned>(1));
  Builder.CreateStore(Val9, Val2);
  return Builder.CreateExtractValue(Val8, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vdwdupq_x_wb_u16:
case ARM::BI__builtin_arm_mve_viwdupq_x_wb_u16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vdwdupq_x_wb_u16:
    Param0 = Intrinsic::arm_mve_vdwdup_predicated;
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_x_wb_u16:
    Param0 = Intrinsic::arm_mve_viwdup_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Address Val2 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val3 = Builder.CreateLoad(Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = EmitScalarExpr(E->getArg(3));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val7});
  Value *Val9 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val1, Val3, Val4, Val5, Val8});
  Value *Val10 = Builder.CreateExtractValue(Val9, static_cast<unsigned>(1));
  Builder.CreateStore(Val10, Val2);
  return Builder.CreateExtractValue(Val9, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vaddq_x_n_s16:
case ARM::BI__builtin_arm_mve_vaddq_x_n_u16:
case ARM::BI__builtin_arm_mve_vmulq_x_n_s16:
case ARM::BI__builtin_arm_mve_vmulq_x_n_u16:
case ARM::BI__builtin_arm_mve_vsubq_x_n_s16:
case ARM::BI__builtin_arm_mve_vsubq_x_n_u16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vaddq_x_n_s16:
    Param0 = Intrinsic::arm_mve_add_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vaddq_x_n_u16:
    Param0 = Intrinsic::arm_mve_add_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulq_x_n_s16:
    Param0 = Intrinsic::arm_mve_mul_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulq_x_n_u16:
    Param0 = Intrinsic::arm_mve_mul_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsubq_x_n_s16:
    Param0 = Intrinsic::arm_mve_sub_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsubq_x_n_u16:
    Param0 = Intrinsic::arm_mve_sub_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = ARMMVEVectorSplat(Builder, Val3);
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val4, Val7, Val1});
}
case ARM::BI__builtin_arm_mve_vhaddq_x_n_s16:
case ARM::BI__builtin_arm_mve_vhaddq_x_n_u16:
case ARM::BI__builtin_arm_mve_vhsubq_x_n_s16:
case ARM::BI__builtin_arm_mve_vhsubq_x_n_u16: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vhaddq_x_n_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_hadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_x_n_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_hadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_x_n_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_hsub_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_x_n_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_hsub_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = ARMMVEVectorSplat(Builder, Val3);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val6 = EmitScalarExpr(E->getArg(2));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val7});
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val4, Val5, Val8, Val1});
}
case ARM::BI__builtin_arm_mve_vabsq_x_s16:
case ARM::BI__builtin_arm_mve_vclsq_x_s16:
case ARM::BI__builtin_arm_mve_vclzq_x_s16:
case ARM::BI__builtin_arm_mve_vclzq_x_u16:
case ARM::BI__builtin_arm_mve_vmvnq_x_s16:
case ARM::BI__builtin_arm_mve_vmvnq_x_u16:
case ARM::BI__builtin_arm_mve_vnegq_x_s16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vabsq_x_s16:
    Param0 = Intrinsic::arm_mve_abs_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vclsq_x_s16:
    Param0 = Intrinsic::arm_mve_cls_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vclzq_x_s16:
    Param0 = Intrinsic::arm_mve_clz_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vclzq_x_u16:
    Param0 = Intrinsic::arm_mve_clz_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_x_s16:
    Param0 = Intrinsic::arm_mve_mvn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_x_u16:
    Param0 = Intrinsic::arm_mve_mvn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vnegq_x_s16:
    Param0 = Intrinsic::arm_mve_neg_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val4});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val5, Val1});
}
case ARM::BI__builtin_arm_mve_vddupq_x_n_u16:
case ARM::BI__builtin_arm_mve_vidupq_x_n_u16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vddupq_x_n_u16:
    Param0 = Intrinsic::arm_mve_vddup_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vidupq_x_n_u16:
    Param0 = Intrinsic::arm_mve_vidup_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val5});
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val1, Val2, Val3, Val6});
  return Builder.CreateExtractValue(Val7, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vbrsrq_x_n_s16:
case ARM::BI__builtin_arm_mve_vbrsrq_x_n_u16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vbrsrq_x_n_s16:
    Param0 = Intrinsic::arm_mve_vbrsr_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbrsrq_x_n_u16:
    Param0 = Intrinsic::arm_mve_vbrsr_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val1, Val2, Val3, Val6});
}
case ARM::BI__builtin_arm_mve_vaddq_x_s16:
case ARM::BI__builtin_arm_mve_vaddq_x_u16:
case ARM::BI__builtin_arm_mve_vmulq_x_s16:
case ARM::BI__builtin_arm_mve_vmulq_x_u16:
case ARM::BI__builtin_arm_mve_vshlq_x_n_s16:
case ARM::BI__builtin_arm_mve_vshlq_x_n_u16:
case ARM::BI__builtin_arm_mve_vsubq_x_s16:
case ARM::BI__builtin_arm_mve_vsubq_x_u16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vaddq_x_s16:
    Param0 = Intrinsic::arm_mve_add_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vaddq_x_u16:
    Param0 = Intrinsic::arm_mve_add_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulq_x_s16:
    Param0 = Intrinsic::arm_mve_mul_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulq_x_u16:
    Param0 = Intrinsic::arm_mve_mul_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshlq_x_n_s16:
    Param0 = Intrinsic::arm_mve_shl_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshlq_x_n_u16:
    Param0 = Intrinsic::arm_mve_shl_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsubq_x_s16:
    Param0 = Intrinsic::arm_mve_sub_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsubq_x_u16:
    Param0 = Intrinsic::arm_mve_sub_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val3, Val6, Val1});
}
case ARM::BI__builtin_arm_mve_vdwdupq_x_n_u16:
case ARM::BI__builtin_arm_mve_viwdupq_x_n_u16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vdwdupq_x_n_u16:
    Param0 = Intrinsic::arm_mve_vdwdup_predicated;
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_x_n_u16:
    Param0 = Intrinsic::arm_mve_viwdup_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val6});
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val1, Val2, Val3, Val4, Val7});
  return Builder.CreateExtractValue(Val8, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vrshlq_x_s16:
case ARM::BI__builtin_arm_mve_vrshlq_x_u16:
case ARM::BI__builtin_arm_mve_vshlq_x_s16:
case ARM::BI__builtin_arm_mve_vshlq_x_u16: {
  uint32_t  Param0;
  uint32_t  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vrshlq_x_s16:
    Param0 = 1;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_x_u16:
    Param0 = 1;
    Param1 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vshlq_x_s16:
    Param0 = 0;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vshlq_x_u16:
    Param0 = 0;
    Param1 = 1;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, 0);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val6 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val7 = EmitScalarExpr(E->getArg(2));
  Value *Val8 = Builder.CreateIntCast(Val7, Int32Ty, false);
  Value *Val9 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val8});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vshl_vector_predicated, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val3, Val4, Val5, Val6, Val9, Val1});
}
case ARM::BI__builtin_arm_mve_vmullbq_poly_x_p8:
case ARM::BI__builtin_arm_mve_vmulltq_poly_x_p8: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmullbq_poly_x_p8:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_poly_x_p8:
    Param0 = 1;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_mull_poly_predicated, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val3, Val4, Val7, Val1});
}
case ARM::BI__builtin_arm_mve_vabdq_x_s16:
case ARM::BI__builtin_arm_mve_vabdq_x_u16:
case ARM::BI__builtin_arm_mve_vhaddq_x_s16:
case ARM::BI__builtin_arm_mve_vhaddq_x_u16:
case ARM::BI__builtin_arm_mve_vhsubq_x_s16:
case ARM::BI__builtin_arm_mve_vhsubq_x_u16:
case ARM::BI__builtin_arm_mve_vmaxq_x_s16:
case ARM::BI__builtin_arm_mve_vmaxq_x_u16:
case ARM::BI__builtin_arm_mve_vminq_x_s16:
case ARM::BI__builtin_arm_mve_vminq_x_u16:
case ARM::BI__builtin_arm_mve_vmulhq_x_s16:
case ARM::BI__builtin_arm_mve_vmulhq_x_u16:
case ARM::BI__builtin_arm_mve_vrhaddq_x_s16:
case ARM::BI__builtin_arm_mve_vrhaddq_x_u16:
case ARM::BI__builtin_arm_mve_vrmulhq_x_s16:
case ARM::BI__builtin_arm_mve_vrmulhq_x_u16:
case ARM::BI__builtin_arm_mve_vrshrq_x_n_s16:
case ARM::BI__builtin_arm_mve_vrshrq_x_n_u16:
case ARM::BI__builtin_arm_mve_vshrq_x_n_s16:
case ARM::BI__builtin_arm_mve_vshrq_x_n_u16: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vabdq_x_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_abd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vabdq_x_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_abd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_x_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_hadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_x_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_hadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_x_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_hsub_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_x_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_hsub_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmaxq_x_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_max_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmaxq_x_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_max_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vminq_x_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_min_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vminq_x_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_min_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_x_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_mulh_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_x_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_mulh_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_x_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_rhadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_x_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_rhadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_x_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_rmulh_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_x_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_rmulh_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_x_n_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vrshr_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_x_n_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vrshr_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshrq_x_n_s16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_shr_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshrq_x_n_u16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_shr_imm_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val3, Val4, Val7, Val1});
}
case ARM::BI__builtin_arm_mve_vmullbq_int_x_s8:
case ARM::BI__builtin_arm_mve_vmullbq_int_x_u8:
case ARM::BI__builtin_arm_mve_vmulltq_int_x_s8:
case ARM::BI__builtin_arm_mve_vmulltq_int_x_u8:
case ARM::BI__builtin_arm_mve_vshllbq_x_n_s8:
case ARM::BI__builtin_arm_mve_vshllbq_x_n_u8:
case ARM::BI__builtin_arm_mve_vshlltq_x_n_s8:
case ARM::BI__builtin_arm_mve_vshlltq_x_n_u8: {
  uint32_t  Param0;
  uint32_t  Param1;
  Intrinsic::ID  Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmullbq_int_x_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_mull_int_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmullbq_int_x_u8:
    Param0 = 1;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_mull_int_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_x_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_mull_int_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_x_u8:
    Param0 = 1;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_mull_int_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshllbq_x_n_s8:
    Param0 = 0;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_vshll_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshllbq_x_n_u8:
    Param0 = 1;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_vshll_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshlltq_x_n_s8:
    Param0 = 0;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_vshll_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshlltq_x_n_u8:
    Param0 = 1;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_vshll_imm_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val6 = EmitScalarExpr(E->getArg(2));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val7});
  return Builder.CreateCall(CGM.getIntrinsic(Param2, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val3, Val4, Val5, Val8, Val1});
}
case ARM::BI__builtin_arm_mve_vcvtq_x_s16_f16:
case ARM::BI__builtin_arm_mve_vcvtq_x_u16_f16: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtq_x_s16_f16:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_x_u16_f16:
    Param0 = 1;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcvt_fp_int_predicated, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(HalfTy, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val3, Val6, Val1});
}
case ARM::BI__builtin_arm_mve_vrev32q_x_s16:
case ARM::BI__builtin_arm_mve_vrev32q_x_u16:
case ARM::BI__builtin_arm_mve_vrev64q_x_s16:
case ARM::BI__builtin_arm_mve_vrev64q_x_u16: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vrev32q_x_s16:
    Param0 = 32;
    break;
  case ARM::BI__builtin_arm_mve_vrev32q_x_u16:
    Param0 = 32;
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_x_s16:
    Param0 = 64;
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_x_u16:
    Param0 = 64;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vrev_predicated, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val3, Val6, Val1});
}
case ARM::BI__builtin_arm_mve_vmovlbq_x_s8:
case ARM::BI__builtin_arm_mve_vmovlbq_x_u8:
case ARM::BI__builtin_arm_mve_vmovltq_x_s8:
case ARM::BI__builtin_arm_mve_vmovltq_x_u8: {
  uint32_t  Param0;
  uint32_t  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmovlbq_x_s8:
    Param0 = 0;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vmovlbq_x_u8:
    Param0 = 1;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vmovltq_x_s8:
    Param0 = 0;
    Param1 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vmovltq_x_u8:
    Param0 = 1;
    Param1 = 1;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val5 = EmitScalarExpr(E->getArg(1));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vmovl_predicated, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val3, Val4, Val7, Val1});
}
case ARM::BI__builtin_arm_mve_vandq_x_s16:
case ARM::BI__builtin_arm_mve_vandq_x_u16:
case ARM::BI__builtin_arm_mve_vbicq_x_s16:
case ARM::BI__builtin_arm_mve_vbicq_x_u16:
case ARM::BI__builtin_arm_mve_veorq_x_s16:
case ARM::BI__builtin_arm_mve_veorq_x_u16:
case ARM::BI__builtin_arm_mve_vornq_x_s16:
case ARM::BI__builtin_arm_mve_vornq_x_u16:
case ARM::BI__builtin_arm_mve_vorrq_x_s16:
case ARM::BI__builtin_arm_mve_vorrq_x_u16: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vandq_x_s16:
    Param0 = Intrinsic::arm_mve_and_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vandq_x_u16:
    Param0 = Intrinsic::arm_mve_and_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_x_s16:
    Param0 = Intrinsic::arm_mve_bic_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_x_u16:
    Param0 = Intrinsic::arm_mve_bic_predicated;
    break;
  case ARM::BI__builtin_arm_mve_veorq_x_s16:
    Param0 = Intrinsic::arm_mve_eor_predicated;
    break;
  case ARM::BI__builtin_arm_mve_veorq_x_u16:
    Param0 = Intrinsic::arm_mve_eor_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vornq_x_s16:
    Param0 = Intrinsic::arm_mve_orn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vornq_x_u16:
    Param0 = Intrinsic::arm_mve_orn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_x_s16:
    Param0 = Intrinsic::arm_mve_orr_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_x_u16:
    Param0 = Intrinsic::arm_mve_orr_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val3 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val4 = Builder.CreateBitCast(Val2, Val3);
  Value *Val5 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val7 = Builder.CreateBitCast(Val5, Val6);
  Value *Val8 = EmitScalarExpr(E->getArg(2));
  Value *Val9 = Builder.CreateIntCast(Val8, Int32Ty, false);
  Value *Val10 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val9});
  llvm::Type *Val11 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val12 = Builder.CreateBitCast(Val1, Val11);
  Value *Val13 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val4, Val7, Val10, Val12});
  llvm::Type *Val14 = llvm::FixedVectorType::get(Int16Ty, 8);
  return Builder.CreateBitCast(Val13, Val14);
}
case ARM::BI__builtin_arm_mve_vdupq_x_n_s16:
case ARM::BI__builtin_arm_mve_vdupq_x_n_u16: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val3});
  Value *Val5 = EmitScalarExpr(E->getArg(0));
  Value *Val6 = ARMMVEVectorSplat(Builder, Val5);
  return Builder.CreateSelect(Val4, Val6, Val1);
}
case ARM::BI__builtin_arm_mve_vmvnq_x_n_s16:
case ARM::BI__builtin_arm_mve_vmvnq_x_n_u16: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmvnq_x_n_s16:
    Param0 = true;
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_x_n_u16:
    Param0 = false;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val3});
  Value *Val5 = EmitScalarExpr(E->getArg(0));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateIntCast(Val6, Int16Ty, Param0);
  Value *Val8 = ARMMVEVectorSplat(Builder, Val7);
  Value *Val9 = Builder.CreateNot(Val8);
  return Builder.CreateSelect(Val4, Val9, Val1);
}
case ARM::BI__builtin_arm_mve_vcvtaq_x_s16_f16:
case ARM::BI__builtin_arm_mve_vcvtaq_x_u16_f16:
case ARM::BI__builtin_arm_mve_vcvtmq_x_s16_f16:
case ARM::BI__builtin_arm_mve_vcvtmq_x_u16_f16:
case ARM::BI__builtin_arm_mve_vcvtnq_x_s16_f16:
case ARM::BI__builtin_arm_mve_vcvtnq_x_u16_f16:
case ARM::BI__builtin_arm_mve_vcvtpq_x_s16_f16:
case ARM::BI__builtin_arm_mve_vcvtpq_x_u16_f16: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtaq_x_s16_f16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvta_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vcvtaq_x_u16_f16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvta_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vcvtmq_x_s16_f16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvtm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vcvtmq_x_u16_f16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvtm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vcvtnq_x_s16_f16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvtn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vcvtnq_x_u16_f16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvtn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vcvtpq_x_s16_f16:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvtp_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vcvtpq_x_u16_f16:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvtp_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(HalfTy, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val1, Val3, Val6});
}
case ARM::BI__builtin_arm_mve_vcvtq_x_n_s16_f16:
case ARM::BI__builtin_arm_mve_vcvtq_x_n_u16_f16: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtq_x_n_s16_f16:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_x_n_u16_f16:
    Param0 = 1;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcvt_fix_predicated, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(HalfTy, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val1, Val3, Val4, Val7});
}
case ARM::BI__builtin_arm_mve_vcaddq_rot270_x_s16:
case ARM::BI__builtin_arm_mve_vcaddq_rot270_x_u16:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_x_s16:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_x_u16:
case ARM::BI__builtin_arm_mve_vhcaddq_rot270_x_s16:
case ARM::BI__builtin_arm_mve_vhcaddq_rot90_x_s16: {
  uint32_t  Param0;
  uint32_t  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_x_s16:
    Param0 = 1;
    Param1 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_x_u16:
    Param0 = 1;
    Param1 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_x_s16:
    Param0 = 1;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_x_u16:
    Param0 = 1;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot270_x_s16:
    Param0 = 0;
    Param1 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot90_x_s16:
    Param0 = 0;
    Param1 = 0;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val4 = EmitScalarExpr(E->getArg(0));
  Value *Val5 = EmitScalarExpr(E->getArg(1));
  Value *Val6 = EmitScalarExpr(E->getArg(2));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val7});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcaddq_predicated, {llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 8)}), {Val2, Val3, Val1, Val4, Val5, Val8});
}
case ARM::BI__builtin_arm_mve_vnegq_s16: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  Value *Val1 = llvm::Constant::getNullValue(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateSub(Val1, Val2);
}
case ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_s16:
case ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_u16:
case ARM::BI__builtin_arm_mve_vuninitializedq_s16:
case ARM::BI__builtin_arm_mve_vuninitializedq_u16: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int16Ty, 8);
  return UndefValue::get(Val0);
}
case ARM::BI__builtin_arm_mve_vddupq_x_wb_u32:
case ARM::BI__builtin_arm_mve_vidupq_x_wb_u32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vddupq_x_wb_u32:
    Param0 = Intrinsic::arm_mve_vddup_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vidupq_x_wb_u32:
    Param0 = Intrinsic::arm_mve_vidup_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Address Val2 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val3 = Builder.CreateLoad(Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val6});
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val1, Val3, Val4, Val7});
  Value *Val9 = Builder.CreateExtractValue(Val8, static_cast<unsigned>(1));
  Builder.CreateStore(Val9, Val2);
  return Builder.CreateExtractValue(Val8, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vdwdupq_x_wb_u32:
case ARM::BI__builtin_arm_mve_viwdupq_x_wb_u32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vdwdupq_x_wb_u32:
    Param0 = Intrinsic::arm_mve_vdwdup_predicated;
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_x_wb_u32:
    Param0 = Intrinsic::arm_mve_viwdup_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Address Val2 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val3 = Builder.CreateLoad(Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = EmitScalarExpr(E->getArg(3));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val7});
  Value *Val9 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val1, Val3, Val4, Val5, Val8});
  Value *Val10 = Builder.CreateExtractValue(Val9, static_cast<unsigned>(1));
  Builder.CreateStore(Val10, Val2);
  return Builder.CreateExtractValue(Val9, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vaddq_x_n_s32:
case ARM::BI__builtin_arm_mve_vaddq_x_n_u32:
case ARM::BI__builtin_arm_mve_vmulq_x_n_s32:
case ARM::BI__builtin_arm_mve_vmulq_x_n_u32:
case ARM::BI__builtin_arm_mve_vsubq_x_n_s32:
case ARM::BI__builtin_arm_mve_vsubq_x_n_u32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vaddq_x_n_s32:
    Param0 = Intrinsic::arm_mve_add_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vaddq_x_n_u32:
    Param0 = Intrinsic::arm_mve_add_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulq_x_n_s32:
    Param0 = Intrinsic::arm_mve_mul_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulq_x_n_u32:
    Param0 = Intrinsic::arm_mve_mul_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsubq_x_n_s32:
    Param0 = Intrinsic::arm_mve_sub_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsubq_x_n_u32:
    Param0 = Intrinsic::arm_mve_sub_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = ARMMVEVectorSplat(Builder, Val3);
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val4, Val7, Val1});
}
case ARM::BI__builtin_arm_mve_vhaddq_x_n_s32:
case ARM::BI__builtin_arm_mve_vhaddq_x_n_u32:
case ARM::BI__builtin_arm_mve_vhsubq_x_n_s32:
case ARM::BI__builtin_arm_mve_vhsubq_x_n_u32: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vhaddq_x_n_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_hadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_x_n_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_hadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_x_n_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_hsub_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_x_n_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_hsub_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = ARMMVEVectorSplat(Builder, Val3);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val6 = EmitScalarExpr(E->getArg(2));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val7});
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val4, Val5, Val8, Val1});
}
case ARM::BI__builtin_arm_mve_vabsq_x_s32:
case ARM::BI__builtin_arm_mve_vclsq_x_s32:
case ARM::BI__builtin_arm_mve_vclzq_x_s32:
case ARM::BI__builtin_arm_mve_vclzq_x_u32:
case ARM::BI__builtin_arm_mve_vmvnq_x_s32:
case ARM::BI__builtin_arm_mve_vmvnq_x_u32:
case ARM::BI__builtin_arm_mve_vnegq_x_s32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vabsq_x_s32:
    Param0 = Intrinsic::arm_mve_abs_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vclsq_x_s32:
    Param0 = Intrinsic::arm_mve_cls_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vclzq_x_s32:
    Param0 = Intrinsic::arm_mve_clz_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vclzq_x_u32:
    Param0 = Intrinsic::arm_mve_clz_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_x_s32:
    Param0 = Intrinsic::arm_mve_mvn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_x_u32:
    Param0 = Intrinsic::arm_mve_mvn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vnegq_x_s32:
    Param0 = Intrinsic::arm_mve_neg_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val4});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val5, Val1});
}
case ARM::BI__builtin_arm_mve_vddupq_x_n_u32:
case ARM::BI__builtin_arm_mve_vidupq_x_n_u32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vddupq_x_n_u32:
    Param0 = Intrinsic::arm_mve_vddup_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vidupq_x_n_u32:
    Param0 = Intrinsic::arm_mve_vidup_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val5});
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val1, Val2, Val3, Val6});
  return Builder.CreateExtractValue(Val7, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vbrsrq_x_n_s32:
case ARM::BI__builtin_arm_mve_vbrsrq_x_n_u32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vbrsrq_x_n_s32:
    Param0 = Intrinsic::arm_mve_vbrsr_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbrsrq_x_n_u32:
    Param0 = Intrinsic::arm_mve_vbrsr_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val1, Val2, Val3, Val6});
}
case ARM::BI__builtin_arm_mve_vaddq_x_s32:
case ARM::BI__builtin_arm_mve_vaddq_x_u32:
case ARM::BI__builtin_arm_mve_vmulq_x_s32:
case ARM::BI__builtin_arm_mve_vmulq_x_u32:
case ARM::BI__builtin_arm_mve_vshlq_x_n_s32:
case ARM::BI__builtin_arm_mve_vshlq_x_n_u32:
case ARM::BI__builtin_arm_mve_vsubq_x_s32:
case ARM::BI__builtin_arm_mve_vsubq_x_u32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vaddq_x_s32:
    Param0 = Intrinsic::arm_mve_add_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vaddq_x_u32:
    Param0 = Intrinsic::arm_mve_add_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulq_x_s32:
    Param0 = Intrinsic::arm_mve_mul_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulq_x_u32:
    Param0 = Intrinsic::arm_mve_mul_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshlq_x_n_s32:
    Param0 = Intrinsic::arm_mve_shl_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshlq_x_n_u32:
    Param0 = Intrinsic::arm_mve_shl_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsubq_x_s32:
    Param0 = Intrinsic::arm_mve_sub_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsubq_x_u32:
    Param0 = Intrinsic::arm_mve_sub_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val3, Val6, Val1});
}
case ARM::BI__builtin_arm_mve_vdwdupq_x_n_u32:
case ARM::BI__builtin_arm_mve_viwdupq_x_n_u32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vdwdupq_x_n_u32:
    Param0 = Intrinsic::arm_mve_vdwdup_predicated;
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_x_n_u32:
    Param0 = Intrinsic::arm_mve_viwdup_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val6});
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val1, Val2, Val3, Val4, Val7});
  return Builder.CreateExtractValue(Val8, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vrshlq_x_s32:
case ARM::BI__builtin_arm_mve_vrshlq_x_u32:
case ARM::BI__builtin_arm_mve_vshlq_x_s32:
case ARM::BI__builtin_arm_mve_vshlq_x_u32: {
  uint32_t  Param0;
  uint32_t  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vrshlq_x_s32:
    Param0 = 1;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_x_u32:
    Param0 = 1;
    Param1 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vshlq_x_s32:
    Param0 = 0;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vshlq_x_u32:
    Param0 = 0;
    Param1 = 1;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, 0);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val6 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val7 = EmitScalarExpr(E->getArg(2));
  Value *Val8 = Builder.CreateIntCast(Val7, Int32Ty, false);
  Value *Val9 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val8});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vshl_vector_predicated, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val3, Val4, Val5, Val6, Val9, Val1});
}
case ARM::BI__builtin_arm_mve_vmullbq_poly_x_p16:
case ARM::BI__builtin_arm_mve_vmulltq_poly_x_p16: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmullbq_poly_x_p16:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_poly_x_p16:
    Param0 = 1;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_mull_poly_predicated, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val3, Val4, Val7, Val1});
}
case ARM::BI__builtin_arm_mve_vabdq_x_s32:
case ARM::BI__builtin_arm_mve_vabdq_x_u32:
case ARM::BI__builtin_arm_mve_vhaddq_x_s32:
case ARM::BI__builtin_arm_mve_vhaddq_x_u32:
case ARM::BI__builtin_arm_mve_vhsubq_x_s32:
case ARM::BI__builtin_arm_mve_vhsubq_x_u32:
case ARM::BI__builtin_arm_mve_vmaxq_x_s32:
case ARM::BI__builtin_arm_mve_vmaxq_x_u32:
case ARM::BI__builtin_arm_mve_vminq_x_s32:
case ARM::BI__builtin_arm_mve_vminq_x_u32:
case ARM::BI__builtin_arm_mve_vmulhq_x_s32:
case ARM::BI__builtin_arm_mve_vmulhq_x_u32:
case ARM::BI__builtin_arm_mve_vrhaddq_x_s32:
case ARM::BI__builtin_arm_mve_vrhaddq_x_u32:
case ARM::BI__builtin_arm_mve_vrmulhq_x_s32:
case ARM::BI__builtin_arm_mve_vrmulhq_x_u32:
case ARM::BI__builtin_arm_mve_vrshrq_x_n_s32:
case ARM::BI__builtin_arm_mve_vrshrq_x_n_u32:
case ARM::BI__builtin_arm_mve_vshrq_x_n_s32:
case ARM::BI__builtin_arm_mve_vshrq_x_n_u32: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vabdq_x_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_abd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vabdq_x_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_abd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_x_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_hadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_x_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_hadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_x_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_hsub_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_x_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_hsub_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmaxq_x_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_max_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmaxq_x_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_max_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vminq_x_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_min_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vminq_x_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_min_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_x_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_mulh_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_x_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_mulh_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_x_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_rhadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_x_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_rhadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_x_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_rmulh_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_x_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_rmulh_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_x_n_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vrshr_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_x_n_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vrshr_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshrq_x_n_s32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_shr_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshrq_x_n_u32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_shr_imm_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val3, Val4, Val7, Val1});
}
case ARM::BI__builtin_arm_mve_vmullbq_int_x_s16:
case ARM::BI__builtin_arm_mve_vmullbq_int_x_u16:
case ARM::BI__builtin_arm_mve_vmulltq_int_x_s16:
case ARM::BI__builtin_arm_mve_vmulltq_int_x_u16:
case ARM::BI__builtin_arm_mve_vshllbq_x_n_s16:
case ARM::BI__builtin_arm_mve_vshllbq_x_n_u16:
case ARM::BI__builtin_arm_mve_vshlltq_x_n_s16:
case ARM::BI__builtin_arm_mve_vshlltq_x_n_u16: {
  uint32_t  Param0;
  uint32_t  Param1;
  Intrinsic::ID  Param2;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmullbq_int_x_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_mull_int_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmullbq_int_x_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_mull_int_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_x_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_mull_int_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_x_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_mull_int_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshllbq_x_n_s16:
    Param0 = 0;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_vshll_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshllbq_x_n_u16:
    Param0 = 1;
    Param1 = 0;
    Param2 = Intrinsic::arm_mve_vshll_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshlltq_x_n_s16:
    Param0 = 0;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_vshll_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshlltq_x_n_u16:
    Param0 = 1;
    Param1 = 1;
    Param2 = Intrinsic::arm_mve_vshll_imm_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val6 = EmitScalarExpr(E->getArg(2));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val7});
  return Builder.CreateCall(CGM.getIntrinsic(Param2, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val3, Val4, Val5, Val8, Val1});
}
case ARM::BI__builtin_arm_mve_vrev64q_x_s32:
case ARM::BI__builtin_arm_mve_vrev64q_x_u32: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, 64);
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vrev_predicated, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val3, Val6, Val1});
}
case ARM::BI__builtin_arm_mve_vcvtq_x_s32_f32:
case ARM::BI__builtin_arm_mve_vcvtq_x_u32_f32: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtq_x_s32_f32:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_x_u32_f32:
    Param0 = 1;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcvt_fp_int_predicated, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(FloatTy, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val3, Val6, Val1});
}
case ARM::BI__builtin_arm_mve_vmovlbq_x_s16:
case ARM::BI__builtin_arm_mve_vmovlbq_x_u16:
case ARM::BI__builtin_arm_mve_vmovltq_x_s16:
case ARM::BI__builtin_arm_mve_vmovltq_x_u16: {
  uint32_t  Param0;
  uint32_t  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmovlbq_x_s16:
    Param0 = 0;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vmovlbq_x_u16:
    Param0 = 1;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vmovltq_x_s16:
    Param0 = 0;
    Param1 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vmovltq_x_u16:
    Param0 = 1;
    Param1 = 1;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val5 = EmitScalarExpr(E->getArg(1));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vmovl_predicated, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Int16Ty, 8), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val3, Val4, Val7, Val1});
}
case ARM::BI__builtin_arm_mve_vandq_x_s32:
case ARM::BI__builtin_arm_mve_vandq_x_u32:
case ARM::BI__builtin_arm_mve_vbicq_x_s32:
case ARM::BI__builtin_arm_mve_vbicq_x_u32:
case ARM::BI__builtin_arm_mve_veorq_x_s32:
case ARM::BI__builtin_arm_mve_veorq_x_u32:
case ARM::BI__builtin_arm_mve_vornq_x_s32:
case ARM::BI__builtin_arm_mve_vornq_x_u32:
case ARM::BI__builtin_arm_mve_vorrq_x_s32:
case ARM::BI__builtin_arm_mve_vorrq_x_u32: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vandq_x_s32:
    Param0 = Intrinsic::arm_mve_and_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vandq_x_u32:
    Param0 = Intrinsic::arm_mve_and_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_x_s32:
    Param0 = Intrinsic::arm_mve_bic_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_x_u32:
    Param0 = Intrinsic::arm_mve_bic_predicated;
    break;
  case ARM::BI__builtin_arm_mve_veorq_x_s32:
    Param0 = Intrinsic::arm_mve_eor_predicated;
    break;
  case ARM::BI__builtin_arm_mve_veorq_x_u32:
    Param0 = Intrinsic::arm_mve_eor_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vornq_x_s32:
    Param0 = Intrinsic::arm_mve_orn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vornq_x_u32:
    Param0 = Intrinsic::arm_mve_orn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_x_s32:
    Param0 = Intrinsic::arm_mve_orr_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_x_u32:
    Param0 = Intrinsic::arm_mve_orr_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val3 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val4 = Builder.CreateBitCast(Val2, Val3);
  Value *Val5 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val7 = Builder.CreateBitCast(Val5, Val6);
  Value *Val8 = EmitScalarExpr(E->getArg(2));
  Value *Val9 = Builder.CreateIntCast(Val8, Int32Ty, false);
  Value *Val10 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val9});
  llvm::Type *Val11 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val12 = Builder.CreateBitCast(Val1, Val11);
  Value *Val13 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val4, Val7, Val10, Val12});
  llvm::Type *Val14 = llvm::FixedVectorType::get(Int32Ty, 4);
  return Builder.CreateBitCast(Val13, Val14);
}
case ARM::BI__builtin_arm_mve_vdupq_x_n_s32:
case ARM::BI__builtin_arm_mve_vdupq_x_n_u32: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val3});
  Value *Val5 = EmitScalarExpr(E->getArg(0));
  Value *Val6 = ARMMVEVectorSplat(Builder, Val5);
  return Builder.CreateSelect(Val4, Val6, Val1);
}
case ARM::BI__builtin_arm_mve_vmvnq_x_n_s32:
case ARM::BI__builtin_arm_mve_vmvnq_x_n_u32: {
  bool  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmvnq_x_n_s32:
    Param0 = true;
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_x_n_u32:
    Param0 = false;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val3});
  Value *Val5 = EmitScalarExpr(E->getArg(0));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, Param0);
  Value *Val7 = ARMMVEVectorSplat(Builder, Val6);
  Value *Val8 = Builder.CreateNot(Val7);
  return Builder.CreateSelect(Val4, Val8, Val1);
}
case ARM::BI__builtin_arm_mve_vcvtaq_x_s32_f32:
case ARM::BI__builtin_arm_mve_vcvtaq_x_u32_f32:
case ARM::BI__builtin_arm_mve_vcvtmq_x_s32_f32:
case ARM::BI__builtin_arm_mve_vcvtmq_x_u32_f32:
case ARM::BI__builtin_arm_mve_vcvtnq_x_s32_f32:
case ARM::BI__builtin_arm_mve_vcvtnq_x_u32_f32:
case ARM::BI__builtin_arm_mve_vcvtpq_x_s32_f32:
case ARM::BI__builtin_arm_mve_vcvtpq_x_u32_f32: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtaq_x_s32_f32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvta_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vcvtaq_x_u32_f32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvta_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vcvtmq_x_s32_f32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvtm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vcvtmq_x_u32_f32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvtm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vcvtnq_x_s32_f32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvtn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vcvtnq_x_u32_f32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvtn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vcvtpq_x_s32_f32:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vcvtp_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vcvtpq_x_u32_f32:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vcvtp_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(FloatTy, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val1, Val3, Val6});
}
case ARM::BI__builtin_arm_mve_vcvtq_x_n_s32_f32:
case ARM::BI__builtin_arm_mve_vcvtq_x_n_u32_f32: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcvtq_x_n_s32_f32:
    Param0 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vcvtq_x_n_u32_f32:
    Param0 = 1;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = EmitScalarExpr(E->getArg(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcvt_fix_predicated, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(FloatTy, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val1, Val3, Val4, Val7});
}
case ARM::BI__builtin_arm_mve_vcaddq_rot270_x_s32:
case ARM::BI__builtin_arm_mve_vcaddq_rot270_x_u32:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_x_s32:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_x_u32:
case ARM::BI__builtin_arm_mve_vhcaddq_rot270_x_s32:
case ARM::BI__builtin_arm_mve_vhcaddq_rot90_x_s32: {
  uint32_t  Param0;
  uint32_t  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_x_s32:
    Param0 = 1;
    Param1 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_x_u32:
    Param0 = 1;
    Param1 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_x_s32:
    Param0 = 1;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_x_u32:
    Param0 = 1;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot270_x_s32:
    Param0 = 0;
    Param1 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot90_x_s32:
    Param0 = 0;
    Param1 = 0;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val4 = EmitScalarExpr(E->getArg(0));
  Value *Val5 = EmitScalarExpr(E->getArg(1));
  Value *Val6 = EmitScalarExpr(E->getArg(2));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val7});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcaddq_predicated, {llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 4)}), {Val2, Val3, Val1, Val4, Val5, Val8});
}
case ARM::BI__builtin_arm_mve_vnegq_s32: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  Value *Val1 = llvm::Constant::getNullValue(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateSub(Val1, Val2);
}
case ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_s32:
case ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_u32:
case ARM::BI__builtin_arm_mve_vuninitializedq_s32:
case ARM::BI__builtin_arm_mve_vuninitializedq_u32: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int32Ty, 4);
  return UndefValue::get(Val0);
}
case ARM::BI__builtin_arm_mve_vcreateq_f32: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int64Ty, 2);
  Value *Val1 = PoisonValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = Builder.CreateInsertElement(Val1, Val2, static_cast<uint64_t>(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateInsertElement(Val3, Val4, static_cast<uint64_t>(1));
  llvm::Type *Val6 = llvm::FixedVectorType::get(FloatTy, 4);
  return ARMMVEVectorReinterpret(Builder, this, Val5, Val6);
}
case ARM::BI__builtin_arm_mve_vcreateq_f16: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int64Ty, 2);
  Value *Val1 = PoisonValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = Builder.CreateInsertElement(Val1, Val2, static_cast<uint64_t>(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateInsertElement(Val3, Val4, static_cast<uint64_t>(1));
  llvm::Type *Val6 = llvm::FixedVectorType::get(HalfTy, 8);
  return ARMMVEVectorReinterpret(Builder, this, Val5, Val6);
}
case ARM::BI__builtin_arm_mve_vcreateq_s16:
case ARM::BI__builtin_arm_mve_vcreateq_u16: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int64Ty, 2);
  Value *Val1 = PoisonValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = Builder.CreateInsertElement(Val1, Val2, static_cast<uint64_t>(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateInsertElement(Val3, Val4, static_cast<uint64_t>(1));
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int16Ty, 8);
  return ARMMVEVectorReinterpret(Builder, this, Val5, Val6);
}
case ARM::BI__builtin_arm_mve_vcreateq_s32:
case ARM::BI__builtin_arm_mve_vcreateq_u32: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int64Ty, 2);
  Value *Val1 = PoisonValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = Builder.CreateInsertElement(Val1, Val2, static_cast<uint64_t>(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateInsertElement(Val3, Val4, static_cast<uint64_t>(1));
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int32Ty, 4);
  return ARMMVEVectorReinterpret(Builder, this, Val5, Val6);
}
case ARM::BI__builtin_arm_mve_vcreateq_s64:
case ARM::BI__builtin_arm_mve_vcreateq_u64: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int64Ty, 2);
  Value *Val1 = PoisonValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = Builder.CreateInsertElement(Val1, Val2, static_cast<uint64_t>(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateInsertElement(Val3, Val4, static_cast<uint64_t>(1));
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int64Ty, 2);
  return ARMMVEVectorReinterpret(Builder, this, Val5, Val6);
}
case ARM::BI__builtin_arm_mve_vcreateq_s8:
case ARM::BI__builtin_arm_mve_vcreateq_u8: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int64Ty, 2);
  Value *Val1 = PoisonValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = Builder.CreateInsertElement(Val1, Val2, static_cast<uint64_t>(0));
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateInsertElement(Val3, Val4, static_cast<uint64_t>(1));
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int8Ty, 16);
  return ARMMVEVectorReinterpret(Builder, this, Val5, Val6);
}
case ARM::BI__builtin_arm_mve_vmullbq_int_x_s32:
case ARM::BI__builtin_arm_mve_vmullbq_int_x_u32:
case ARM::BI__builtin_arm_mve_vmulltq_int_x_s32:
case ARM::BI__builtin_arm_mve_vmulltq_int_x_u32: {
  uint32_t  Param0;
  uint32_t  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vmullbq_int_x_s32:
    Param0 = 0;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vmullbq_int_x_u32:
    Param0 = 1;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_x_s32:
    Param0 = 0;
    Param1 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vmulltq_int_x_u32:
    Param0 = 1;
    Param1 = 1;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int64Ty, 2);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val6 = EmitScalarExpr(E->getArg(2));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 2)}), {Val7});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_mull_int_predicated, {llvm::FixedVectorType::get(Int64Ty, 2), llvm::FixedVectorType::get(Int32Ty, 4), llvm::FixedVectorType::get(Builder.getInt1Ty(), 2)}), {Val2, Val3, Val4, Val5, Val8, Val1});
}
case ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_s64:
case ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_u64:
case ARM::BI__builtin_arm_mve_vuninitializedq_s64:
case ARM::BI__builtin_arm_mve_vuninitializedq_u64: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int64Ty, 2);
  return UndefValue::get(Val0);
}
case ARM::BI__builtin_arm_mve_vddupq_x_wb_u8:
case ARM::BI__builtin_arm_mve_vidupq_x_wb_u8: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vddupq_x_wb_u8:
    Param0 = Intrinsic::arm_mve_vddup_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vidupq_x_wb_u8:
    Param0 = Intrinsic::arm_mve_vidup_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val1 = UndefValue::get(Val0);
  Address Val2 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val3 = Builder.CreateLoad(Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val6});
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val1, Val3, Val4, Val7});
  Value *Val9 = Builder.CreateExtractValue(Val8, static_cast<unsigned>(1));
  Builder.CreateStore(Val9, Val2);
  return Builder.CreateExtractValue(Val8, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vdwdupq_x_wb_u8:
case ARM::BI__builtin_arm_mve_viwdupq_x_wb_u8: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vdwdupq_x_wb_u8:
    Param0 = Intrinsic::arm_mve_vdwdup_predicated;
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_x_wb_u8:
    Param0 = Intrinsic::arm_mve_viwdup_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val1 = UndefValue::get(Val0);
  Address Val2 = EmitPointerWithAlignment(E->getArg(0));
  Value *Val3 = Builder.CreateLoad(Val2);
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = EmitScalarExpr(E->getArg(3));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val7});
  Value *Val9 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val1, Val3, Val4, Val5, Val8});
  Value *Val10 = Builder.CreateExtractValue(Val9, static_cast<unsigned>(1));
  Builder.CreateStore(Val10, Val2);
  return Builder.CreateExtractValue(Val9, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vaddq_x_n_s8:
case ARM::BI__builtin_arm_mve_vaddq_x_n_u8:
case ARM::BI__builtin_arm_mve_vmulq_x_n_s8:
case ARM::BI__builtin_arm_mve_vmulq_x_n_u8:
case ARM::BI__builtin_arm_mve_vsubq_x_n_s8:
case ARM::BI__builtin_arm_mve_vsubq_x_n_u8: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vaddq_x_n_s8:
    Param0 = Intrinsic::arm_mve_add_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vaddq_x_n_u8:
    Param0 = Intrinsic::arm_mve_add_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulq_x_n_s8:
    Param0 = Intrinsic::arm_mve_mul_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulq_x_n_u8:
    Param0 = Intrinsic::arm_mve_mul_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsubq_x_n_s8:
    Param0 = Intrinsic::arm_mve_sub_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsubq_x_n_u8:
    Param0 = Intrinsic::arm_mve_sub_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = ARMMVEVectorSplat(Builder, Val3);
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val2, Val4, Val7, Val1});
}
case ARM::BI__builtin_arm_mve_vhaddq_x_n_s8:
case ARM::BI__builtin_arm_mve_vhaddq_x_n_u8:
case ARM::BI__builtin_arm_mve_vhsubq_x_n_s8:
case ARM::BI__builtin_arm_mve_vhsubq_x_n_u8: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vhaddq_x_n_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_hadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_x_n_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_hadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_x_n_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_hsub_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_x_n_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_hsub_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = ARMMVEVectorSplat(Builder, Val3);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val6 = EmitScalarExpr(E->getArg(2));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val7});
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val2, Val4, Val5, Val8, Val1});
}
case ARM::BI__builtin_arm_mve_vabsq_x_s8:
case ARM::BI__builtin_arm_mve_vclsq_x_s8:
case ARM::BI__builtin_arm_mve_vclzq_x_s8:
case ARM::BI__builtin_arm_mve_vclzq_x_u8:
case ARM::BI__builtin_arm_mve_vmvnq_x_s8:
case ARM::BI__builtin_arm_mve_vmvnq_x_u8:
case ARM::BI__builtin_arm_mve_vnegq_x_s8: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vabsq_x_s8:
    Param0 = Intrinsic::arm_mve_abs_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vclsq_x_s8:
    Param0 = Intrinsic::arm_mve_cls_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vclzq_x_s8:
    Param0 = Intrinsic::arm_mve_clz_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vclzq_x_u8:
    Param0 = Intrinsic::arm_mve_clz_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_x_s8:
    Param0 = Intrinsic::arm_mve_mvn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmvnq_x_u8:
    Param0 = Intrinsic::arm_mve_mvn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vnegq_x_s8:
    Param0 = Intrinsic::arm_mve_neg_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = Builder.CreateIntCast(Val3, Int32Ty, false);
  Value *Val5 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val4});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val2, Val5, Val1});
}
case ARM::BI__builtin_arm_mve_vddupq_x_n_u8:
case ARM::BI__builtin_arm_mve_vidupq_x_n_u8: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vddupq_x_n_u8:
    Param0 = Intrinsic::arm_mve_vddup_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vidupq_x_n_u8:
    Param0 = Intrinsic::arm_mve_vidup_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val5});
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val1, Val2, Val3, Val6});
  return Builder.CreateExtractValue(Val7, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vbrsrq_x_n_s8:
case ARM::BI__builtin_arm_mve_vbrsrq_x_n_u8: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vbrsrq_x_n_s8:
    Param0 = Intrinsic::arm_mve_vbrsr_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbrsrq_x_n_u8:
    Param0 = Intrinsic::arm_mve_vbrsr_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val1, Val2, Val3, Val6});
}
case ARM::BI__builtin_arm_mve_vaddq_x_s8:
case ARM::BI__builtin_arm_mve_vaddq_x_u8:
case ARM::BI__builtin_arm_mve_vmulq_x_s8:
case ARM::BI__builtin_arm_mve_vmulq_x_u8:
case ARM::BI__builtin_arm_mve_vshlq_x_n_s8:
case ARM::BI__builtin_arm_mve_vshlq_x_n_u8:
case ARM::BI__builtin_arm_mve_vsubq_x_s8:
case ARM::BI__builtin_arm_mve_vsubq_x_u8: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vaddq_x_s8:
    Param0 = Intrinsic::arm_mve_add_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vaddq_x_u8:
    Param0 = Intrinsic::arm_mve_add_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulq_x_s8:
    Param0 = Intrinsic::arm_mve_mul_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulq_x_u8:
    Param0 = Intrinsic::arm_mve_mul_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshlq_x_n_s8:
    Param0 = Intrinsic::arm_mve_shl_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshlq_x_n_u8:
    Param0 = Intrinsic::arm_mve_shl_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsubq_x_s8:
    Param0 = Intrinsic::arm_mve_sub_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vsubq_x_u8:
    Param0 = Intrinsic::arm_mve_sub_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val2, Val3, Val6, Val1});
}
case ARM::BI__builtin_arm_mve_vdwdupq_x_n_u8:
case ARM::BI__builtin_arm_mve_viwdupq_x_n_u8: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vdwdupq_x_n_u8:
    Param0 = Intrinsic::arm_mve_vdwdup_predicated;
    break;
  case ARM::BI__builtin_arm_mve_viwdupq_x_n_u8:
    Param0 = Intrinsic::arm_mve_viwdup_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = EmitScalarExpr(E->getArg(2));
  Value *Val5 = EmitScalarExpr(E->getArg(3));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val6});
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val1, Val2, Val3, Val4, Val7});
  return Builder.CreateExtractValue(Val8, static_cast<unsigned>(0));
}
case ARM::BI__builtin_arm_mve_vrshlq_x_s8:
case ARM::BI__builtin_arm_mve_vrshlq_x_u8:
case ARM::BI__builtin_arm_mve_vshlq_x_s8:
case ARM::BI__builtin_arm_mve_vshlq_x_u8: {
  uint32_t  Param0;
  uint32_t  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vrshlq_x_s8:
    Param0 = 1;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vrshlq_x_u8:
    Param0 = 1;
    Param1 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vshlq_x_s8:
    Param0 = 0;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vshlq_x_u8:
    Param0 = 0;
    Param1 = 1;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, 0);
  Value *Val5 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val6 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val7 = EmitScalarExpr(E->getArg(2));
  Value *Val8 = Builder.CreateIntCast(Val7, Int32Ty, false);
  Value *Val9 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val8});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vshl_vector_predicated, {llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val2, Val3, Val4, Val5, Val6, Val9, Val1});
}
case ARM::BI__builtin_arm_mve_vabdq_x_s8:
case ARM::BI__builtin_arm_mve_vabdq_x_u8:
case ARM::BI__builtin_arm_mve_vhaddq_x_s8:
case ARM::BI__builtin_arm_mve_vhaddq_x_u8:
case ARM::BI__builtin_arm_mve_vhsubq_x_s8:
case ARM::BI__builtin_arm_mve_vhsubq_x_u8:
case ARM::BI__builtin_arm_mve_vmaxq_x_s8:
case ARM::BI__builtin_arm_mve_vmaxq_x_u8:
case ARM::BI__builtin_arm_mve_vminq_x_s8:
case ARM::BI__builtin_arm_mve_vminq_x_u8:
case ARM::BI__builtin_arm_mve_vmulhq_x_s8:
case ARM::BI__builtin_arm_mve_vmulhq_x_u8:
case ARM::BI__builtin_arm_mve_vrhaddq_x_s8:
case ARM::BI__builtin_arm_mve_vrhaddq_x_u8:
case ARM::BI__builtin_arm_mve_vrmulhq_x_s8:
case ARM::BI__builtin_arm_mve_vrmulhq_x_u8:
case ARM::BI__builtin_arm_mve_vrshrq_x_n_s8:
case ARM::BI__builtin_arm_mve_vrshrq_x_n_u8:
case ARM::BI__builtin_arm_mve_vshrq_x_n_s8:
case ARM::BI__builtin_arm_mve_vshrq_x_n_u8: {
  uint32_t  Param0;
  Intrinsic::ID  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vabdq_x_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_abd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vabdq_x_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_abd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_x_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_hadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhaddq_x_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_hadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_x_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_hsub_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vhsubq_x_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_hsub_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmaxq_x_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_max_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmaxq_x_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_max_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vminq_x_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_min_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vminq_x_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_min_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_x_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_mulh_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vmulhq_x_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_mulh_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_x_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_rhadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrhaddq_x_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_rhadd_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_x_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_rmulh_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrmulhq_x_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_rmulh_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_x_n_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_vrshr_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vrshrq_x_n_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_vrshr_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshrq_x_n_s8:
    Param0 = 0;
    Param1 = Intrinsic::arm_mve_shr_imm_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vshrq_x_n_u8:
    Param0 = 1;
    Param1 = Intrinsic::arm_mve_shr_imm_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = EmitScalarExpr(E->getArg(1));
  Value *Val4 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val5 = EmitScalarExpr(E->getArg(2));
  Value *Val6 = Builder.CreateIntCast(Val5, Int32Ty, false);
  Value *Val7 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val6});
  return Builder.CreateCall(CGM.getIntrinsic(Param1, {llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val2, Val3, Val4, Val7, Val1});
}
case ARM::BI__builtin_arm_mve_vrev16q_x_s8:
case ARM::BI__builtin_arm_mve_vrev16q_x_u8:
case ARM::BI__builtin_arm_mve_vrev32q_x_s8:
case ARM::BI__builtin_arm_mve_vrev32q_x_u8:
case ARM::BI__builtin_arm_mve_vrev64q_x_s8:
case ARM::BI__builtin_arm_mve_vrev64q_x_u8: {
  uint32_t  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vrev16q_x_s8:
    Param0 = 16;
    break;
  case ARM::BI__builtin_arm_mve_vrev16q_x_u8:
    Param0 = 16;
    break;
  case ARM::BI__builtin_arm_mve_vrev32q_x_s8:
    Param0 = 32;
    break;
  case ARM::BI__builtin_arm_mve_vrev32q_x_u8:
    Param0 = 32;
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_x_s8:
    Param0 = 64;
    break;
  case ARM::BI__builtin_arm_mve_vrev64q_x_u8:
    Param0 = 64;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val4 = EmitScalarExpr(E->getArg(1));
  Value *Val5 = Builder.CreateIntCast(Val4, Int32Ty, false);
  Value *Val6 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val5});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vrev_predicated, {llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val2, Val3, Val6, Val1});
}
case ARM::BI__builtin_arm_mve_vandq_x_s8:
case ARM::BI__builtin_arm_mve_vandq_x_u8:
case ARM::BI__builtin_arm_mve_vbicq_x_s8:
case ARM::BI__builtin_arm_mve_vbicq_x_u8:
case ARM::BI__builtin_arm_mve_veorq_x_s8:
case ARM::BI__builtin_arm_mve_veorq_x_u8:
case ARM::BI__builtin_arm_mve_vornq_x_s8:
case ARM::BI__builtin_arm_mve_vornq_x_u8:
case ARM::BI__builtin_arm_mve_vorrq_x_s8:
case ARM::BI__builtin_arm_mve_vorrq_x_u8: {
  Intrinsic::ID  Param0;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vandq_x_s8:
    Param0 = Intrinsic::arm_mve_and_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vandq_x_u8:
    Param0 = Intrinsic::arm_mve_and_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_x_s8:
    Param0 = Intrinsic::arm_mve_bic_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vbicq_x_u8:
    Param0 = Intrinsic::arm_mve_bic_predicated;
    break;
  case ARM::BI__builtin_arm_mve_veorq_x_s8:
    Param0 = Intrinsic::arm_mve_eor_predicated;
    break;
  case ARM::BI__builtin_arm_mve_veorq_x_u8:
    Param0 = Intrinsic::arm_mve_eor_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vornq_x_s8:
    Param0 = Intrinsic::arm_mve_orn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vornq_x_u8:
    Param0 = Intrinsic::arm_mve_orn_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_x_s8:
    Param0 = Intrinsic::arm_mve_orr_predicated;
    break;
  case ARM::BI__builtin_arm_mve_vorrq_x_u8:
    Param0 = Intrinsic::arm_mve_orr_predicated;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  llvm::Type *Val3 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val4 = Builder.CreateBitCast(Val2, Val3);
  Value *Val5 = EmitScalarExpr(E->getArg(1));
  llvm::Type *Val6 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val7 = Builder.CreateBitCast(Val5, Val6);
  Value *Val8 = EmitScalarExpr(E->getArg(2));
  Value *Val9 = Builder.CreateIntCast(Val8, Int32Ty, false);
  Value *Val10 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val9});
  llvm::Type *Val11 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val12 = Builder.CreateBitCast(Val1, Val11);
  Value *Val13 = Builder.CreateCall(CGM.getIntrinsic(Param0, {llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val4, Val7, Val10, Val12});
  llvm::Type *Val14 = llvm::FixedVectorType::get(Int8Ty, 16);
  return Builder.CreateBitCast(Val13, Val14);
}
case ARM::BI__builtin_arm_mve_vdupq_x_n_s8:
case ARM::BI__builtin_arm_mve_vdupq_x_n_u8: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(1));
  Value *Val3 = Builder.CreateIntCast(Val2, Int32Ty, false);
  Value *Val4 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val3});
  Value *Val5 = EmitScalarExpr(E->getArg(0));
  Value *Val6 = ARMMVEVectorSplat(Builder, Val5);
  return Builder.CreateSelect(Val4, Val6, Val1);
}
case ARM::BI__builtin_arm_mve_vcaddq_rot270_x_s8:
case ARM::BI__builtin_arm_mve_vcaddq_rot270_x_u8:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_x_s8:
case ARM::BI__builtin_arm_mve_vcaddq_rot90_x_u8:
case ARM::BI__builtin_arm_mve_vhcaddq_rot270_x_s8:
case ARM::BI__builtin_arm_mve_vhcaddq_rot90_x_s8: {
  uint32_t  Param0;
  uint32_t  Param1;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_x_s8:
    Param0 = 1;
    Param1 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot270_x_u8:
    Param0 = 1;
    Param1 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_x_s8:
    Param0 = 1;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vcaddq_rot90_x_u8:
    Param0 = 1;
    Param1 = 0;
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot270_x_s8:
    Param0 = 0;
    Param1 = 1;
    break;
  case ARM::BI__builtin_arm_mve_vhcaddq_rot90_x_s8:
    Param0 = 0;
    Param1 = 0;
    break;
  }
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val1 = UndefValue::get(Val0);
  Value *Val2 = llvm::ConstantInt::get(Int32Ty, Param0);
  Value *Val3 = llvm::ConstantInt::get(Int32Ty, Param1);
  Value *Val4 = EmitScalarExpr(E->getArg(0));
  Value *Val5 = EmitScalarExpr(E->getArg(1));
  Value *Val6 = EmitScalarExpr(E->getArg(2));
  Value *Val7 = Builder.CreateIntCast(Val6, Int32Ty, false);
  Value *Val8 = Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_pred_i2v, {llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val7});
  return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_mve_vcaddq_predicated, {llvm::FixedVectorType::get(Int8Ty, 16), llvm::FixedVectorType::get(Builder.getInt1Ty(), 16)}), {Val2, Val3, Val1, Val4, Val5, Val8});
}
case ARM::BI__builtin_arm_mve_vnegq_s8: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int8Ty, 16);
  Value *Val1 = llvm::Constant::getNullValue(Val0);
  Value *Val2 = EmitScalarExpr(E->getArg(0));
  return Builder.CreateSub(Val1, Val2);
}
case ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_s8:
case ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_u8:
case ARM::BI__builtin_arm_mve_vuninitializedq_s8:
case ARM::BI__builtin_arm_mve_vuninitializedq_u8: {
  llvm::Type *Val0 = llvm::FixedVectorType::get(Int8Ty, 16);
  return UndefValue::get(Val0);
}
