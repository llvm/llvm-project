case RISCVVector::BI__builtin_rvv_sf_vc_fv_se:
  ID = Intrinsic::riscv_sf_vc_fv_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {Ops[0]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_fvv_se:
  ID = Intrinsic::riscv_sf_vc_fvv_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_fvw_se:
  ID = Intrinsic::riscv_sf_vc_fvw_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_i_se:
  ID = Intrinsic::riscv_sf_vc_i_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_iv_se:
  ID = Intrinsic::riscv_sf_vc_iv_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {Ops[0]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_ivv_se:
  ID = Intrinsic::riscv_sf_vc_ivv_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_ivw_se:
  ID = Intrinsic::riscv_sf_vc_ivw_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_fv:
  ID = Intrinsic::riscv_sf_vc_v_fv;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_fv_se:
  ID = Intrinsic::riscv_sf_vc_v_fv_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_fvv:
  ID = Intrinsic::riscv_sf_vc_v_fvv;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_fvv_se:
  ID = Intrinsic::riscv_sf_vc_v_fvv_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_fvw:
  ID = Intrinsic::riscv_sf_vc_v_fvw;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_fvw_se:
  ID = Intrinsic::riscv_sf_vc_v_fvw_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_i:
  ID = Intrinsic::riscv_sf_vc_v_i;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_i_se:
  ID = Intrinsic::riscv_sf_vc_v_i_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_iv:
  ID = Intrinsic::riscv_sf_vc_v_iv;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_iv_se:
  ID = Intrinsic::riscv_sf_vc_v_iv_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_ivv:
  ID = Intrinsic::riscv_sf_vc_v_ivv;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_ivv_se:
  ID = Intrinsic::riscv_sf_vc_v_ivv_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_ivw:
  ID = Intrinsic::riscv_sf_vc_v_ivw;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_ivw_se:
  ID = Intrinsic::riscv_sf_vc_v_ivw_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_vv:
  ID = Intrinsic::riscv_sf_vc_v_vv;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_vv_se:
  ID = Intrinsic::riscv_sf_vc_v_vv_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_vvv:
  ID = Intrinsic::riscv_sf_vc_v_vvv;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_vvv_se:
  ID = Intrinsic::riscv_sf_vc_v_vvv_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_vvw:
  ID = Intrinsic::riscv_sf_vc_v_vvw;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_vvw_se:
  ID = Intrinsic::riscv_sf_vc_v_vvw_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_x:
  ID = Intrinsic::riscv_sf_vc_v_x;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_x_se:
  ID = Intrinsic::riscv_sf_vc_v_x_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_xv:
  ID = Intrinsic::riscv_sf_vc_v_xv;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_xv_se:
  ID = Intrinsic::riscv_sf_vc_v_xv_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_xvv:
  ID = Intrinsic::riscv_sf_vc_v_xvv;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_xvv_se:
  ID = Intrinsic::riscv_sf_vc_v_xvv_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_xvw:
  ID = Intrinsic::riscv_sf_vc_v_xvw;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_v_xvw_se:
  ID = Intrinsic::riscv_sf_vc_v_xvw_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {ResultType, Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_vv_se:
  ID = Intrinsic::riscv_sf_vc_vv_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {Ops[0]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_vvv_se:
  ID = Intrinsic::riscv_sf_vc_vvv_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_vvw_se:
  ID = Intrinsic::riscv_sf_vc_vvw_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_x_se:
  ID = Intrinsic::riscv_sf_vc_x_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {Ops[0]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_xv_se:
  ID = Intrinsic::riscv_sf_vc_xv_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {Ops[0]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_xvv_se:
  ID = Intrinsic::riscv_sf_vc_xvv_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vc_xvw_se:
  ID = Intrinsic::riscv_sf_vc_xvw_se;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(), Ops[2]->getType(), Ops[3]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_x_f_qf_rm_tu:
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_x_f_qf_tu:
  ID = Intrinsic::riscv_sf_vfnrclip_x_f_qf;
  PolicyAttrs = 2;
  SegInstSEW = 0;
IsMasked = false;
if (SegInstSEW == (unsigned)-1) {
  auto PointeeType = E->getArg(4294967295      )->getType()->getPointeeType();
  SegInstSEW =       llvm::Log2_64(getContext().getTypeSize(PointeeType));
}

  {
    // LLVM intrinsic
    // Unmasked: (passthru, vector_in, scalar_in, frm, vl)
    // Masked:   (passthru, vector_in, scalar_in, mask, frm, vl, policy)

    SmallVector<llvm::Value*, 7> Operands;
    bool HasMaskedOff = !(
        (IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
    bool HasRoundModeOp = IsMasked ?
      (HasMaskedOff ? Ops.size() == 6 : Ops.size() == 5) :
      (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4);

    unsigned Offset = IsMasked ?
        (HasMaskedOff ? 2 : 1) : (HasMaskedOff ? 1 : 0);

    if (!HasMaskedOff)
      Operands.push_back(llvm::PoisonValue::get(ResultType));
    else
      Operands.push_back(Ops[IsMasked ? 1 : 0]);

    Operands.push_back(Ops[Offset]); // op0
    Operands.push_back(Ops[Offset + 1]); // op1

    if (IsMasked)
      Operands.push_back(Ops[0]); // mask

    if (HasRoundModeOp) {
      Operands.push_back(Ops[Offset + 2]); // frm
      Operands.push_back(Ops[Offset + 3]); // vl
    } else {
      Operands.push_back(ConstantInt::get(Ops[Offset + 2]->getType(), 7)); // frm
      Operands.push_back(Ops[Offset + 2]); // vl
    }

    if (IsMasked)
      Operands.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

    IntrinsicTypes = {ResultType, Ops[Offset]->getType(), Operands.back()->getType()};
    llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
    return Builder.CreateCall(F, Operands, "");
  }
break;
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_x_f_qf_rm:
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_x_f_qf:
  ID = Intrinsic::riscv_sf_vfnrclip_x_f_qf;
  PolicyAttrs = 3;
  SegInstSEW = 0;
IsMasked = false;
if (SegInstSEW == (unsigned)-1) {
  auto PointeeType = E->getArg(4294967295      )->getType()->getPointeeType();
  SegInstSEW =       llvm::Log2_64(getContext().getTypeSize(PointeeType));
}

  {
    // LLVM intrinsic
    // Unmasked: (passthru, vector_in, scalar_in, frm, vl)
    // Masked:   (passthru, vector_in, scalar_in, mask, frm, vl, policy)

    SmallVector<llvm::Value*, 7> Operands;
    bool HasMaskedOff = !(
        (IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
    bool HasRoundModeOp = IsMasked ?
      (HasMaskedOff ? Ops.size() == 6 : Ops.size() == 5) :
      (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4);

    unsigned Offset = IsMasked ?
        (HasMaskedOff ? 2 : 1) : (HasMaskedOff ? 1 : 0);

    if (!HasMaskedOff)
      Operands.push_back(llvm::PoisonValue::get(ResultType));
    else
      Operands.push_back(Ops[IsMasked ? 1 : 0]);

    Operands.push_back(Ops[Offset]); // op0
    Operands.push_back(Ops[Offset + 1]); // op1

    if (IsMasked)
      Operands.push_back(Ops[0]); // mask

    if (HasRoundModeOp) {
      Operands.push_back(Ops[Offset + 2]); // frm
      Operands.push_back(Ops[Offset + 3]); // vl
    } else {
      Operands.push_back(ConstantInt::get(Ops[Offset + 2]->getType(), 7)); // frm
      Operands.push_back(Ops[Offset + 2]); // vl
    }

    if (IsMasked)
      Operands.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

    IntrinsicTypes = {ResultType, Ops[Offset]->getType(), Operands.back()->getType()};
    llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
    return Builder.CreateCall(F, Operands, "");
  }
break;
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_x_f_qf_rm_tum:
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_x_f_qf_tum:
  ID = Intrinsic::riscv_sf_vfnrclip_x_f_qf_mask;
  PolicyAttrs = 2;
  SegInstSEW = 0;
IsMasked = true;
if (SegInstSEW == (unsigned)-1) {
  auto PointeeType = E->getArg(4294967295      )->getType()->getPointeeType();
  SegInstSEW =       llvm::Log2_64(getContext().getTypeSize(PointeeType));
}

  {
    // LLVM intrinsic
    // Unmasked: (passthru, vector_in, scalar_in, frm, vl)
    // Masked:   (passthru, vector_in, scalar_in, mask, frm, vl, policy)

    SmallVector<llvm::Value*, 7> Operands;
    bool HasMaskedOff = !(
        (IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
    bool HasRoundModeOp = IsMasked ?
      (HasMaskedOff ? Ops.size() == 6 : Ops.size() == 5) :
      (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4);

    unsigned Offset = IsMasked ?
        (HasMaskedOff ? 2 : 1) : (HasMaskedOff ? 1 : 0);

    if (!HasMaskedOff)
      Operands.push_back(llvm::PoisonValue::get(ResultType));
    else
      Operands.push_back(Ops[IsMasked ? 1 : 0]);

    Operands.push_back(Ops[Offset]); // op0
    Operands.push_back(Ops[Offset + 1]); // op1

    if (IsMasked)
      Operands.push_back(Ops[0]); // mask

    if (HasRoundModeOp) {
      Operands.push_back(Ops[Offset + 2]); // frm
      Operands.push_back(Ops[Offset + 3]); // vl
    } else {
      Operands.push_back(ConstantInt::get(Ops[Offset + 2]->getType(), 7)); // frm
      Operands.push_back(Ops[Offset + 2]); // vl
    }

    if (IsMasked)
      Operands.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

    IntrinsicTypes = {ResultType, Ops[Offset]->getType(), Operands.back()->getType()};
    llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
    return Builder.CreateCall(F, Operands, "");
  }
break;
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_x_f_qf_rm_m:
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_x_f_qf_m:
  ID = Intrinsic::riscv_sf_vfnrclip_x_f_qf_mask;
  PolicyAttrs = 3;
  SegInstSEW = 0;
IsMasked = true;
if (SegInstSEW == (unsigned)-1) {
  auto PointeeType = E->getArg(4294967295      )->getType()->getPointeeType();
  SegInstSEW =       llvm::Log2_64(getContext().getTypeSize(PointeeType));
}

  {
    // LLVM intrinsic
    // Unmasked: (passthru, vector_in, scalar_in, frm, vl)
    // Masked:   (passthru, vector_in, scalar_in, mask, frm, vl, policy)

    SmallVector<llvm::Value*, 7> Operands;
    bool HasMaskedOff = !(
        (IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
    bool HasRoundModeOp = IsMasked ?
      (HasMaskedOff ? Ops.size() == 6 : Ops.size() == 5) :
      (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4);

    unsigned Offset = IsMasked ?
        (HasMaskedOff ? 2 : 1) : (HasMaskedOff ? 1 : 0);

    if (!HasMaskedOff)
      Operands.push_back(llvm::PoisonValue::get(ResultType));
    else
      Operands.push_back(Ops[IsMasked ? 1 : 0]);

    Operands.push_back(Ops[Offset]); // op0
    Operands.push_back(Ops[Offset + 1]); // op1

    if (IsMasked)
      Operands.push_back(Ops[0]); // mask

    if (HasRoundModeOp) {
      Operands.push_back(Ops[Offset + 2]); // frm
      Operands.push_back(Ops[Offset + 3]); // vl
    } else {
      Operands.push_back(ConstantInt::get(Ops[Offset + 2]->getType(), 7)); // frm
      Operands.push_back(Ops[Offset + 2]); // vl
    }

    if (IsMasked)
      Operands.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

    IntrinsicTypes = {ResultType, Ops[Offset]->getType(), Operands.back()->getType()};
    llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
    return Builder.CreateCall(F, Operands, "");
  }
break;
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_x_f_qf_rm_tumu:
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_x_f_qf_tumu:
  ID = Intrinsic::riscv_sf_vfnrclip_x_f_qf_mask;
  PolicyAttrs = 0;
  SegInstSEW = 0;
IsMasked = true;
if (SegInstSEW == (unsigned)-1) {
  auto PointeeType = E->getArg(4294967295      )->getType()->getPointeeType();
  SegInstSEW =       llvm::Log2_64(getContext().getTypeSize(PointeeType));
}

  {
    // LLVM intrinsic
    // Unmasked: (passthru, vector_in, scalar_in, frm, vl)
    // Masked:   (passthru, vector_in, scalar_in, mask, frm, vl, policy)

    SmallVector<llvm::Value*, 7> Operands;
    bool HasMaskedOff = !(
        (IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
    bool HasRoundModeOp = IsMasked ?
      (HasMaskedOff ? Ops.size() == 6 : Ops.size() == 5) :
      (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4);

    unsigned Offset = IsMasked ?
        (HasMaskedOff ? 2 : 1) : (HasMaskedOff ? 1 : 0);

    if (!HasMaskedOff)
      Operands.push_back(llvm::PoisonValue::get(ResultType));
    else
      Operands.push_back(Ops[IsMasked ? 1 : 0]);

    Operands.push_back(Ops[Offset]); // op0
    Operands.push_back(Ops[Offset + 1]); // op1

    if (IsMasked)
      Operands.push_back(Ops[0]); // mask

    if (HasRoundModeOp) {
      Operands.push_back(Ops[Offset + 2]); // frm
      Operands.push_back(Ops[Offset + 3]); // vl
    } else {
      Operands.push_back(ConstantInt::get(Ops[Offset + 2]->getType(), 7)); // frm
      Operands.push_back(Ops[Offset + 2]); // vl
    }

    if (IsMasked)
      Operands.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

    IntrinsicTypes = {ResultType, Ops[Offset]->getType(), Operands.back()->getType()};
    llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
    return Builder.CreateCall(F, Operands, "");
  }
break;
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_x_f_qf_rm_mu:
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_x_f_qf_mu:
  ID = Intrinsic::riscv_sf_vfnrclip_x_f_qf_mask;
  PolicyAttrs = 1;
  SegInstSEW = 0;
IsMasked = true;
if (SegInstSEW == (unsigned)-1) {
  auto PointeeType = E->getArg(4294967295      )->getType()->getPointeeType();
  SegInstSEW =       llvm::Log2_64(getContext().getTypeSize(PointeeType));
}

  {
    // LLVM intrinsic
    // Unmasked: (passthru, vector_in, scalar_in, frm, vl)
    // Masked:   (passthru, vector_in, scalar_in, mask, frm, vl, policy)

    SmallVector<llvm::Value*, 7> Operands;
    bool HasMaskedOff = !(
        (IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
    bool HasRoundModeOp = IsMasked ?
      (HasMaskedOff ? Ops.size() == 6 : Ops.size() == 5) :
      (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4);

    unsigned Offset = IsMasked ?
        (HasMaskedOff ? 2 : 1) : (HasMaskedOff ? 1 : 0);

    if (!HasMaskedOff)
      Operands.push_back(llvm::PoisonValue::get(ResultType));
    else
      Operands.push_back(Ops[IsMasked ? 1 : 0]);

    Operands.push_back(Ops[Offset]); // op0
    Operands.push_back(Ops[Offset + 1]); // op1

    if (IsMasked)
      Operands.push_back(Ops[0]); // mask

    if (HasRoundModeOp) {
      Operands.push_back(Ops[Offset + 2]); // frm
      Operands.push_back(Ops[Offset + 3]); // vl
    } else {
      Operands.push_back(ConstantInt::get(Ops[Offset + 2]->getType(), 7)); // frm
      Operands.push_back(Ops[Offset + 2]); // vl
    }

    if (IsMasked)
      Operands.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

    IntrinsicTypes = {ResultType, Ops[Offset]->getType(), Operands.back()->getType()};
    llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
    return Builder.CreateCall(F, Operands, "");
  }
break;
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_xu_f_qf_rm_tu:
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_xu_f_qf_tu:
  ID = Intrinsic::riscv_sf_vfnrclip_xu_f_qf;
  PolicyAttrs = 2;
  SegInstSEW = 0;
IsMasked = false;
if (SegInstSEW == (unsigned)-1) {
  auto PointeeType = E->getArg(4294967295      )->getType()->getPointeeType();
  SegInstSEW =       llvm::Log2_64(getContext().getTypeSize(PointeeType));
}

  {
    // LLVM intrinsic
    // Unmasked: (passthru, vector_in, scalar_in, frm, vl)
    // Masked:   (passthru, vector_in, scalar_in, mask, frm, vl, policy)

    SmallVector<llvm::Value*, 7> Operands;
    bool HasMaskedOff = !(
        (IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
    bool HasRoundModeOp = IsMasked ?
      (HasMaskedOff ? Ops.size() == 6 : Ops.size() == 5) :
      (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4);

    unsigned Offset = IsMasked ?
        (HasMaskedOff ? 2 : 1) : (HasMaskedOff ? 1 : 0);

    if (!HasMaskedOff)
      Operands.push_back(llvm::PoisonValue::get(ResultType));
    else
      Operands.push_back(Ops[IsMasked ? 1 : 0]);

    Operands.push_back(Ops[Offset]); // op0
    Operands.push_back(Ops[Offset + 1]); // op1

    if (IsMasked)
      Operands.push_back(Ops[0]); // mask

    if (HasRoundModeOp) {
      Operands.push_back(Ops[Offset + 2]); // frm
      Operands.push_back(Ops[Offset + 3]); // vl
    } else {
      Operands.push_back(ConstantInt::get(Ops[Offset + 2]->getType(), 7)); // frm
      Operands.push_back(Ops[Offset + 2]); // vl
    }

    if (IsMasked)
      Operands.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

    IntrinsicTypes = {ResultType, Ops[Offset]->getType(), Operands.back()->getType()};
    llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
    return Builder.CreateCall(F, Operands, "");
  }
break;
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_xu_f_qf_rm:
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_xu_f_qf:
  ID = Intrinsic::riscv_sf_vfnrclip_xu_f_qf;
  PolicyAttrs = 3;
  SegInstSEW = 0;
IsMasked = false;
if (SegInstSEW == (unsigned)-1) {
  auto PointeeType = E->getArg(4294967295      )->getType()->getPointeeType();
  SegInstSEW =       llvm::Log2_64(getContext().getTypeSize(PointeeType));
}

  {
    // LLVM intrinsic
    // Unmasked: (passthru, vector_in, scalar_in, frm, vl)
    // Masked:   (passthru, vector_in, scalar_in, mask, frm, vl, policy)

    SmallVector<llvm::Value*, 7> Operands;
    bool HasMaskedOff = !(
        (IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
    bool HasRoundModeOp = IsMasked ?
      (HasMaskedOff ? Ops.size() == 6 : Ops.size() == 5) :
      (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4);

    unsigned Offset = IsMasked ?
        (HasMaskedOff ? 2 : 1) : (HasMaskedOff ? 1 : 0);

    if (!HasMaskedOff)
      Operands.push_back(llvm::PoisonValue::get(ResultType));
    else
      Operands.push_back(Ops[IsMasked ? 1 : 0]);

    Operands.push_back(Ops[Offset]); // op0
    Operands.push_back(Ops[Offset + 1]); // op1

    if (IsMasked)
      Operands.push_back(Ops[0]); // mask

    if (HasRoundModeOp) {
      Operands.push_back(Ops[Offset + 2]); // frm
      Operands.push_back(Ops[Offset + 3]); // vl
    } else {
      Operands.push_back(ConstantInt::get(Ops[Offset + 2]->getType(), 7)); // frm
      Operands.push_back(Ops[Offset + 2]); // vl
    }

    if (IsMasked)
      Operands.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

    IntrinsicTypes = {ResultType, Ops[Offset]->getType(), Operands.back()->getType()};
    llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
    return Builder.CreateCall(F, Operands, "");
  }
break;
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_xu_f_qf_rm_tum:
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_xu_f_qf_tum:
  ID = Intrinsic::riscv_sf_vfnrclip_xu_f_qf_mask;
  PolicyAttrs = 2;
  SegInstSEW = 0;
IsMasked = true;
if (SegInstSEW == (unsigned)-1) {
  auto PointeeType = E->getArg(4294967295      )->getType()->getPointeeType();
  SegInstSEW =       llvm::Log2_64(getContext().getTypeSize(PointeeType));
}

  {
    // LLVM intrinsic
    // Unmasked: (passthru, vector_in, scalar_in, frm, vl)
    // Masked:   (passthru, vector_in, scalar_in, mask, frm, vl, policy)

    SmallVector<llvm::Value*, 7> Operands;
    bool HasMaskedOff = !(
        (IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
    bool HasRoundModeOp = IsMasked ?
      (HasMaskedOff ? Ops.size() == 6 : Ops.size() == 5) :
      (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4);

    unsigned Offset = IsMasked ?
        (HasMaskedOff ? 2 : 1) : (HasMaskedOff ? 1 : 0);

    if (!HasMaskedOff)
      Operands.push_back(llvm::PoisonValue::get(ResultType));
    else
      Operands.push_back(Ops[IsMasked ? 1 : 0]);

    Operands.push_back(Ops[Offset]); // op0
    Operands.push_back(Ops[Offset + 1]); // op1

    if (IsMasked)
      Operands.push_back(Ops[0]); // mask

    if (HasRoundModeOp) {
      Operands.push_back(Ops[Offset + 2]); // frm
      Operands.push_back(Ops[Offset + 3]); // vl
    } else {
      Operands.push_back(ConstantInt::get(Ops[Offset + 2]->getType(), 7)); // frm
      Operands.push_back(Ops[Offset + 2]); // vl
    }

    if (IsMasked)
      Operands.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

    IntrinsicTypes = {ResultType, Ops[Offset]->getType(), Operands.back()->getType()};
    llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
    return Builder.CreateCall(F, Operands, "");
  }
break;
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_xu_f_qf_rm_m:
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_xu_f_qf_m:
  ID = Intrinsic::riscv_sf_vfnrclip_xu_f_qf_mask;
  PolicyAttrs = 3;
  SegInstSEW = 0;
IsMasked = true;
if (SegInstSEW == (unsigned)-1) {
  auto PointeeType = E->getArg(4294967295      )->getType()->getPointeeType();
  SegInstSEW =       llvm::Log2_64(getContext().getTypeSize(PointeeType));
}

  {
    // LLVM intrinsic
    // Unmasked: (passthru, vector_in, scalar_in, frm, vl)
    // Masked:   (passthru, vector_in, scalar_in, mask, frm, vl, policy)

    SmallVector<llvm::Value*, 7> Operands;
    bool HasMaskedOff = !(
        (IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
    bool HasRoundModeOp = IsMasked ?
      (HasMaskedOff ? Ops.size() == 6 : Ops.size() == 5) :
      (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4);

    unsigned Offset = IsMasked ?
        (HasMaskedOff ? 2 : 1) : (HasMaskedOff ? 1 : 0);

    if (!HasMaskedOff)
      Operands.push_back(llvm::PoisonValue::get(ResultType));
    else
      Operands.push_back(Ops[IsMasked ? 1 : 0]);

    Operands.push_back(Ops[Offset]); // op0
    Operands.push_back(Ops[Offset + 1]); // op1

    if (IsMasked)
      Operands.push_back(Ops[0]); // mask

    if (HasRoundModeOp) {
      Operands.push_back(Ops[Offset + 2]); // frm
      Operands.push_back(Ops[Offset + 3]); // vl
    } else {
      Operands.push_back(ConstantInt::get(Ops[Offset + 2]->getType(), 7)); // frm
      Operands.push_back(Ops[Offset + 2]); // vl
    }

    if (IsMasked)
      Operands.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

    IntrinsicTypes = {ResultType, Ops[Offset]->getType(), Operands.back()->getType()};
    llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
    return Builder.CreateCall(F, Operands, "");
  }
break;
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_xu_f_qf_rm_tumu:
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_xu_f_qf_tumu:
  ID = Intrinsic::riscv_sf_vfnrclip_xu_f_qf_mask;
  PolicyAttrs = 0;
  SegInstSEW = 0;
IsMasked = true;
if (SegInstSEW == (unsigned)-1) {
  auto PointeeType = E->getArg(4294967295      )->getType()->getPointeeType();
  SegInstSEW =       llvm::Log2_64(getContext().getTypeSize(PointeeType));
}

  {
    // LLVM intrinsic
    // Unmasked: (passthru, vector_in, scalar_in, frm, vl)
    // Masked:   (passthru, vector_in, scalar_in, mask, frm, vl, policy)

    SmallVector<llvm::Value*, 7> Operands;
    bool HasMaskedOff = !(
        (IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
    bool HasRoundModeOp = IsMasked ?
      (HasMaskedOff ? Ops.size() == 6 : Ops.size() == 5) :
      (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4);

    unsigned Offset = IsMasked ?
        (HasMaskedOff ? 2 : 1) : (HasMaskedOff ? 1 : 0);

    if (!HasMaskedOff)
      Operands.push_back(llvm::PoisonValue::get(ResultType));
    else
      Operands.push_back(Ops[IsMasked ? 1 : 0]);

    Operands.push_back(Ops[Offset]); // op0
    Operands.push_back(Ops[Offset + 1]); // op1

    if (IsMasked)
      Operands.push_back(Ops[0]); // mask

    if (HasRoundModeOp) {
      Operands.push_back(Ops[Offset + 2]); // frm
      Operands.push_back(Ops[Offset + 3]); // vl
    } else {
      Operands.push_back(ConstantInt::get(Ops[Offset + 2]->getType(), 7)); // frm
      Operands.push_back(Ops[Offset + 2]); // vl
    }

    if (IsMasked)
      Operands.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

    IntrinsicTypes = {ResultType, Ops[Offset]->getType(), Operands.back()->getType()};
    llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
    return Builder.CreateCall(F, Operands, "");
  }
break;
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_xu_f_qf_rm_mu:
case RISCVVector::BI__builtin_rvv_sf_vfnrclip_xu_f_qf_mu:
  ID = Intrinsic::riscv_sf_vfnrclip_xu_f_qf_mask;
  PolicyAttrs = 1;
  SegInstSEW = 0;
IsMasked = true;
if (SegInstSEW == (unsigned)-1) {
  auto PointeeType = E->getArg(4294967295      )->getType()->getPointeeType();
  SegInstSEW =       llvm::Log2_64(getContext().getTypeSize(PointeeType));
}

  {
    // LLVM intrinsic
    // Unmasked: (passthru, vector_in, scalar_in, frm, vl)
    // Masked:   (passthru, vector_in, scalar_in, mask, frm, vl, policy)

    SmallVector<llvm::Value*, 7> Operands;
    bool HasMaskedOff = !(
        (IsMasked && (PolicyAttrs & RVV_VTA) && (PolicyAttrs & RVV_VMA)) ||
        (!IsMasked && PolicyAttrs & RVV_VTA));
    bool HasRoundModeOp = IsMasked ?
      (HasMaskedOff ? Ops.size() == 6 : Ops.size() == 5) :
      (HasMaskedOff ? Ops.size() == 5 : Ops.size() == 4);

    unsigned Offset = IsMasked ?
        (HasMaskedOff ? 2 : 1) : (HasMaskedOff ? 1 : 0);

    if (!HasMaskedOff)
      Operands.push_back(llvm::PoisonValue::get(ResultType));
    else
      Operands.push_back(Ops[IsMasked ? 1 : 0]);

    Operands.push_back(Ops[Offset]); // op0
    Operands.push_back(Ops[Offset + 1]); // op1

    if (IsMasked)
      Operands.push_back(Ops[0]); // mask

    if (HasRoundModeOp) {
      Operands.push_back(Ops[Offset + 2]); // frm
      Operands.push_back(Ops[Offset + 3]); // vl
    } else {
      Operands.push_back(ConstantInt::get(Ops[Offset + 2]->getType(), 7)); // frm
      Operands.push_back(Ops[Offset + 2]); // vl
    }

    if (IsMasked)
      Operands.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));

    IntrinsicTypes = {ResultType, Ops[Offset]->getType(), Operands.back()->getType()};
    llvm::Function *F = CGM.getIntrinsic(ID, IntrinsicTypes);
    return Builder.CreateCall(F, Operands, "");
  }
break;
case RISCVVector::BI__builtin_rvv_sf_vfwmacc_4x4x4_tu:
  ID = Intrinsic::riscv_sf_vfwmacc_4x4x4;
  PolicyAttrs = 2;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vfwmacc_4x4x4:
  ID = Intrinsic::riscv_sf_vfwmacc_4x4x4;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vqmacc_2x8x2_tu:
  ID = Intrinsic::riscv_sf_vqmacc_2x8x2;
  PolicyAttrs = 2;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vqmacc_2x8x2:
  ID = Intrinsic::riscv_sf_vqmacc_2x8x2;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vqmacc_4x8x4_tu:
  ID = Intrinsic::riscv_sf_vqmacc_4x8x4;
  PolicyAttrs = 2;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vqmacc_4x8x4:
  ID = Intrinsic::riscv_sf_vqmacc_4x8x4;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vqmaccsu_2x8x2_tu:
  ID = Intrinsic::riscv_sf_vqmaccsu_2x8x2;
  PolicyAttrs = 2;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vqmaccsu_2x8x2:
  ID = Intrinsic::riscv_sf_vqmaccsu_2x8x2;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vqmaccsu_4x8x4_tu:
  ID = Intrinsic::riscv_sf_vqmaccsu_4x8x4;
  PolicyAttrs = 2;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vqmaccsu_4x8x4:
  ID = Intrinsic::riscv_sf_vqmaccsu_4x8x4;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vqmaccu_2x8x2_tu:
  ID = Intrinsic::riscv_sf_vqmaccu_2x8x2;
  PolicyAttrs = 2;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vqmaccu_2x8x2:
  ID = Intrinsic::riscv_sf_vqmaccu_2x8x2;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vqmaccu_4x8x4_tu:
  ID = Intrinsic::riscv_sf_vqmaccu_4x8x4;
  PolicyAttrs = 2;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vqmaccu_4x8x4:
  ID = Intrinsic::riscv_sf_vqmaccu_4x8x4;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vqmaccus_2x8x2_tu:
  ID = Intrinsic::riscv_sf_vqmaccus_2x8x2;
  PolicyAttrs = 2;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vqmaccus_2x8x2:
  ID = Intrinsic::riscv_sf_vqmaccus_2x8x2;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vqmaccus_4x8x4_tu:
  ID = Intrinsic::riscv_sf_vqmaccus_4x8x4;
  PolicyAttrs = 2;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;
case RISCVVector::BI__builtin_rvv_sf_vqmaccus_4x8x4:
  ID = Intrinsic::riscv_sf_vqmaccus_4x8x4;
  PolicyAttrs = 3;
  SegInstSEW = 0;
  Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
  IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[2]->getType(), Ops.back()->getType()};
  break;

