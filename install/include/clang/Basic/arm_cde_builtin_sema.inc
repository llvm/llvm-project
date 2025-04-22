case ARM::BI__builtin_arm_cde_cx1:
case ARM::BI__builtin_arm_cde_cx1d:
  Err = SemaRef.BuiltinConstantArgRange(TheCall, 0, 0x0, 0x7) ||
         SemaRef.BuiltinConstantArgRange(TheCall, 1, 0x0, 0x1FFF);
  break;
case ARM::BI__builtin_arm_cde_vcx1_u32:
case ARM::BI__builtin_arm_cde_vcx1d_u64:
  Err = SemaRef.BuiltinConstantArgRange(TheCall, 0, 0x0, 0x7) ||
         SemaRef.BuiltinConstantArgRange(TheCall, 1, 0x0, 0x7FF);
  break;
case ARM::BI__builtin_arm_cde_vcx1q_u8:
  Err = SemaRef.BuiltinConstantArgRange(TheCall, 0, 0x0, 0x7) ||
         SemaRef.BuiltinConstantArgRange(TheCall, 1, 0x0, 0xFFF);
  break;
case ARM::BI__builtin_arm_cde_cx2:
case ARM::BI__builtin_arm_cde_cx2d:
  Err = SemaRef.BuiltinConstantArgRange(TheCall, 0, 0x0, 0x7) ||
         SemaRef.BuiltinConstantArgRange(TheCall, 2, 0x0, 0x1FF);
  break;
case ARM::BI__builtin_arm_cde_cx1a:
case ARM::BI__builtin_arm_cde_cx1da:
  Err = SemaRef.BuiltinConstantArgRange(TheCall, 0, 0x0, 0x7) ||
         SemaRef.BuiltinConstantArgRange(TheCall, 2, 0x0, 0x1FFF);
  break;
case ARM::BI__builtin_arm_cde_vcx2_u32:
case ARM::BI__builtin_arm_cde_vcx2d_u64:
  Err = SemaRef.BuiltinConstantArgRange(TheCall, 0, 0x0, 0x7) ||
         SemaRef.BuiltinConstantArgRange(TheCall, 2, 0x0, 0x3F);
  break;
case ARM::BI__builtin_arm_cde_vcx2q_f16:
case ARM::BI__builtin_arm_cde_vcx2q_f32:
case ARM::BI__builtin_arm_cde_vcx2q_s16:
case ARM::BI__builtin_arm_cde_vcx2q_s32:
case ARM::BI__builtin_arm_cde_vcx2q_s64:
case ARM::BI__builtin_arm_cde_vcx2q_s8:
case ARM::BI__builtin_arm_cde_vcx2q_u16:
case ARM::BI__builtin_arm_cde_vcx2q_u32:
case ARM::BI__builtin_arm_cde_vcx2q_u64:
case ARM::BI__builtin_arm_cde_vcx2q_u8:
case ARM::BI__builtin_arm_cde_vcx2q_u8_f16:
case ARM::BI__builtin_arm_cde_vcx2q_u8_f32:
case ARM::BI__builtin_arm_cde_vcx2q_u8_s16:
case ARM::BI__builtin_arm_cde_vcx2q_u8_s32:
case ARM::BI__builtin_arm_cde_vcx2q_u8_s64:
case ARM::BI__builtin_arm_cde_vcx2q_u8_s8:
case ARM::BI__builtin_arm_cde_vcx2q_u8_u16:
case ARM::BI__builtin_arm_cde_vcx2q_u8_u32:
case ARM::BI__builtin_arm_cde_vcx2q_u8_u64:
case ARM::BI__builtin_arm_cde_vcx2q_u8_u8:
  Err = SemaRef.BuiltinConstantArgRange(TheCall, 0, 0x0, 0x7) ||
         SemaRef.BuiltinConstantArgRange(TheCall, 2, 0x0, 0x7F);
  break;
case ARM::BI__builtin_arm_cde_vcx1a_u32:
case ARM::BI__builtin_arm_cde_vcx1da_u64:
  Err = SemaRef.BuiltinConstantArgRange(TheCall, 0, 0x0, 0x7) ||
         SemaRef.BuiltinConstantArgRange(TheCall, 2, 0x0, 0x7FF);
  break;
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
case ARM::BI__builtin_arm_cde_vcx1qa_f16:
case ARM::BI__builtin_arm_cde_vcx1qa_f32:
case ARM::BI__builtin_arm_cde_vcx1qa_m_f16:
case ARM::BI__builtin_arm_cde_vcx1qa_m_f32:
case ARM::BI__builtin_arm_cde_vcx1qa_m_s16:
case ARM::BI__builtin_arm_cde_vcx1qa_m_s32:
case ARM::BI__builtin_arm_cde_vcx1qa_m_s64:
case ARM::BI__builtin_arm_cde_vcx1qa_m_s8:
case ARM::BI__builtin_arm_cde_vcx1qa_m_u16:
case ARM::BI__builtin_arm_cde_vcx1qa_m_u32:
case ARM::BI__builtin_arm_cde_vcx1qa_m_u64:
case ARM::BI__builtin_arm_cde_vcx1qa_m_u8:
case ARM::BI__builtin_arm_cde_vcx1qa_s16:
case ARM::BI__builtin_arm_cde_vcx1qa_s32:
case ARM::BI__builtin_arm_cde_vcx1qa_s64:
case ARM::BI__builtin_arm_cde_vcx1qa_s8:
case ARM::BI__builtin_arm_cde_vcx1qa_u16:
case ARM::BI__builtin_arm_cde_vcx1qa_u32:
case ARM::BI__builtin_arm_cde_vcx1qa_u64:
case ARM::BI__builtin_arm_cde_vcx1qa_u8:
  Err = SemaRef.BuiltinConstantArgRange(TheCall, 0, 0x0, 0x7) ||
         SemaRef.BuiltinConstantArgRange(TheCall, 2, 0x0, 0xFFF);
  break;
case ARM::BI__builtin_arm_cde_cx2a:
case ARM::BI__builtin_arm_cde_cx2da:
  Err = SemaRef.BuiltinConstantArgRange(TheCall, 0, 0x0, 0x7) ||
         SemaRef.BuiltinConstantArgRange(TheCall, 3, 0x0, 0x1FF);
  break;
case ARM::BI__builtin_arm_cde_cx3:
case ARM::BI__builtin_arm_cde_cx3d:
case ARM::BI__builtin_arm_cde_vcx2a_u32:
case ARM::BI__builtin_arm_cde_vcx2da_u64:
  Err = SemaRef.BuiltinConstantArgRange(TheCall, 0, 0x0, 0x7) ||
         SemaRef.BuiltinConstantArgRange(TheCall, 3, 0x0, 0x3F);
  break;
case ARM::BI__builtin_arm_cde_vcx3_u32:
case ARM::BI__builtin_arm_cde_vcx3d_u64:
  Err = SemaRef.BuiltinConstantArgRange(TheCall, 0, 0x0, 0x7) ||
         SemaRef.BuiltinConstantArgRange(TheCall, 3, 0x0, 0x7);
  break;
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
case ARM::BI__builtin_arm_cde_vcx2qa_impl_f16:
case ARM::BI__builtin_arm_cde_vcx2qa_impl_f32:
case ARM::BI__builtin_arm_cde_vcx2qa_impl_s16:
case ARM::BI__builtin_arm_cde_vcx2qa_impl_s32:
case ARM::BI__builtin_arm_cde_vcx2qa_impl_s64:
case ARM::BI__builtin_arm_cde_vcx2qa_impl_s8:
case ARM::BI__builtin_arm_cde_vcx2qa_impl_u16:
case ARM::BI__builtin_arm_cde_vcx2qa_impl_u32:
case ARM::BI__builtin_arm_cde_vcx2qa_impl_u64:
case ARM::BI__builtin_arm_cde_vcx2qa_impl_u8:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_f16:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_f32:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_s16:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_s32:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_s64:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_s8:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_u16:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_u32:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_u64:
case ARM::BI__builtin_arm_cde_vcx2qa_m_impl_u8:
  Err = SemaRef.BuiltinConstantArgRange(TheCall, 0, 0x0, 0x7) ||
         SemaRef.BuiltinConstantArgRange(TheCall, 3, 0x0, 0x7F);
  break;
case ARM::BI__builtin_arm_cde_vcx3q_impl_f16:
case ARM::BI__builtin_arm_cde_vcx3q_impl_f32:
case ARM::BI__builtin_arm_cde_vcx3q_impl_s16:
case ARM::BI__builtin_arm_cde_vcx3q_impl_s32:
case ARM::BI__builtin_arm_cde_vcx3q_impl_s64:
case ARM::BI__builtin_arm_cde_vcx3q_impl_s8:
case ARM::BI__builtin_arm_cde_vcx3q_impl_u16:
case ARM::BI__builtin_arm_cde_vcx3q_impl_u32:
case ARM::BI__builtin_arm_cde_vcx3q_impl_u64:
case ARM::BI__builtin_arm_cde_vcx3q_impl_u8:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_f16:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_f32:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_s16:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_s32:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_s64:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_s8:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_u16:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_u32:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_u64:
case ARM::BI__builtin_arm_cde_vcx3q_u8_impl_u8:
  Err = SemaRef.BuiltinConstantArgRange(TheCall, 0, 0x0, 0x7) ||
         SemaRef.BuiltinConstantArgRange(TheCall, 3, 0x0, 0xF);
  break;
case ARM::BI__builtin_arm_cde_cx3a:
case ARM::BI__builtin_arm_cde_cx3da:
  Err = SemaRef.BuiltinConstantArgRange(TheCall, 0, 0x0, 0x7) ||
         SemaRef.BuiltinConstantArgRange(TheCall, 4, 0x0, 0x3F);
  break;
case ARM::BI__builtin_arm_cde_vcx3a_u32:
case ARM::BI__builtin_arm_cde_vcx3da_u64:
  Err = SemaRef.BuiltinConstantArgRange(TheCall, 0, 0x0, 0x7) ||
         SemaRef.BuiltinConstantArgRange(TheCall, 4, 0x0, 0x7);
  break;
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
case ARM::BI__builtin_arm_cde_vcx3qa_impl_f16:
case ARM::BI__builtin_arm_cde_vcx3qa_impl_f32:
case ARM::BI__builtin_arm_cde_vcx3qa_impl_s16:
case ARM::BI__builtin_arm_cde_vcx3qa_impl_s32:
case ARM::BI__builtin_arm_cde_vcx3qa_impl_s64:
case ARM::BI__builtin_arm_cde_vcx3qa_impl_s8:
case ARM::BI__builtin_arm_cde_vcx3qa_impl_u16:
case ARM::BI__builtin_arm_cde_vcx3qa_impl_u32:
case ARM::BI__builtin_arm_cde_vcx3qa_impl_u64:
case ARM::BI__builtin_arm_cde_vcx3qa_impl_u8:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_f16:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_f32:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_s16:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_s32:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_s64:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_s8:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_u16:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_u32:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_u64:
case ARM::BI__builtin_arm_cde_vcx3qa_m_impl_u8:
  Err = SemaRef.BuiltinConstantArgRange(TheCall, 0, 0x0, 0x7) ||
         SemaRef.BuiltinConstantArgRange(TheCall, 4, 0x0, 0xF);
  break;
