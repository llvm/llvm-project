#ifdef GET_SME_IMMEDIATE_CHECK
case SME::BI__builtin_sme_svaddha_za32_s32_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svaddha_za32_u32_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svaddha_za64_s64_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svaddha_za64_u64_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svaddva_za32_s32_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svaddva_za32_u32_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svaddva_za64_s64_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svaddva_za64_u64_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svbmopa_za32_s32_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svbmopa_za32_u32_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svbmops_za32_s32_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svbmops_za32_u32_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za16_mf8_vg1x2_fpm:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za16_mf8_vg1x4_fpm:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za32_bf16_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za32_bf16_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za32_f16_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za32_f16_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za32_mf8_vg1x2_fpm:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za32_mf8_vg1x4_fpm:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za32_s16_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za32_s16_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za32_s8_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za32_s8_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za32_u16_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za32_u16_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za32_u8_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za32_u8_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za64_s16_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 14, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za64_s16_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 14, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za64_u16_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 14, 0));
  break;
case SME::BI__builtin_sme_svdot_lane_za64_u16_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 14, 0));
  break;
case SME::BI__builtin_sme_svld1_hor_vnum_za128:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svld1_hor_vnum_za16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svld1_hor_vnum_za32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svld1_hor_vnum_za64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svld1_hor_vnum_za8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svld1_hor_za128:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svld1_hor_za16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svld1_hor_za32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svld1_hor_za64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svld1_hor_za8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svld1_ver_vnum_za128:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svld1_ver_vnum_za16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svld1_ver_vnum_za32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svld1_ver_vnum_za64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svld1_ver_vnum_za8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svld1_ver_za128:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svld1_ver_za16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svld1_ver_za32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svld1_ver_za64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svld1_ver_za8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svldr_zt:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_bf16:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_bf16_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_bf16_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_f16:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_f16_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_f16_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_f32:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_f32_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_f32_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_mf8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_mf8_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_mf8_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_s16:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_s16_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_s16_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_s32:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_s32_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_s32_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_s8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_s8_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_s8_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_u16:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_u16_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_u16_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_u32:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_u32_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_u32_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_u8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_u8_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti2_lane_zt_u8_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_bf16:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_bf16_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_bf16_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 14, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_f16:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_f16_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_f16_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 14, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_f32:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_f32_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_f32_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 14, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_mf8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_mf8_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_s16:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_s16_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_s16_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 14, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_s32:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_s32_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_s32_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 14, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_s8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_s8_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_u16:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_u16_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_u16_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 14, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_u32:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_u32_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_u32_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 14, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_u8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svluti4_lane_zt_u8_x2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svluti4_zt_s8_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svluti4_zt_u8_x4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za16_bf16_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za16_bf16_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za16_f16_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za16_f16_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za16_mf8_vg2x1_fpm:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za16_mf8_vg2x2_fpm:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za16_mf8_vg2x4_fpm:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_bf16_vg2x1:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_bf16_vg2x2:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_bf16_vg2x4:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_f16_vg2x1:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_f16_vg2x2:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_f16_vg2x4:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_f32_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_f32_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_mf8_vg4x1_fpm:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_mf8_vg4x2_fpm:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_mf8_vg4x4_fpm:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_s16_vg2x1:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_s16_vg2x2:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_s16_vg2x4:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_s8_vg4x1:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_s8_vg4x2:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_s8_vg4x4:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_u16_vg2x1:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_u16_vg2x2:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_u16_vg2x4:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_u8_vg4x1:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_u8_vg4x2:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za32_u8_vg4x4:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za64_f64_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 14, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za64_f64_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 14, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za64_s16_vg4x1:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za64_s16_vg4x2:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za64_s16_vg4x4:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za64_u16_vg4x1:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za64_u16_vg4x2:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmla_lane_za64_u16_vg4x4:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za16_bf16_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za16_bf16_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za16_f16_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za16_f16_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_bf16_vg2x1:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_bf16_vg2x2:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_bf16_vg2x4:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_f16_vg2x1:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_f16_vg2x2:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_f16_vg2x4:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_f32_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_f32_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_s16_vg2x1:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_s16_vg2x2:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_s16_vg2x4:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_s8_vg4x1:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_s8_vg4x2:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_s8_vg4x4:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_u16_vg2x1:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_u16_vg2x2:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_u16_vg2x4:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_u8_vg4x1:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_u8_vg4x2:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za32_u8_vg4x4:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za64_f64_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 14, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za64_f64_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 14, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za64_s16_vg4x1:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za64_s16_vg4x2:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za64_s16_vg4x4:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za64_u16_vg4x1:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za64_u16_vg4x2:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmls_lane_za64_u16_vg4x4:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x1_za16_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x1_za16_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x1_za32_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x1_za32_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x1_za32_f32_f32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x1_za32_s16_s16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x1_za32_s8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x1_za32_s8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x1_za32_u16_u16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x1_za32_u8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x1_za32_u8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x1_za64_f64_f64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x1_za64_s16_s16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x1_za64_s16_u16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x1_za64_u16_s16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x1_za64_u16_u16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x2_za16_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x2_za16_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x2_za32_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x2_za32_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x2_za32_f32_f32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x2_za32_s16_s16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x2_za32_s8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x2_za32_s8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x2_za32_u16_u16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x2_za32_u8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x2_za32_u8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x2_za64_f64_f64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x2_za64_s16_s16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x2_za64_s16_u16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x2_za64_u16_s16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_1x2_za64_u16_u16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x1_za16_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x1_za16_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x1_za32_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x1_za32_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x1_za32_f32_f32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x1_za32_s16_s16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x1_za32_s8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x1_za32_s8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x1_za32_u16_u16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x1_za32_u8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x1_za32_u8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x1_za64_f64_f64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x1_za64_s16_s16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x1_za64_s16_u16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x1_za64_u16_s16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x1_za64_u16_u16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x2_za16_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x2_za16_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x2_za32_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x2_za32_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x2_za32_f32_f32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x2_za32_s16_s16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x2_za32_s8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x2_za32_s8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x2_za32_u16_u16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x2_za32_u8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x2_za32_u8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x2_za64_f64_f64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x2_za64_s16_s16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x2_za64_s16_u16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x2_za64_u16_s16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4a_2x2_za64_u16_u16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x1_za16_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x1_za16_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x1_za32_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x1_za32_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x1_za32_f32_f32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x1_za32_s16_s16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x1_za32_s8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x1_za32_s8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x1_za32_u16_u16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x1_za32_u8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x1_za32_u8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x1_za64_f64_f64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x1_za64_s16_s16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x1_za64_s16_u16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x1_za64_u16_s16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x1_za64_u16_u16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x2_za16_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x2_za16_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x2_za32_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x2_za32_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x2_za32_f32_f32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x2_za32_s16_s16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x2_za32_s8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x2_za32_s8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x2_za32_u16_u16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x2_za32_u8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x2_za32_u8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x2_za64_f64_f64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x2_za64_s16_s16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x2_za64_s16_u16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x2_za64_u16_s16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_1x2_za64_u16_u16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x1_za16_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x1_za16_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x1_za32_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x1_za32_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x1_za32_f32_f32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x1_za32_s16_s16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x1_za32_s8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x1_za32_s8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x1_za32_u16_u16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x1_za32_u8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x1_za32_u8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x1_za64_f64_f64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x1_za64_s16_s16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x1_za64_s16_u16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x1_za64_u16_s16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x1_za64_u16_u16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x2_za16_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x2_za16_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x2_za32_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x2_za32_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x2_za32_f32_f32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x2_za32_s16_s16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x2_za32_s8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x2_za32_s8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x2_za32_u16_u16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x2_za32_u8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x2_za32_u8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x2_za64_f64_f64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x2_za64_s16_s16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x2_za64_s16_u16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x2_za64_u16_s16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmop4s_2x2_za64_u16_u16:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmopa_za16_bf16_m:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmopa_za16_f16_m:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmopa_za16_mf8_m_fpm:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmopa_za32_bf16_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmopa_za32_f16_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmopa_za32_f32_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmopa_za32_mf8_m_fpm:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmopa_za32_s16_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmopa_za32_s8_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmopa_za32_u16_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmopa_za32_u8_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmopa_za64_f64_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmopa_za64_s16_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmopa_za64_u16_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmops_za16_bf16_m:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmops_za16_f16_m:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svmops_za32_bf16_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmops_za32_f16_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmops_za32_f32_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmops_za32_s16_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmops_za32_s8_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmops_za32_u16_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmops_za32_u8_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svmops_za64_f64_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmops_za64_s16_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svmops_za64_u16_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za128_bf16_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za128_f16_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za128_f32_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za128_f64_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za128_mf8_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za128_s16_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za128_s32_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za128_s64_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za128_s8_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za128_u16_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za128_u32_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za128_u64_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za128_u8_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za16_bf16_m:
ImmChecks.push_back(std::make_tuple(2, 14, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za16_bf16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za16_bf16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za16_f16_m:
ImmChecks.push_back(std::make_tuple(2, 14, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za16_f16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za16_f16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za16_s16_m:
ImmChecks.push_back(std::make_tuple(2, 14, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za16_s16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za16_s16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za16_u16_m:
ImmChecks.push_back(std::make_tuple(2, 14, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za16_u16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za16_u16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za32_f32_m:
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za32_f32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za32_f32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za32_s32_m:
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za32_s32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za32_s32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za32_u32_m:
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za32_u32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za32_u32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za64_f64_m:
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za64_f64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za64_f64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za64_s64_m:
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za64_s64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za64_s64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za64_u64_m:
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za64_u64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za64_u64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za8_mf8_m:
ImmChecks.push_back(std::make_tuple(2, 17, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za8_mf8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za8_mf8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za8_s8_m:
ImmChecks.push_back(std::make_tuple(2, 17, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za8_s8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za8_s8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za8_u8_m:
ImmChecks.push_back(std::make_tuple(2, 17, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za8_u8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svread_hor_za8_u8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za128_bf16_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za128_f16_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za128_f32_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za128_f64_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za128_mf8_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za128_s16_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za128_s32_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za128_s64_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za128_s8_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za128_u16_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za128_u32_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za128_u64_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za128_u8_m:
ImmChecks.push_back(std::make_tuple(2, 18, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za16_bf16_m:
ImmChecks.push_back(std::make_tuple(2, 14, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za16_bf16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za16_bf16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za16_f16_m:
ImmChecks.push_back(std::make_tuple(2, 14, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za16_f16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za16_f16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za16_s16_m:
ImmChecks.push_back(std::make_tuple(2, 14, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za16_s16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za16_s16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za16_u16_m:
ImmChecks.push_back(std::make_tuple(2, 14, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za16_u16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za16_u16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za32_f32_m:
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za32_f32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za32_f32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za32_s32_m:
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za32_s32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za32_s32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za32_u32_m:
ImmChecks.push_back(std::make_tuple(2, 16, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za32_u32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za32_u32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za64_f64_m:
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za64_f64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za64_f64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za64_s64_m:
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za64_s64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za64_s64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za64_u64_m:
ImmChecks.push_back(std::make_tuple(2, 6, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za64_u64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za64_u64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za8_mf8_m:
ImmChecks.push_back(std::make_tuple(2, 17, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za8_mf8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za8_mf8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za8_s8_m:
ImmChecks.push_back(std::make_tuple(2, 17, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za8_s8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za8_s8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za8_u8_m:
ImmChecks.push_back(std::make_tuple(2, 17, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za8_u8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svread_ver_za8_u8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za128_bf16:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za128_f16:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za128_f32:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za128_f64:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za128_mf8:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za128_s16:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za128_s32:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za128_s64:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za128_s8:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za128_u16:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za128_u32:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za128_u64:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za128_u8:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za16_bf16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za16_bf16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za16_bf16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za16_f16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za16_f16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za16_f16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za16_s16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za16_s16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za16_s16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za16_u16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za16_u16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za16_u16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za32_f32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za32_f32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za32_f32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za32_s32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za32_s32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za32_s32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za32_u32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za32_u32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za32_u32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za64_f64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za64_f64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za64_f64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za64_s64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za64_s64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za64_s64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za64_u64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za64_u64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za64_u64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za8_mf8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za8_mf8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za8_mf8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za8_s8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za8_s8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za8_s8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za8_u8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za8_u8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_hor_za8_u8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za128_bf16:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za128_f16:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za128_f32:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za128_f64:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za128_mf8:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za128_s16:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za128_s32:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za128_s64:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za128_s8:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za128_u16:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za128_u32:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za128_u64:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za128_u8:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za16_bf16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za16_bf16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za16_bf16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za16_f16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za16_f16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za16_f16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za16_s16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za16_s16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za16_s16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za16_u16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za16_u16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za16_u16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za32_f32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za32_f32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za32_f32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za32_s32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za32_s32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za32_s32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za32_u32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za32_u32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za32_u32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za64_f64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za64_f64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za64_f64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za64_s64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za64_s64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za64_s64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za64_u64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za64_u64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za64_u64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za8_mf8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za8_mf8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za8_mf8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za8_s8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za8_s8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za8_s8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za8_u8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za8_u8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svreadz_ver_za8_u8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svst1_hor_vnum_za128:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svst1_hor_vnum_za16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svst1_hor_vnum_za32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svst1_hor_vnum_za64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svst1_hor_vnum_za8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svst1_hor_za128:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svst1_hor_za16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svst1_hor_za32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svst1_hor_za64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svst1_hor_za8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svst1_ver_vnum_za128:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svst1_ver_vnum_za16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svst1_ver_vnum_za32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svst1_ver_vnum_za64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svst1_ver_vnum_za8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svst1_ver_za128:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svst1_ver_za16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svst1_ver_za32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svst1_ver_za64:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svst1_ver_za8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svstr_zt:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svsudot_lane_za32_s8_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svsudot_lane_za32_s8_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svsumla_lane_za32_s8_vg4x1:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svsumla_lane_za32_s8_vg4x2:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svsumla_lane_za32_s8_vg4x4:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svsumopa_za32_s8_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svsumopa_za64_s16_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svsumops_za32_s8_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svsumops_za64_s16_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svsuvdot_lane_za32_s8_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svtmopa_lane_za16_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
ImmChecks.push_back(std::make_tuple(4, 16, 0));
  break;
case SME::BI__builtin_sme_svtmopa_lane_za16_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
ImmChecks.push_back(std::make_tuple(4, 16, 0));
  break;
case SME::BI__builtin_sme_svtmopa_lane_za16_mf8_mf8_fpm:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
ImmChecks.push_back(std::make_tuple(4, 16, 0));
  break;
case SME::BI__builtin_sme_svtmopa_lane_za32_bf16_bf16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
ImmChecks.push_back(std::make_tuple(4, 16, 0));
  break;
case SME::BI__builtin_sme_svtmopa_lane_za32_f16_f16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
ImmChecks.push_back(std::make_tuple(4, 16, 0));
  break;
case SME::BI__builtin_sme_svtmopa_lane_za32_f32_f32:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
ImmChecks.push_back(std::make_tuple(4, 16, 0));
  break;
case SME::BI__builtin_sme_svtmopa_lane_za32_mf8_mf8_fpm:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
ImmChecks.push_back(std::make_tuple(4, 16, 0));
  break;
case SME::BI__builtin_sme_svtmopa_lane_za32_s16_s16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
ImmChecks.push_back(std::make_tuple(4, 16, 0));
  break;
case SME::BI__builtin_sme_svtmopa_lane_za32_s8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
ImmChecks.push_back(std::make_tuple(4, 16, 0));
  break;
case SME::BI__builtin_sme_svtmopa_lane_za32_s8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
ImmChecks.push_back(std::make_tuple(4, 16, 0));
  break;
case SME::BI__builtin_sme_svtmopa_lane_za32_u16_u16:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
ImmChecks.push_back(std::make_tuple(4, 16, 0));
  break;
case SME::BI__builtin_sme_svtmopa_lane_za32_u8_s8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
ImmChecks.push_back(std::make_tuple(4, 16, 0));
  break;
case SME::BI__builtin_sme_svtmopa_lane_za32_u8_u8:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
ImmChecks.push_back(std::make_tuple(4, 16, 0));
  break;
case SME::BI__builtin_sme_svusdot_lane_za32_u8_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svusdot_lane_za32_u8_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svusmla_lane_za32_u8_vg4x1:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svusmla_lane_za32_u8_vg4x2:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svusmla_lane_za32_u8_vg4x4:
ImmChecks.push_back(std::make_tuple(3, 18, 0));
  break;
case SME::BI__builtin_sme_svusmopa_za32_u8_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svusmopa_za64_u16_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svusmops_za32_u8_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svusmops_za64_u16_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svusvdot_lane_za32_u8_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svvdot_lane_za16_mf8_vg1x2_fpm:
ImmChecks.push_back(std::make_tuple(3, 6, 0));
  break;
case SME::BI__builtin_sme_svvdot_lane_za32_bf16_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svvdot_lane_za32_f16_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svvdot_lane_za32_s16_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svvdot_lane_za32_s8_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svvdot_lane_za32_u16_vg1x2:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svvdot_lane_za32_u8_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svvdot_lane_za64_s16_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 14, 0));
  break;
case SME::BI__builtin_sme_svvdot_lane_za64_u16_vg1x4:
ImmChecks.push_back(std::make_tuple(3, 14, 0));
  break;
case SME::BI__builtin_sme_svvdotb_lane_za32_mf8_vg1x4_fpm:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svvdott_lane_za32_mf8_vg1x4_fpm:
ImmChecks.push_back(std::make_tuple(3, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za128_bf16_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za128_f16_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za128_f32_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za128_f64_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za128_mf8_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za128_s16_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za128_s32_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za128_s64_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za128_s8_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za128_u16_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za128_u32_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za128_u64_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za128_u8_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za16_bf16_m:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za16_bf16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za16_bf16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za16_f16_m:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za16_f16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za16_f16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za16_s16_m:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za16_s16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za16_s16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za16_u16_m:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za16_u16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za16_u16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za32_f32_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za32_f32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za32_f32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za32_s32_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za32_s32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za32_s32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za32_u32_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za32_u32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za32_u32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za64_f64_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za64_f64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za64_f64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za64_s64_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za64_s64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za64_s64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za64_u64_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za64_u64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za64_u64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za8_mf8_m:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za8_mf8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za8_mf8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za8_s8_m:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za8_s8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za8_s8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za8_u8_m:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za8_u8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_hor_za8_u8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_lane_zt_bf16:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 22, 0));
  break;
case SME::BI__builtin_sme_svwrite_lane_zt_f16:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 22, 0));
  break;
case SME::BI__builtin_sme_svwrite_lane_zt_f32:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 22, 0));
  break;
case SME::BI__builtin_sme_svwrite_lane_zt_f64:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 22, 0));
  break;
case SME::BI__builtin_sme_svwrite_lane_zt_s16:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 22, 0));
  break;
case SME::BI__builtin_sme_svwrite_lane_zt_s32:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 22, 0));
  break;
case SME::BI__builtin_sme_svwrite_lane_zt_s64:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 22, 0));
  break;
case SME::BI__builtin_sme_svwrite_lane_zt_s8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 22, 0));
  break;
case SME::BI__builtin_sme_svwrite_lane_zt_u16:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 22, 0));
  break;
case SME::BI__builtin_sme_svwrite_lane_zt_u32:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 22, 0));
  break;
case SME::BI__builtin_sme_svwrite_lane_zt_u64:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 22, 0));
  break;
case SME::BI__builtin_sme_svwrite_lane_zt_u8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
ImmChecks.push_back(std::make_tuple(2, 22, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za128_bf16_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za128_f16_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za128_f32_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za128_f64_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za128_mf8_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za128_s16_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za128_s32_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za128_s64_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za128_s8_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za128_u16_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za128_u32_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za128_u64_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za128_u8_m:
ImmChecks.push_back(std::make_tuple(0, 18, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za16_bf16_m:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za16_bf16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za16_bf16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za16_f16_m:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za16_f16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za16_f16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za16_s16_m:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za16_s16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za16_s16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za16_u16_m:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za16_u16_vg2:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za16_u16_vg4:
ImmChecks.push_back(std::make_tuple(0, 14, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za32_f32_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za32_f32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za32_f32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za32_s32_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za32_s32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za32_s32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za32_u32_m:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za32_u32_vg2:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za32_u32_vg4:
ImmChecks.push_back(std::make_tuple(0, 16, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za64_f64_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za64_f64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za64_f64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za64_s64_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za64_s64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za64_s64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za64_u64_m:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za64_u64_vg2:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za64_u64_vg4:
ImmChecks.push_back(std::make_tuple(0, 6, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za8_mf8_m:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za8_mf8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za8_mf8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za8_s8_m:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za8_s8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za8_s8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za8_u8_m:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za8_u8_vg2:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_ver_za8_u8_vg4:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_zt_bf16:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_zt_f16:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_zt_f32:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_zt_f64:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_zt_s16:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_zt_s32:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_zt_s64:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_zt_s8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_zt_u16:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_zt_u32:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_zt_u64:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svwrite_zt_u8:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
case SME::BI__builtin_sme_svzero_mask_za:
ImmChecks.push_back(std::make_tuple(0, 19, 0));
  break;
case SME::BI__builtin_sme_svzero_zt:
ImmChecks.push_back(std::make_tuple(0, 17, 0));
  break;
#endif

