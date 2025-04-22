static const IntrinToName MapData[] = {
  { ARM::BI__builtin_arm_cde_cx1, 1, -1},
  { ARM::BI__builtin_arm_cde_cx1a, 5, -1},
  { ARM::BI__builtin_arm_cde_cx1d, 10, -1},
  { ARM::BI__builtin_arm_cde_cx1da, 15, -1},
  { ARM::BI__builtin_arm_cde_cx2, 21, -1},
  { ARM::BI__builtin_arm_cde_cx2a, 25, -1},
  { ARM::BI__builtin_arm_cde_cx2d, 30, -1},
  { ARM::BI__builtin_arm_cde_cx2da, 35, -1},
  { ARM::BI__builtin_arm_cde_cx3, 41, -1},
  { ARM::BI__builtin_arm_cde_cx3a, 45, -1},
  { ARM::BI__builtin_arm_cde_cx3d, 50, -1},
  { ARM::BI__builtin_arm_cde_cx3da, 55, -1},
  { ARM::BI__builtin_arm_cde_vcx1_u32, 61, -1},
  { ARM::BI__builtin_arm_cde_vcx1a_u32, 70, -1},
  { ARM::BI__builtin_arm_cde_vcx1d_u64, 80, -1},
  { ARM::BI__builtin_arm_cde_vcx1da_u64, 90, -1},
  { ARM::BI__builtin_arm_cde_vcx1q_m_f16, 109, 101},
  { ARM::BI__builtin_arm_cde_vcx1q_m_f32, 121, 101},
  { ARM::BI__builtin_arm_cde_vcx1q_m_s16, 133, 101},
  { ARM::BI__builtin_arm_cde_vcx1q_m_s32, 145, 101},
  { ARM::BI__builtin_arm_cde_vcx1q_m_s64, 157, 101},
  { ARM::BI__builtin_arm_cde_vcx1q_m_s8, 169, 101},
  { ARM::BI__builtin_arm_cde_vcx1q_m_u16, 180, 101},
  { ARM::BI__builtin_arm_cde_vcx1q_m_u32, 192, 101},
  { ARM::BI__builtin_arm_cde_vcx1q_m_u64, 204, 101},
  { ARM::BI__builtin_arm_cde_vcx1q_m_u8, 216, 101},
  { ARM::BI__builtin_arm_cde_vcx1q_u8, 227, -1},
  { ARM::BI__builtin_arm_cde_vcx1qa_f16, 243, 236},
  { ARM::BI__builtin_arm_cde_vcx1qa_f32, 254, 236},
  { ARM::BI__builtin_arm_cde_vcx1qa_m_f16, 274, 265},
  { ARM::BI__builtin_arm_cde_vcx1qa_m_f32, 287, 265},
  { ARM::BI__builtin_arm_cde_vcx1qa_m_s16, 300, 265},
  { ARM::BI__builtin_arm_cde_vcx1qa_m_s32, 313, 265},
  { ARM::BI__builtin_arm_cde_vcx1qa_m_s64, 326, 265},
  { ARM::BI__builtin_arm_cde_vcx1qa_m_s8, 339, 265},
  { ARM::BI__builtin_arm_cde_vcx1qa_m_u16, 351, 265},
  { ARM::BI__builtin_arm_cde_vcx1qa_m_u32, 364, 265},
  { ARM::BI__builtin_arm_cde_vcx1qa_m_u64, 377, 265},
  { ARM::BI__builtin_arm_cde_vcx1qa_m_u8, 390, 265},
  { ARM::BI__builtin_arm_cde_vcx1qa_s16, 402, 236},
  { ARM::BI__builtin_arm_cde_vcx1qa_s32, 413, 236},
  { ARM::BI__builtin_arm_cde_vcx1qa_s64, 424, 236},
  { ARM::BI__builtin_arm_cde_vcx1qa_s8, 435, 236},
  { ARM::BI__builtin_arm_cde_vcx1qa_u16, 445, 236},
  { ARM::BI__builtin_arm_cde_vcx1qa_u32, 456, 236},
  { ARM::BI__builtin_arm_cde_vcx1qa_u64, 467, 236},
  { ARM::BI__builtin_arm_cde_vcx1qa_u8, 478, 236},
  { ARM::BI__builtin_arm_cde_vcx2_u32, 488, -1},
  { ARM::BI__builtin_arm_cde_vcx2a_u32, 497, -1},
  { ARM::BI__builtin_arm_cde_vcx2d_u64, 507, -1},
  { ARM::BI__builtin_arm_cde_vcx2da_u64, 517, -1},
  { ARM::BI__builtin_arm_cde_vcx2q_f16, 534, 528},
  { ARM::BI__builtin_arm_cde_vcx2q_f32, 544, 528},
  { ARM::BI__builtin_arm_cde_vcx2q_m_impl_f16, 567, 554},
  { ARM::BI__builtin_arm_cde_vcx2q_m_impl_f32, 584, 554},
  { ARM::BI__builtin_arm_cde_vcx2q_m_impl_s16, 601, 554},
  { ARM::BI__builtin_arm_cde_vcx2q_m_impl_s32, 618, 554},
  { ARM::BI__builtin_arm_cde_vcx2q_m_impl_s64, 635, 554},
  { ARM::BI__builtin_arm_cde_vcx2q_m_impl_s8, 652, 554},
  { ARM::BI__builtin_arm_cde_vcx2q_m_impl_u16, 668, 554},
  { ARM::BI__builtin_arm_cde_vcx2q_m_impl_u32, 685, 554},
  { ARM::BI__builtin_arm_cde_vcx2q_m_impl_u64, 702, 554},
  { ARM::BI__builtin_arm_cde_vcx2q_m_impl_u8, 719, 554},
  { ARM::BI__builtin_arm_cde_vcx2q_s16, 735, 528},
  { ARM::BI__builtin_arm_cde_vcx2q_s32, 745, 528},
  { ARM::BI__builtin_arm_cde_vcx2q_s64, 755, 528},
  { ARM::BI__builtin_arm_cde_vcx2q_s8, 765, 528},
  { ARM::BI__builtin_arm_cde_vcx2q_u16, 774, 528},
  { ARM::BI__builtin_arm_cde_vcx2q_u32, 784, 528},
  { ARM::BI__builtin_arm_cde_vcx2q_u64, 794, 528},
  { ARM::BI__builtin_arm_cde_vcx2q_u8, 804, 528},
  { ARM::BI__builtin_arm_cde_vcx2q_u8_f16, 813, 804},
  { ARM::BI__builtin_arm_cde_vcx2q_u8_f32, 826, 804},
  { ARM::BI__builtin_arm_cde_vcx2q_u8_s16, 839, 804},
  { ARM::BI__builtin_arm_cde_vcx2q_u8_s32, 852, 804},
  { ARM::BI__builtin_arm_cde_vcx2q_u8_s64, 865, 804},
  { ARM::BI__builtin_arm_cde_vcx2q_u8_s8, 878, 804},
  { ARM::BI__builtin_arm_cde_vcx2q_u8_u16, 890, 804},
  { ARM::BI__builtin_arm_cde_vcx2q_u8_u32, 903, 804},
  { ARM::BI__builtin_arm_cde_vcx2q_u8_u64, 916, 804},
  { ARM::BI__builtin_arm_cde_vcx2q_u8_u8, 929, 804},
  { ARM::BI__builtin_arm_cde_vcx2qa_impl_f16, 953, 941},
  { ARM::BI__builtin_arm_cde_vcx2qa_impl_f32, 969, 941},
  { ARM::BI__builtin_arm_cde_vcx2qa_impl_s16, 985, 941},
  { ARM::BI__builtin_arm_cde_vcx2qa_impl_s32, 1001, 941},
  { ARM::BI__builtin_arm_cde_vcx2qa_impl_s64, 1017, 941},
  { ARM::BI__builtin_arm_cde_vcx2qa_impl_s8, 1033, 941},
  { ARM::BI__builtin_arm_cde_vcx2qa_impl_u16, 1048, 941},
  { ARM::BI__builtin_arm_cde_vcx2qa_impl_u32, 1064, 941},
  { ARM::BI__builtin_arm_cde_vcx2qa_impl_u64, 1080, 941},
  { ARM::BI__builtin_arm_cde_vcx2qa_impl_u8, 1096, 941},
  { ARM::BI__builtin_arm_cde_vcx2qa_m_impl_f16, 1125, 1111},
  { ARM::BI__builtin_arm_cde_vcx2qa_m_impl_f32, 1143, 1111},
  { ARM::BI__builtin_arm_cde_vcx2qa_m_impl_s16, 1161, 1111},
  { ARM::BI__builtin_arm_cde_vcx2qa_m_impl_s32, 1179, 1111},
  { ARM::BI__builtin_arm_cde_vcx2qa_m_impl_s64, 1197, 1111},
  { ARM::BI__builtin_arm_cde_vcx2qa_m_impl_s8, 1215, 1111},
  { ARM::BI__builtin_arm_cde_vcx2qa_m_impl_u16, 1232, 1111},
  { ARM::BI__builtin_arm_cde_vcx2qa_m_impl_u32, 1250, 1111},
  { ARM::BI__builtin_arm_cde_vcx2qa_m_impl_u64, 1268, 1111},
  { ARM::BI__builtin_arm_cde_vcx2qa_m_impl_u8, 1286, 1111},
  { ARM::BI__builtin_arm_cde_vcx3_u32, 1303, -1},
  { ARM::BI__builtin_arm_cde_vcx3a_u32, 1312, -1},
  { ARM::BI__builtin_arm_cde_vcx3d_u64, 1322, -1},
  { ARM::BI__builtin_arm_cde_vcx3da_u64, 1332, -1},
  { ARM::BI__builtin_arm_cde_vcx3q_impl_f16, 1354, 1343},
  { ARM::BI__builtin_arm_cde_vcx3q_impl_f32, 1369, 1343},
  { ARM::BI__builtin_arm_cde_vcx3q_impl_s16, 1384, 1343},
  { ARM::BI__builtin_arm_cde_vcx3q_impl_s32, 1399, 1343},
  { ARM::BI__builtin_arm_cde_vcx3q_impl_s64, 1414, 1343},
  { ARM::BI__builtin_arm_cde_vcx3q_impl_s8, 1429, 1343},
  { ARM::BI__builtin_arm_cde_vcx3q_impl_u16, 1443, 1343},
  { ARM::BI__builtin_arm_cde_vcx3q_impl_u32, 1458, 1343},
  { ARM::BI__builtin_arm_cde_vcx3q_impl_u64, 1473, 1343},
  { ARM::BI__builtin_arm_cde_vcx3q_impl_u8, 1488, 1343},
  { ARM::BI__builtin_arm_cde_vcx3q_m_impl_f16, 1515, 1502},
  { ARM::BI__builtin_arm_cde_vcx3q_m_impl_f32, 1532, 1502},
  { ARM::BI__builtin_arm_cde_vcx3q_m_impl_s16, 1549, 1502},
  { ARM::BI__builtin_arm_cde_vcx3q_m_impl_s32, 1566, 1502},
  { ARM::BI__builtin_arm_cde_vcx3q_m_impl_s64, 1583, 1502},
  { ARM::BI__builtin_arm_cde_vcx3q_m_impl_s8, 1600, 1502},
  { ARM::BI__builtin_arm_cde_vcx3q_m_impl_u16, 1616, 1502},
  { ARM::BI__builtin_arm_cde_vcx3q_m_impl_u32, 1633, 1502},
  { ARM::BI__builtin_arm_cde_vcx3q_m_impl_u64, 1650, 1502},
  { ARM::BI__builtin_arm_cde_vcx3q_m_impl_u8, 1667, 1502},
  { ARM::BI__builtin_arm_cde_vcx3q_u8_impl_f16, 1697, 1683},
  { ARM::BI__builtin_arm_cde_vcx3q_u8_impl_f32, 1715, 1683},
  { ARM::BI__builtin_arm_cde_vcx3q_u8_impl_s16, 1733, 1683},
  { ARM::BI__builtin_arm_cde_vcx3q_u8_impl_s32, 1751, 1683},
  { ARM::BI__builtin_arm_cde_vcx3q_u8_impl_s64, 1769, 1683},
  { ARM::BI__builtin_arm_cde_vcx3q_u8_impl_s8, 1787, 1683},
  { ARM::BI__builtin_arm_cde_vcx3q_u8_impl_u16, 1804, 1683},
  { ARM::BI__builtin_arm_cde_vcx3q_u8_impl_u32, 1822, 1683},
  { ARM::BI__builtin_arm_cde_vcx3q_u8_impl_u64, 1840, 1683},
  { ARM::BI__builtin_arm_cde_vcx3q_u8_impl_u8, 1858, 1683},
  { ARM::BI__builtin_arm_cde_vcx3qa_impl_f16, 1887, 1875},
  { ARM::BI__builtin_arm_cde_vcx3qa_impl_f32, 1903, 1875},
  { ARM::BI__builtin_arm_cde_vcx3qa_impl_s16, 1919, 1875},
  { ARM::BI__builtin_arm_cde_vcx3qa_impl_s32, 1935, 1875},
  { ARM::BI__builtin_arm_cde_vcx3qa_impl_s64, 1951, 1875},
  { ARM::BI__builtin_arm_cde_vcx3qa_impl_s8, 1967, 1875},
  { ARM::BI__builtin_arm_cde_vcx3qa_impl_u16, 1982, 1875},
  { ARM::BI__builtin_arm_cde_vcx3qa_impl_u32, 1998, 1875},
  { ARM::BI__builtin_arm_cde_vcx3qa_impl_u64, 2014, 1875},
  { ARM::BI__builtin_arm_cde_vcx3qa_impl_u8, 2030, 1875},
  { ARM::BI__builtin_arm_cde_vcx3qa_m_impl_f16, 2059, 2045},
  { ARM::BI__builtin_arm_cde_vcx3qa_m_impl_f32, 2077, 2045},
  { ARM::BI__builtin_arm_cde_vcx3qa_m_impl_s16, 2095, 2045},
  { ARM::BI__builtin_arm_cde_vcx3qa_m_impl_s32, 2113, 2045},
  { ARM::BI__builtin_arm_cde_vcx3qa_m_impl_s64, 2131, 2045},
  { ARM::BI__builtin_arm_cde_vcx3qa_m_impl_s8, 2149, 2045},
  { ARM::BI__builtin_arm_cde_vcx3qa_m_impl_u16, 2166, 2045},
  { ARM::BI__builtin_arm_cde_vcx3qa_m_impl_u32, 2184, 2045},
  { ARM::BI__builtin_arm_cde_vcx3qa_m_impl_u64, 2202, 2045},
  { ARM::BI__builtin_arm_cde_vcx3qa_m_impl_u8, 2220, 2045},
  { ARM::BI__builtin_arm_cde_vreinterpretq_u8_u8, 2254, 2237},
};

ArrayRef<IntrinToName> Map(MapData);

static const char IntrinNames[] = {
    "\000cx1\000cx1a\000cx1d\000cx1da\000cx2\000cx2a\000cx2d\000cx2da\000cx3"
    "\000cx3a\000cx3d\000cx3da\000vcx1_u32\000vcx1a_u32\000vcx1d_u64\000vcx1"
    "da_u64\000vcx1q_m\000vcx1q_m_f16\000vcx1q_m_f32\000vcx1q_m_s16\000vcx1q"
    "_m_s32\000vcx1q_m_s64\000vcx1q_m_s8\000vcx1q_m_u16\000vcx1q_m_u32\000vc"
    "x1q_m_u64\000vcx1q_m_u8\000vcx1q_u8\000vcx1qa\000vcx1qa_f16\000vcx1qa_f"
    "32\000vcx1qa_m\000vcx1qa_m_f16\000vcx1qa_m_f32\000vcx1qa_m_s16\000vcx1q"
    "a_m_s32\000vcx1qa_m_s64\000vcx1qa_m_s8\000vcx1qa_m_u16\000vcx1qa_m_u32\000"
    "vcx1qa_m_u64\000vcx1qa_m_u8\000vcx1qa_s16\000vcx1qa_s32\000vcx1qa_s64\000"
    "vcx1qa_s8\000vcx1qa_u16\000vcx1qa_u32\000vcx1qa_u64\000vcx1qa_u8\000vcx"
    "2_u32\000vcx2a_u32\000vcx2d_u64\000vcx2da_u64\000vcx2q\000vcx2q_f16\000"
    "vcx2q_f32\000vcx2q_m_impl\000vcx2q_m_impl_f16\000vcx2q_m_impl_f32\000vc"
    "x2q_m_impl_s16\000vcx2q_m_impl_s32\000vcx2q_m_impl_s64\000vcx2q_m_impl_"
    "s8\000vcx2q_m_impl_u16\000vcx2q_m_impl_u32\000vcx2q_m_impl_u64\000vcx2q"
    "_m_impl_u8\000vcx2q_s16\000vcx2q_s32\000vcx2q_s64\000vcx2q_s8\000vcx2q_"
    "u16\000vcx2q_u32\000vcx2q_u64\000vcx2q_u8\000vcx2q_u8_f16\000vcx2q_u8_f"
    "32\000vcx2q_u8_s16\000vcx2q_u8_s32\000vcx2q_u8_s64\000vcx2q_u8_s8\000vc"
    "x2q_u8_u16\000vcx2q_u8_u32\000vcx2q_u8_u64\000vcx2q_u8_u8\000vcx2qa_imp"
    "l\000vcx2qa_impl_f16\000vcx2qa_impl_f32\000vcx2qa_impl_s16\000vcx2qa_im"
    "pl_s32\000vcx2qa_impl_s64\000vcx2qa_impl_s8\000vcx2qa_impl_u16\000vcx2q"
    "a_impl_u32\000vcx2qa_impl_u64\000vcx2qa_impl_u8\000vcx2qa_m_impl\000vcx"
    "2qa_m_impl_f16\000vcx2qa_m_impl_f32\000vcx2qa_m_impl_s16\000vcx2qa_m_im"
    "pl_s32\000vcx2qa_m_impl_s64\000vcx2qa_m_impl_s8\000vcx2qa_m_impl_u16\000"
    "vcx2qa_m_impl_u32\000vcx2qa_m_impl_u64\000vcx2qa_m_impl_u8\000vcx3_u32\000"
    "vcx3a_u32\000vcx3d_u64\000vcx3da_u64\000vcx3q_impl\000vcx3q_impl_f16\000"
    "vcx3q_impl_f32\000vcx3q_impl_s16\000vcx3q_impl_s32\000vcx3q_impl_s64\000"
    "vcx3q_impl_s8\000vcx3q_impl_u16\000vcx3q_impl_u32\000vcx3q_impl_u64\000"
    "vcx3q_impl_u8\000vcx3q_m_impl\000vcx3q_m_impl_f16\000vcx3q_m_impl_f32\000"
    "vcx3q_m_impl_s16\000vcx3q_m_impl_s32\000vcx3q_m_impl_s64\000vcx3q_m_imp"
    "l_s8\000vcx3q_m_impl_u16\000vcx3q_m_impl_u32\000vcx3q_m_impl_u64\000vcx"
    "3q_m_impl_u8\000vcx3q_u8_impl\000vcx3q_u8_impl_f16\000vcx3q_u8_impl_f32"
    "\000vcx3q_u8_impl_s16\000vcx3q_u8_impl_s32\000vcx3q_u8_impl_s64\000vcx3"
    "q_u8_impl_s8\000vcx3q_u8_impl_u16\000vcx3q_u8_impl_u32\000vcx3q_u8_impl"
    "_u64\000vcx3q_u8_impl_u8\000vcx3qa_impl\000vcx3qa_impl_f16\000vcx3qa_im"
    "pl_f32\000vcx3qa_impl_s16\000vcx3qa_impl_s32\000vcx3qa_impl_s64\000vcx3"
    "qa_impl_s8\000vcx3qa_impl_u16\000vcx3qa_impl_u32\000vcx3qa_impl_u64\000"
    "vcx3qa_impl_u8\000vcx3qa_m_impl\000vcx3qa_m_impl_f16\000vcx3qa_m_impl_f"
    "32\000vcx3qa_m_impl_s16\000vcx3qa_m_impl_s32\000vcx3qa_m_impl_s64\000vc"
    "x3qa_m_impl_s8\000vcx3qa_m_impl_u16\000vcx3qa_m_impl_u32\000vcx3qa_m_im"
    "pl_u64\000vcx3qa_m_impl_u8\000vreinterpretq_u8\000vreinterpretq_u8_u8\000"};

