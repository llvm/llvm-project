static const IntrinToName MapData[] = {
  { ARM::BI__builtin_arm_mve_asrl, 1, -1},
  { ARM::BI__builtin_arm_mve_lsll, 6, -1},
  { ARM::BI__builtin_arm_mve_sqrshr, 11, -1},
  { ARM::BI__builtin_arm_mve_sqrshrl, 18, -1},
  { ARM::BI__builtin_arm_mve_sqrshrl_sat48, 26, -1},
  { ARM::BI__builtin_arm_mve_sqshl, 40, -1},
  { ARM::BI__builtin_arm_mve_sqshll, 46, -1},
  { ARM::BI__builtin_arm_mve_srshr, 53, -1},
  { ARM::BI__builtin_arm_mve_srshrl, 59, -1},
  { ARM::BI__builtin_arm_mve_uqrshl, 66, -1},
  { ARM::BI__builtin_arm_mve_uqrshll, 73, -1},
  { ARM::BI__builtin_arm_mve_uqrshll_sat48, 81, -1},
  { ARM::BI__builtin_arm_mve_uqshl, 95, -1},
  { ARM::BI__builtin_arm_mve_uqshll, 101, -1},
  { ARM::BI__builtin_arm_mve_urshr, 108, -1},
  { ARM::BI__builtin_arm_mve_urshrl, 114, -1},
  { ARM::BI__builtin_arm_mve_vabavq_p_s16, 130, 121},
  { ARM::BI__builtin_arm_mve_vabavq_p_s32, 143, 121},
  { ARM::BI__builtin_arm_mve_vabavq_p_s8, 156, 121},
  { ARM::BI__builtin_arm_mve_vabavq_p_u16, 168, 121},
  { ARM::BI__builtin_arm_mve_vabavq_p_u32, 181, 121},
  { ARM::BI__builtin_arm_mve_vabavq_p_u8, 194, 121},
  { ARM::BI__builtin_arm_mve_vabavq_s16, 213, 206},
  { ARM::BI__builtin_arm_mve_vabavq_s32, 224, 206},
  { ARM::BI__builtin_arm_mve_vabavq_s8, 235, 206},
  { ARM::BI__builtin_arm_mve_vabavq_u16, 245, 206},
  { ARM::BI__builtin_arm_mve_vabavq_u32, 256, 206},
  { ARM::BI__builtin_arm_mve_vabavq_u8, 267, 206},
  { ARM::BI__builtin_arm_mve_vabdq_f16, 283, 277},
  { ARM::BI__builtin_arm_mve_vabdq_f32, 293, 277},
  { ARM::BI__builtin_arm_mve_vabdq_m_f16, 311, 303},
  { ARM::BI__builtin_arm_mve_vabdq_m_f32, 323, 303},
  { ARM::BI__builtin_arm_mve_vabdq_m_s16, 335, 303},
  { ARM::BI__builtin_arm_mve_vabdq_m_s32, 347, 303},
  { ARM::BI__builtin_arm_mve_vabdq_m_s8, 359, 303},
  { ARM::BI__builtin_arm_mve_vabdq_m_u16, 370, 303},
  { ARM::BI__builtin_arm_mve_vabdq_m_u32, 382, 303},
  { ARM::BI__builtin_arm_mve_vabdq_m_u8, 394, 303},
  { ARM::BI__builtin_arm_mve_vabdq_s16, 405, 277},
  { ARM::BI__builtin_arm_mve_vabdq_s32, 415, 277},
  { ARM::BI__builtin_arm_mve_vabdq_s8, 425, 277},
  { ARM::BI__builtin_arm_mve_vabdq_u16, 434, 277},
  { ARM::BI__builtin_arm_mve_vabdq_u32, 444, 277},
  { ARM::BI__builtin_arm_mve_vabdq_u8, 454, 277},
  { ARM::BI__builtin_arm_mve_vabdq_x_f16, 471, 463},
  { ARM::BI__builtin_arm_mve_vabdq_x_f32, 483, 463},
  { ARM::BI__builtin_arm_mve_vabdq_x_s16, 495, 463},
  { ARM::BI__builtin_arm_mve_vabdq_x_s32, 507, 463},
  { ARM::BI__builtin_arm_mve_vabdq_x_s8, 519, 463},
  { ARM::BI__builtin_arm_mve_vabdq_x_u16, 530, 463},
  { ARM::BI__builtin_arm_mve_vabdq_x_u32, 542, 463},
  { ARM::BI__builtin_arm_mve_vabdq_x_u8, 554, 463},
  { ARM::BI__builtin_arm_mve_vabsq_f16, 571, 565},
  { ARM::BI__builtin_arm_mve_vabsq_f32, 581, 565},
  { ARM::BI__builtin_arm_mve_vabsq_m_f16, 599, 591},
  { ARM::BI__builtin_arm_mve_vabsq_m_f32, 611, 591},
  { ARM::BI__builtin_arm_mve_vabsq_m_s16, 623, 591},
  { ARM::BI__builtin_arm_mve_vabsq_m_s32, 635, 591},
  { ARM::BI__builtin_arm_mve_vabsq_m_s8, 647, 591},
  { ARM::BI__builtin_arm_mve_vabsq_s16, 658, 565},
  { ARM::BI__builtin_arm_mve_vabsq_s32, 668, 565},
  { ARM::BI__builtin_arm_mve_vabsq_s8, 678, 565},
  { ARM::BI__builtin_arm_mve_vabsq_x_f16, 695, 687},
  { ARM::BI__builtin_arm_mve_vabsq_x_f32, 707, 687},
  { ARM::BI__builtin_arm_mve_vabsq_x_s16, 719, 687},
  { ARM::BI__builtin_arm_mve_vabsq_x_s32, 731, 687},
  { ARM::BI__builtin_arm_mve_vabsq_x_s8, 743, 687},
  { ARM::BI__builtin_arm_mve_vadciq_m_s32, 763, 754},
  { ARM::BI__builtin_arm_mve_vadciq_m_u32, 776, 754},
  { ARM::BI__builtin_arm_mve_vadciq_s32, 796, 789},
  { ARM::BI__builtin_arm_mve_vadciq_u32, 807, 789},
  { ARM::BI__builtin_arm_mve_vadcq_m_s32, 826, 818},
  { ARM::BI__builtin_arm_mve_vadcq_m_u32, 838, 818},
  { ARM::BI__builtin_arm_mve_vadcq_s32, 856, 850},
  { ARM::BI__builtin_arm_mve_vadcq_u32, 866, 850},
  { ARM::BI__builtin_arm_mve_vaddlvaq_p_s32, 887, 876},
  { ARM::BI__builtin_arm_mve_vaddlvaq_p_u32, 902, 876},
  { ARM::BI__builtin_arm_mve_vaddlvaq_s32, 926, 917},
  { ARM::BI__builtin_arm_mve_vaddlvaq_u32, 939, 917},
  { ARM::BI__builtin_arm_mve_vaddlvq_p_s32, 962, 952},
  { ARM::BI__builtin_arm_mve_vaddlvq_p_u32, 976, 952},
  { ARM::BI__builtin_arm_mve_vaddlvq_s32, 998, 990},
  { ARM::BI__builtin_arm_mve_vaddlvq_u32, 1010, 990},
  { ARM::BI__builtin_arm_mve_vaddq_f16, 1028, 1022},
  { ARM::BI__builtin_arm_mve_vaddq_f32, 1038, 1022},
  { ARM::BI__builtin_arm_mve_vaddq_m_f16, 1056, 1048},
  { ARM::BI__builtin_arm_mve_vaddq_m_f32, 1068, 1048},
  { ARM::BI__builtin_arm_mve_vaddq_m_n_f16, 1080, 1048},
  { ARM::BI__builtin_arm_mve_vaddq_m_n_f32, 1094, 1048},
  { ARM::BI__builtin_arm_mve_vaddq_m_n_s16, 1108, 1048},
  { ARM::BI__builtin_arm_mve_vaddq_m_n_s32, 1122, 1048},
  { ARM::BI__builtin_arm_mve_vaddq_m_n_s8, 1136, 1048},
  { ARM::BI__builtin_arm_mve_vaddq_m_n_u16, 1149, 1048},
  { ARM::BI__builtin_arm_mve_vaddq_m_n_u32, 1163, 1048},
  { ARM::BI__builtin_arm_mve_vaddq_m_n_u8, 1177, 1048},
  { ARM::BI__builtin_arm_mve_vaddq_m_s16, 1190, 1048},
  { ARM::BI__builtin_arm_mve_vaddq_m_s32, 1202, 1048},
  { ARM::BI__builtin_arm_mve_vaddq_m_s8, 1214, 1048},
  { ARM::BI__builtin_arm_mve_vaddq_m_u16, 1225, 1048},
  { ARM::BI__builtin_arm_mve_vaddq_m_u32, 1237, 1048},
  { ARM::BI__builtin_arm_mve_vaddq_m_u8, 1249, 1048},
  { ARM::BI__builtin_arm_mve_vaddq_n_f16, 1260, 1022},
  { ARM::BI__builtin_arm_mve_vaddq_n_f32, 1272, 1022},
  { ARM::BI__builtin_arm_mve_vaddq_n_s16, 1284, 1022},
  { ARM::BI__builtin_arm_mve_vaddq_n_s32, 1296, 1022},
  { ARM::BI__builtin_arm_mve_vaddq_n_s8, 1308, 1022},
  { ARM::BI__builtin_arm_mve_vaddq_n_u16, 1319, 1022},
  { ARM::BI__builtin_arm_mve_vaddq_n_u32, 1331, 1022},
  { ARM::BI__builtin_arm_mve_vaddq_n_u8, 1343, 1022},
  { ARM::BI__builtin_arm_mve_vaddq_s16, 1354, 1022},
  { ARM::BI__builtin_arm_mve_vaddq_s32, 1364, 1022},
  { ARM::BI__builtin_arm_mve_vaddq_s8, 1374, 1022},
  { ARM::BI__builtin_arm_mve_vaddq_u16, 1383, 1022},
  { ARM::BI__builtin_arm_mve_vaddq_u32, 1393, 1022},
  { ARM::BI__builtin_arm_mve_vaddq_u8, 1403, 1022},
  { ARM::BI__builtin_arm_mve_vaddq_x_f16, 1420, 1412},
  { ARM::BI__builtin_arm_mve_vaddq_x_f32, 1432, 1412},
  { ARM::BI__builtin_arm_mve_vaddq_x_n_f16, 1444, 1412},
  { ARM::BI__builtin_arm_mve_vaddq_x_n_f32, 1458, 1412},
  { ARM::BI__builtin_arm_mve_vaddq_x_n_s16, 1472, 1412},
  { ARM::BI__builtin_arm_mve_vaddq_x_n_s32, 1486, 1412},
  { ARM::BI__builtin_arm_mve_vaddq_x_n_s8, 1500, 1412},
  { ARM::BI__builtin_arm_mve_vaddq_x_n_u16, 1513, 1412},
  { ARM::BI__builtin_arm_mve_vaddq_x_n_u32, 1527, 1412},
  { ARM::BI__builtin_arm_mve_vaddq_x_n_u8, 1541, 1412},
  { ARM::BI__builtin_arm_mve_vaddq_x_s16, 1554, 1412},
  { ARM::BI__builtin_arm_mve_vaddq_x_s32, 1566, 1412},
  { ARM::BI__builtin_arm_mve_vaddq_x_s8, 1578, 1412},
  { ARM::BI__builtin_arm_mve_vaddq_x_u16, 1589, 1412},
  { ARM::BI__builtin_arm_mve_vaddq_x_u32, 1601, 1412},
  { ARM::BI__builtin_arm_mve_vaddq_x_u8, 1613, 1412},
  { ARM::BI__builtin_arm_mve_vaddvaq_p_s16, 1634, 1624},
  { ARM::BI__builtin_arm_mve_vaddvaq_p_s32, 1648, 1624},
  { ARM::BI__builtin_arm_mve_vaddvaq_p_s8, 1662, 1624},
  { ARM::BI__builtin_arm_mve_vaddvaq_p_u16, 1675, 1624},
  { ARM::BI__builtin_arm_mve_vaddvaq_p_u32, 1689, 1624},
  { ARM::BI__builtin_arm_mve_vaddvaq_p_u8, 1703, 1624},
  { ARM::BI__builtin_arm_mve_vaddvaq_s16, 1724, 1716},
  { ARM::BI__builtin_arm_mve_vaddvaq_s32, 1736, 1716},
  { ARM::BI__builtin_arm_mve_vaddvaq_s8, 1748, 1716},
  { ARM::BI__builtin_arm_mve_vaddvaq_u16, 1759, 1716},
  { ARM::BI__builtin_arm_mve_vaddvaq_u32, 1771, 1716},
  { ARM::BI__builtin_arm_mve_vaddvaq_u8, 1783, 1716},
  { ARM::BI__builtin_arm_mve_vaddvq_p_s16, 1803, 1794},
  { ARM::BI__builtin_arm_mve_vaddvq_p_s32, 1816, 1794},
  { ARM::BI__builtin_arm_mve_vaddvq_p_s8, 1829, 1794},
  { ARM::BI__builtin_arm_mve_vaddvq_p_u16, 1841, 1794},
  { ARM::BI__builtin_arm_mve_vaddvq_p_u32, 1854, 1794},
  { ARM::BI__builtin_arm_mve_vaddvq_p_u8, 1867, 1794},
  { ARM::BI__builtin_arm_mve_vaddvq_s16, 1886, 1879},
  { ARM::BI__builtin_arm_mve_vaddvq_s32, 1897, 1879},
  { ARM::BI__builtin_arm_mve_vaddvq_s8, 1908, 1879},
  { ARM::BI__builtin_arm_mve_vaddvq_u16, 1918, 1879},
  { ARM::BI__builtin_arm_mve_vaddvq_u32, 1929, 1879},
  { ARM::BI__builtin_arm_mve_vaddvq_u8, 1940, 1879},
  { ARM::BI__builtin_arm_mve_vandq_f16, 1956, 1950},
  { ARM::BI__builtin_arm_mve_vandq_f32, 1966, 1950},
  { ARM::BI__builtin_arm_mve_vandq_m_f16, 1984, 1976},
  { ARM::BI__builtin_arm_mve_vandq_m_f32, 1996, 1976},
  { ARM::BI__builtin_arm_mve_vandq_m_s16, 2008, 1976},
  { ARM::BI__builtin_arm_mve_vandq_m_s32, 2020, 1976},
  { ARM::BI__builtin_arm_mve_vandq_m_s8, 2032, 1976},
  { ARM::BI__builtin_arm_mve_vandq_m_u16, 2043, 1976},
  { ARM::BI__builtin_arm_mve_vandq_m_u32, 2055, 1976},
  { ARM::BI__builtin_arm_mve_vandq_m_u8, 2067, 1976},
  { ARM::BI__builtin_arm_mve_vandq_s16, 2078, 1950},
  { ARM::BI__builtin_arm_mve_vandq_s32, 2088, 1950},
  { ARM::BI__builtin_arm_mve_vandq_s8, 2098, 1950},
  { ARM::BI__builtin_arm_mve_vandq_u16, 2107, 1950},
  { ARM::BI__builtin_arm_mve_vandq_u32, 2117, 1950},
  { ARM::BI__builtin_arm_mve_vandq_u8, 2127, 1950},
  { ARM::BI__builtin_arm_mve_vandq_x_f16, 2144, 2136},
  { ARM::BI__builtin_arm_mve_vandq_x_f32, 2156, 2136},
  { ARM::BI__builtin_arm_mve_vandq_x_s16, 2168, 2136},
  { ARM::BI__builtin_arm_mve_vandq_x_s32, 2180, 2136},
  { ARM::BI__builtin_arm_mve_vandq_x_s8, 2192, 2136},
  { ARM::BI__builtin_arm_mve_vandq_x_u16, 2203, 2136},
  { ARM::BI__builtin_arm_mve_vandq_x_u32, 2215, 2136},
  { ARM::BI__builtin_arm_mve_vandq_x_u8, 2227, 2136},
  { ARM::BI__builtin_arm_mve_vbicq_f16, 2244, 2238},
  { ARM::BI__builtin_arm_mve_vbicq_f32, 2254, 2238},
  { ARM::BI__builtin_arm_mve_vbicq_m_f16, 2272, 2264},
  { ARM::BI__builtin_arm_mve_vbicq_m_f32, 2284, 2264},
  { ARM::BI__builtin_arm_mve_vbicq_m_n_s16, 2306, 2296},
  { ARM::BI__builtin_arm_mve_vbicq_m_n_s32, 2320, 2296},
  { ARM::BI__builtin_arm_mve_vbicq_m_n_u16, 2334, 2296},
  { ARM::BI__builtin_arm_mve_vbicq_m_n_u32, 2348, 2296},
  { ARM::BI__builtin_arm_mve_vbicq_m_s16, 2362, 2264},
  { ARM::BI__builtin_arm_mve_vbicq_m_s32, 2374, 2264},
  { ARM::BI__builtin_arm_mve_vbicq_m_s8, 2386, 2264},
  { ARM::BI__builtin_arm_mve_vbicq_m_u16, 2397, 2264},
  { ARM::BI__builtin_arm_mve_vbicq_m_u32, 2409, 2264},
  { ARM::BI__builtin_arm_mve_vbicq_m_u8, 2421, 2264},
  { ARM::BI__builtin_arm_mve_vbicq_n_s16, 2432, 2238},
  { ARM::BI__builtin_arm_mve_vbicq_n_s32, 2444, 2238},
  { ARM::BI__builtin_arm_mve_vbicq_n_u16, 2456, 2238},
  { ARM::BI__builtin_arm_mve_vbicq_n_u32, 2468, 2238},
  { ARM::BI__builtin_arm_mve_vbicq_s16, 2480, 2238},
  { ARM::BI__builtin_arm_mve_vbicq_s32, 2490, 2238},
  { ARM::BI__builtin_arm_mve_vbicq_s8, 2500, 2238},
  { ARM::BI__builtin_arm_mve_vbicq_u16, 2509, 2238},
  { ARM::BI__builtin_arm_mve_vbicq_u32, 2519, 2238},
  { ARM::BI__builtin_arm_mve_vbicq_u8, 2529, 2238},
  { ARM::BI__builtin_arm_mve_vbicq_x_f16, 2546, 2538},
  { ARM::BI__builtin_arm_mve_vbicq_x_f32, 2558, 2538},
  { ARM::BI__builtin_arm_mve_vbicq_x_s16, 2570, 2538},
  { ARM::BI__builtin_arm_mve_vbicq_x_s32, 2582, 2538},
  { ARM::BI__builtin_arm_mve_vbicq_x_s8, 2594, 2538},
  { ARM::BI__builtin_arm_mve_vbicq_x_u16, 2605, 2538},
  { ARM::BI__builtin_arm_mve_vbicq_x_u32, 2617, 2538},
  { ARM::BI__builtin_arm_mve_vbicq_x_u8, 2629, 2538},
  { ARM::BI__builtin_arm_mve_vbrsrq_m_n_f16, 2649, 2640},
  { ARM::BI__builtin_arm_mve_vbrsrq_m_n_f32, 2664, 2640},
  { ARM::BI__builtin_arm_mve_vbrsrq_m_n_s16, 2679, 2640},
  { ARM::BI__builtin_arm_mve_vbrsrq_m_n_s32, 2694, 2640},
  { ARM::BI__builtin_arm_mve_vbrsrq_m_n_s8, 2709, 2640},
  { ARM::BI__builtin_arm_mve_vbrsrq_m_n_u16, 2723, 2640},
  { ARM::BI__builtin_arm_mve_vbrsrq_m_n_u32, 2738, 2640},
  { ARM::BI__builtin_arm_mve_vbrsrq_m_n_u8, 2753, 2640},
  { ARM::BI__builtin_arm_mve_vbrsrq_n_f16, 2774, 2767},
  { ARM::BI__builtin_arm_mve_vbrsrq_n_f32, 2787, 2767},
  { ARM::BI__builtin_arm_mve_vbrsrq_n_s16, 2800, 2767},
  { ARM::BI__builtin_arm_mve_vbrsrq_n_s32, 2813, 2767},
  { ARM::BI__builtin_arm_mve_vbrsrq_n_s8, 2826, 2767},
  { ARM::BI__builtin_arm_mve_vbrsrq_n_u16, 2838, 2767},
  { ARM::BI__builtin_arm_mve_vbrsrq_n_u32, 2851, 2767},
  { ARM::BI__builtin_arm_mve_vbrsrq_n_u8, 2864, 2767},
  { ARM::BI__builtin_arm_mve_vbrsrq_x_n_f16, 2885, 2876},
  { ARM::BI__builtin_arm_mve_vbrsrq_x_n_f32, 2900, 2876},
  { ARM::BI__builtin_arm_mve_vbrsrq_x_n_s16, 2915, 2876},
  { ARM::BI__builtin_arm_mve_vbrsrq_x_n_s32, 2930, 2876},
  { ARM::BI__builtin_arm_mve_vbrsrq_x_n_s8, 2945, 2876},
  { ARM::BI__builtin_arm_mve_vbrsrq_x_n_u16, 2959, 2876},
  { ARM::BI__builtin_arm_mve_vbrsrq_x_n_u32, 2974, 2876},
  { ARM::BI__builtin_arm_mve_vbrsrq_x_n_u8, 2989, 2876},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_f16, 3017, 3003},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_f32, 3035, 3003},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_m_f16, 3069, 3053},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_m_f32, 3089, 3053},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_m_s16, 3109, 3053},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_m_s32, 3129, 3053},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_m_s8, 3149, 3053},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_m_u16, 3168, 3053},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_m_u32, 3188, 3053},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_m_u8, 3208, 3053},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_s16, 3227, 3003},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_s32, 3245, 3003},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_s8, 3263, 3003},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_u16, 3280, 3003},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_u32, 3298, 3003},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_u8, 3316, 3003},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_x_f16, 3349, 3333},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_x_f32, 3369, 3333},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_x_s16, 3389, 3333},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_x_s32, 3409, 3333},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_x_s8, 3429, 3333},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_x_u16, 3448, 3333},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_x_u32, 3468, 3333},
  { ARM::BI__builtin_arm_mve_vcaddq_rot270_x_u8, 3488, 3333},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_f16, 3520, 3507},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_f32, 3537, 3507},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_m_f16, 3569, 3554},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_m_f32, 3588, 3554},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_m_s16, 3607, 3554},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_m_s32, 3626, 3554},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_m_s8, 3645, 3554},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_m_u16, 3663, 3554},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_m_u32, 3682, 3554},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_m_u8, 3701, 3554},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_s16, 3719, 3507},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_s32, 3736, 3507},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_s8, 3753, 3507},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_u16, 3769, 3507},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_u32, 3786, 3507},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_u8, 3803, 3507},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_x_f16, 3834, 3819},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_x_f32, 3853, 3819},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_x_s16, 3872, 3819},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_x_s32, 3891, 3819},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_x_s8, 3910, 3819},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_x_u16, 3928, 3819},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_x_u32, 3947, 3819},
  { ARM::BI__builtin_arm_mve_vcaddq_rot90_x_u8, 3966, 3819},
  { ARM::BI__builtin_arm_mve_vclsq_m_s16, 3992, 3984},
  { ARM::BI__builtin_arm_mve_vclsq_m_s32, 4004, 3984},
  { ARM::BI__builtin_arm_mve_vclsq_m_s8, 4016, 3984},
  { ARM::BI__builtin_arm_mve_vclsq_s16, 4033, 4027},
  { ARM::BI__builtin_arm_mve_vclsq_s32, 4043, 4027},
  { ARM::BI__builtin_arm_mve_vclsq_s8, 4053, 4027},
  { ARM::BI__builtin_arm_mve_vclsq_x_s16, 4070, 4062},
  { ARM::BI__builtin_arm_mve_vclsq_x_s32, 4082, 4062},
  { ARM::BI__builtin_arm_mve_vclsq_x_s8, 4094, 4062},
  { ARM::BI__builtin_arm_mve_vclzq_m_s16, 4113, 4105},
  { ARM::BI__builtin_arm_mve_vclzq_m_s32, 4125, 4105},
  { ARM::BI__builtin_arm_mve_vclzq_m_s8, 4137, 4105},
  { ARM::BI__builtin_arm_mve_vclzq_m_u16, 4148, 4105},
  { ARM::BI__builtin_arm_mve_vclzq_m_u32, 4160, 4105},
  { ARM::BI__builtin_arm_mve_vclzq_m_u8, 4172, 4105},
  { ARM::BI__builtin_arm_mve_vclzq_s16, 4189, 4183},
  { ARM::BI__builtin_arm_mve_vclzq_s32, 4199, 4183},
  { ARM::BI__builtin_arm_mve_vclzq_s8, 4209, 4183},
  { ARM::BI__builtin_arm_mve_vclzq_u16, 4218, 4183},
  { ARM::BI__builtin_arm_mve_vclzq_u32, 4228, 4183},
  { ARM::BI__builtin_arm_mve_vclzq_u8, 4238, 4183},
  { ARM::BI__builtin_arm_mve_vclzq_x_s16, 4255, 4247},
  { ARM::BI__builtin_arm_mve_vclzq_x_s32, 4267, 4247},
  { ARM::BI__builtin_arm_mve_vclzq_x_s8, 4279, 4247},
  { ARM::BI__builtin_arm_mve_vclzq_x_u16, 4290, 4247},
  { ARM::BI__builtin_arm_mve_vclzq_x_u32, 4302, 4247},
  { ARM::BI__builtin_arm_mve_vclzq_x_u8, 4314, 4247},
  { ARM::BI__builtin_arm_mve_vcmlaq_f16, 4332, 4325},
  { ARM::BI__builtin_arm_mve_vcmlaq_f32, 4343, 4325},
  { ARM::BI__builtin_arm_mve_vcmlaq_m_f16, 4363, 4354},
  { ARM::BI__builtin_arm_mve_vcmlaq_m_f32, 4376, 4354},
  { ARM::BI__builtin_arm_mve_vcmlaq_rot180_f16, 4403, 4389},
  { ARM::BI__builtin_arm_mve_vcmlaq_rot180_f32, 4421, 4389},
  { ARM::BI__builtin_arm_mve_vcmlaq_rot180_m_f16, 4455, 4439},
  { ARM::BI__builtin_arm_mve_vcmlaq_rot180_m_f32, 4475, 4439},
  { ARM::BI__builtin_arm_mve_vcmlaq_rot270_f16, 4509, 4495},
  { ARM::BI__builtin_arm_mve_vcmlaq_rot270_f32, 4527, 4495},
  { ARM::BI__builtin_arm_mve_vcmlaq_rot270_m_f16, 4561, 4545},
  { ARM::BI__builtin_arm_mve_vcmlaq_rot270_m_f32, 4581, 4545},
  { ARM::BI__builtin_arm_mve_vcmlaq_rot90_f16, 4614, 4601},
  { ARM::BI__builtin_arm_mve_vcmlaq_rot90_f32, 4631, 4601},
  { ARM::BI__builtin_arm_mve_vcmlaq_rot90_m_f16, 4663, 4648},
  { ARM::BI__builtin_arm_mve_vcmlaq_rot90_m_f32, 4682, 4648},
  { ARM::BI__builtin_arm_mve_vcmpcsq_m_n_u16, 4711, 4701},
  { ARM::BI__builtin_arm_mve_vcmpcsq_m_n_u32, 4727, 4701},
  { ARM::BI__builtin_arm_mve_vcmpcsq_m_n_u8, 4743, 4701},
  { ARM::BI__builtin_arm_mve_vcmpcsq_m_u16, 4758, 4701},
  { ARM::BI__builtin_arm_mve_vcmpcsq_m_u32, 4772, 4701},
  { ARM::BI__builtin_arm_mve_vcmpcsq_m_u8, 4786, 4701},
  { ARM::BI__builtin_arm_mve_vcmpcsq_n_u16, 4807, 4799},
  { ARM::BI__builtin_arm_mve_vcmpcsq_n_u32, 4821, 4799},
  { ARM::BI__builtin_arm_mve_vcmpcsq_n_u8, 4835, 4799},
  { ARM::BI__builtin_arm_mve_vcmpcsq_u16, 4848, 4799},
  { ARM::BI__builtin_arm_mve_vcmpcsq_u32, 4860, 4799},
  { ARM::BI__builtin_arm_mve_vcmpcsq_u8, 4872, 4799},
  { ARM::BI__builtin_arm_mve_vcmpeqq_f16, 4891, 4883},
  { ARM::BI__builtin_arm_mve_vcmpeqq_f32, 4903, 4883},
  { ARM::BI__builtin_arm_mve_vcmpeqq_m_f16, 4925, 4915},
  { ARM::BI__builtin_arm_mve_vcmpeqq_m_f32, 4939, 4915},
  { ARM::BI__builtin_arm_mve_vcmpeqq_m_n_f16, 4953, 4915},
  { ARM::BI__builtin_arm_mve_vcmpeqq_m_n_f32, 4969, 4915},
  { ARM::BI__builtin_arm_mve_vcmpeqq_m_n_s16, 4985, 4915},
  { ARM::BI__builtin_arm_mve_vcmpeqq_m_n_s32, 5001, 4915},
  { ARM::BI__builtin_arm_mve_vcmpeqq_m_n_s8, 5017, 4915},
  { ARM::BI__builtin_arm_mve_vcmpeqq_m_n_u16, 5032, 4915},
  { ARM::BI__builtin_arm_mve_vcmpeqq_m_n_u32, 5048, 4915},
  { ARM::BI__builtin_arm_mve_vcmpeqq_m_n_u8, 5064, 4915},
  { ARM::BI__builtin_arm_mve_vcmpeqq_m_s16, 5079, 4915},
  { ARM::BI__builtin_arm_mve_vcmpeqq_m_s32, 5093, 4915},
  { ARM::BI__builtin_arm_mve_vcmpeqq_m_s8, 5107, 4915},
  { ARM::BI__builtin_arm_mve_vcmpeqq_m_u16, 5120, 4915},
  { ARM::BI__builtin_arm_mve_vcmpeqq_m_u32, 5134, 4915},
  { ARM::BI__builtin_arm_mve_vcmpeqq_m_u8, 5148, 4915},
  { ARM::BI__builtin_arm_mve_vcmpeqq_n_f16, 5161, 4883},
  { ARM::BI__builtin_arm_mve_vcmpeqq_n_f32, 5175, 4883},
  { ARM::BI__builtin_arm_mve_vcmpeqq_n_s16, 5189, 4883},
  { ARM::BI__builtin_arm_mve_vcmpeqq_n_s32, 5203, 4883},
  { ARM::BI__builtin_arm_mve_vcmpeqq_n_s8, 5217, 4883},
  { ARM::BI__builtin_arm_mve_vcmpeqq_n_u16, 5230, 4883},
  { ARM::BI__builtin_arm_mve_vcmpeqq_n_u32, 5244, 4883},
  { ARM::BI__builtin_arm_mve_vcmpeqq_n_u8, 5258, 4883},
  { ARM::BI__builtin_arm_mve_vcmpeqq_s16, 5271, 4883},
  { ARM::BI__builtin_arm_mve_vcmpeqq_s32, 5283, 4883},
  { ARM::BI__builtin_arm_mve_vcmpeqq_s8, 5295, 4883},
  { ARM::BI__builtin_arm_mve_vcmpeqq_u16, 5306, 4883},
  { ARM::BI__builtin_arm_mve_vcmpeqq_u32, 5318, 4883},
  { ARM::BI__builtin_arm_mve_vcmpeqq_u8, 5330, 4883},
  { ARM::BI__builtin_arm_mve_vcmpgeq_f16, 5349, 5341},
  { ARM::BI__builtin_arm_mve_vcmpgeq_f32, 5361, 5341},
  { ARM::BI__builtin_arm_mve_vcmpgeq_m_f16, 5383, 5373},
  { ARM::BI__builtin_arm_mve_vcmpgeq_m_f32, 5397, 5373},
  { ARM::BI__builtin_arm_mve_vcmpgeq_m_n_f16, 5411, 5373},
  { ARM::BI__builtin_arm_mve_vcmpgeq_m_n_f32, 5427, 5373},
  { ARM::BI__builtin_arm_mve_vcmpgeq_m_n_s16, 5443, 5373},
  { ARM::BI__builtin_arm_mve_vcmpgeq_m_n_s32, 5459, 5373},
  { ARM::BI__builtin_arm_mve_vcmpgeq_m_n_s8, 5475, 5373},
  { ARM::BI__builtin_arm_mve_vcmpgeq_m_s16, 5490, 5373},
  { ARM::BI__builtin_arm_mve_vcmpgeq_m_s32, 5504, 5373},
  { ARM::BI__builtin_arm_mve_vcmpgeq_m_s8, 5518, 5373},
  { ARM::BI__builtin_arm_mve_vcmpgeq_n_f16, 5531, 5341},
  { ARM::BI__builtin_arm_mve_vcmpgeq_n_f32, 5545, 5341},
  { ARM::BI__builtin_arm_mve_vcmpgeq_n_s16, 5559, 5341},
  { ARM::BI__builtin_arm_mve_vcmpgeq_n_s32, 5573, 5341},
  { ARM::BI__builtin_arm_mve_vcmpgeq_n_s8, 5587, 5341},
  { ARM::BI__builtin_arm_mve_vcmpgeq_s16, 5600, 5341},
  { ARM::BI__builtin_arm_mve_vcmpgeq_s32, 5612, 5341},
  { ARM::BI__builtin_arm_mve_vcmpgeq_s8, 5624, 5341},
  { ARM::BI__builtin_arm_mve_vcmpgtq_f16, 5643, 5635},
  { ARM::BI__builtin_arm_mve_vcmpgtq_f32, 5655, 5635},
  { ARM::BI__builtin_arm_mve_vcmpgtq_m_f16, 5677, 5667},
  { ARM::BI__builtin_arm_mve_vcmpgtq_m_f32, 5691, 5667},
  { ARM::BI__builtin_arm_mve_vcmpgtq_m_n_f16, 5705, 5667},
  { ARM::BI__builtin_arm_mve_vcmpgtq_m_n_f32, 5721, 5667},
  { ARM::BI__builtin_arm_mve_vcmpgtq_m_n_s16, 5737, 5667},
  { ARM::BI__builtin_arm_mve_vcmpgtq_m_n_s32, 5753, 5667},
  { ARM::BI__builtin_arm_mve_vcmpgtq_m_n_s8, 5769, 5667},
  { ARM::BI__builtin_arm_mve_vcmpgtq_m_s16, 5784, 5667},
  { ARM::BI__builtin_arm_mve_vcmpgtq_m_s32, 5798, 5667},
  { ARM::BI__builtin_arm_mve_vcmpgtq_m_s8, 5812, 5667},
  { ARM::BI__builtin_arm_mve_vcmpgtq_n_f16, 5825, 5635},
  { ARM::BI__builtin_arm_mve_vcmpgtq_n_f32, 5839, 5635},
  { ARM::BI__builtin_arm_mve_vcmpgtq_n_s16, 5853, 5635},
  { ARM::BI__builtin_arm_mve_vcmpgtq_n_s32, 5867, 5635},
  { ARM::BI__builtin_arm_mve_vcmpgtq_n_s8, 5881, 5635},
  { ARM::BI__builtin_arm_mve_vcmpgtq_s16, 5894, 5635},
  { ARM::BI__builtin_arm_mve_vcmpgtq_s32, 5906, 5635},
  { ARM::BI__builtin_arm_mve_vcmpgtq_s8, 5918, 5635},
  { ARM::BI__builtin_arm_mve_vcmphiq_m_n_u16, 5939, 5929},
  { ARM::BI__builtin_arm_mve_vcmphiq_m_n_u32, 5955, 5929},
  { ARM::BI__builtin_arm_mve_vcmphiq_m_n_u8, 5971, 5929},
  { ARM::BI__builtin_arm_mve_vcmphiq_m_u16, 5986, 5929},
  { ARM::BI__builtin_arm_mve_vcmphiq_m_u32, 6000, 5929},
  { ARM::BI__builtin_arm_mve_vcmphiq_m_u8, 6014, 5929},
  { ARM::BI__builtin_arm_mve_vcmphiq_n_u16, 6035, 6027},
  { ARM::BI__builtin_arm_mve_vcmphiq_n_u32, 6049, 6027},
  { ARM::BI__builtin_arm_mve_vcmphiq_n_u8, 6063, 6027},
  { ARM::BI__builtin_arm_mve_vcmphiq_u16, 6076, 6027},
  { ARM::BI__builtin_arm_mve_vcmphiq_u32, 6088, 6027},
  { ARM::BI__builtin_arm_mve_vcmphiq_u8, 6100, 6027},
  { ARM::BI__builtin_arm_mve_vcmpleq_f16, 6119, 6111},
  { ARM::BI__builtin_arm_mve_vcmpleq_f32, 6131, 6111},
  { ARM::BI__builtin_arm_mve_vcmpleq_m_f16, 6153, 6143},
  { ARM::BI__builtin_arm_mve_vcmpleq_m_f32, 6167, 6143},
  { ARM::BI__builtin_arm_mve_vcmpleq_m_n_f16, 6181, 6143},
  { ARM::BI__builtin_arm_mve_vcmpleq_m_n_f32, 6197, 6143},
  { ARM::BI__builtin_arm_mve_vcmpleq_m_n_s16, 6213, 6143},
  { ARM::BI__builtin_arm_mve_vcmpleq_m_n_s32, 6229, 6143},
  { ARM::BI__builtin_arm_mve_vcmpleq_m_n_s8, 6245, 6143},
  { ARM::BI__builtin_arm_mve_vcmpleq_m_s16, 6260, 6143},
  { ARM::BI__builtin_arm_mve_vcmpleq_m_s32, 6274, 6143},
  { ARM::BI__builtin_arm_mve_vcmpleq_m_s8, 6288, 6143},
  { ARM::BI__builtin_arm_mve_vcmpleq_n_f16, 6301, 6111},
  { ARM::BI__builtin_arm_mve_vcmpleq_n_f32, 6315, 6111},
  { ARM::BI__builtin_arm_mve_vcmpleq_n_s16, 6329, 6111},
  { ARM::BI__builtin_arm_mve_vcmpleq_n_s32, 6343, 6111},
  { ARM::BI__builtin_arm_mve_vcmpleq_n_s8, 6357, 6111},
  { ARM::BI__builtin_arm_mve_vcmpleq_s16, 6370, 6111},
  { ARM::BI__builtin_arm_mve_vcmpleq_s32, 6382, 6111},
  { ARM::BI__builtin_arm_mve_vcmpleq_s8, 6394, 6111},
  { ARM::BI__builtin_arm_mve_vcmpltq_f16, 6413, 6405},
  { ARM::BI__builtin_arm_mve_vcmpltq_f32, 6425, 6405},
  { ARM::BI__builtin_arm_mve_vcmpltq_m_f16, 6447, 6437},
  { ARM::BI__builtin_arm_mve_vcmpltq_m_f32, 6461, 6437},
  { ARM::BI__builtin_arm_mve_vcmpltq_m_n_f16, 6475, 6437},
  { ARM::BI__builtin_arm_mve_vcmpltq_m_n_f32, 6491, 6437},
  { ARM::BI__builtin_arm_mve_vcmpltq_m_n_s16, 6507, 6437},
  { ARM::BI__builtin_arm_mve_vcmpltq_m_n_s32, 6523, 6437},
  { ARM::BI__builtin_arm_mve_vcmpltq_m_n_s8, 6539, 6437},
  { ARM::BI__builtin_arm_mve_vcmpltq_m_s16, 6554, 6437},
  { ARM::BI__builtin_arm_mve_vcmpltq_m_s32, 6568, 6437},
  { ARM::BI__builtin_arm_mve_vcmpltq_m_s8, 6582, 6437},
  { ARM::BI__builtin_arm_mve_vcmpltq_n_f16, 6595, 6405},
  { ARM::BI__builtin_arm_mve_vcmpltq_n_f32, 6609, 6405},
  { ARM::BI__builtin_arm_mve_vcmpltq_n_s16, 6623, 6405},
  { ARM::BI__builtin_arm_mve_vcmpltq_n_s32, 6637, 6405},
  { ARM::BI__builtin_arm_mve_vcmpltq_n_s8, 6651, 6405},
  { ARM::BI__builtin_arm_mve_vcmpltq_s16, 6664, 6405},
  { ARM::BI__builtin_arm_mve_vcmpltq_s32, 6676, 6405},
  { ARM::BI__builtin_arm_mve_vcmpltq_s8, 6688, 6405},
  { ARM::BI__builtin_arm_mve_vcmpneq_f16, 6707, 6699},
  { ARM::BI__builtin_arm_mve_vcmpneq_f32, 6719, 6699},
  { ARM::BI__builtin_arm_mve_vcmpneq_m_f16, 6741, 6731},
  { ARM::BI__builtin_arm_mve_vcmpneq_m_f32, 6755, 6731},
  { ARM::BI__builtin_arm_mve_vcmpneq_m_n_f16, 6769, 6731},
  { ARM::BI__builtin_arm_mve_vcmpneq_m_n_f32, 6785, 6731},
  { ARM::BI__builtin_arm_mve_vcmpneq_m_n_s16, 6801, 6731},
  { ARM::BI__builtin_arm_mve_vcmpneq_m_n_s32, 6817, 6731},
  { ARM::BI__builtin_arm_mve_vcmpneq_m_n_s8, 6833, 6731},
  { ARM::BI__builtin_arm_mve_vcmpneq_m_n_u16, 6848, 6731},
  { ARM::BI__builtin_arm_mve_vcmpneq_m_n_u32, 6864, 6731},
  { ARM::BI__builtin_arm_mve_vcmpneq_m_n_u8, 6880, 6731},
  { ARM::BI__builtin_arm_mve_vcmpneq_m_s16, 6895, 6731},
  { ARM::BI__builtin_arm_mve_vcmpneq_m_s32, 6909, 6731},
  { ARM::BI__builtin_arm_mve_vcmpneq_m_s8, 6923, 6731},
  { ARM::BI__builtin_arm_mve_vcmpneq_m_u16, 6936, 6731},
  { ARM::BI__builtin_arm_mve_vcmpneq_m_u32, 6950, 6731},
  { ARM::BI__builtin_arm_mve_vcmpneq_m_u8, 6964, 6731},
  { ARM::BI__builtin_arm_mve_vcmpneq_n_f16, 6977, 6699},
  { ARM::BI__builtin_arm_mve_vcmpneq_n_f32, 6991, 6699},
  { ARM::BI__builtin_arm_mve_vcmpneq_n_s16, 7005, 6699},
  { ARM::BI__builtin_arm_mve_vcmpneq_n_s32, 7019, 6699},
  { ARM::BI__builtin_arm_mve_vcmpneq_n_s8, 7033, 6699},
  { ARM::BI__builtin_arm_mve_vcmpneq_n_u16, 7046, 6699},
  { ARM::BI__builtin_arm_mve_vcmpneq_n_u32, 7060, 6699},
  { ARM::BI__builtin_arm_mve_vcmpneq_n_u8, 7074, 6699},
  { ARM::BI__builtin_arm_mve_vcmpneq_s16, 7087, 6699},
  { ARM::BI__builtin_arm_mve_vcmpneq_s32, 7099, 6699},
  { ARM::BI__builtin_arm_mve_vcmpneq_s8, 7111, 6699},
  { ARM::BI__builtin_arm_mve_vcmpneq_u16, 7122, 6699},
  { ARM::BI__builtin_arm_mve_vcmpneq_u32, 7134, 6699},
  { ARM::BI__builtin_arm_mve_vcmpneq_u8, 7146, 6699},
  { ARM::BI__builtin_arm_mve_vcmulq_f16, 7164, 7157},
  { ARM::BI__builtin_arm_mve_vcmulq_f32, 7175, 7157},
  { ARM::BI__builtin_arm_mve_vcmulq_m_f16, 7195, 7186},
  { ARM::BI__builtin_arm_mve_vcmulq_m_f32, 7208, 7186},
  { ARM::BI__builtin_arm_mve_vcmulq_rot180_f16, 7235, 7221},
  { ARM::BI__builtin_arm_mve_vcmulq_rot180_f32, 7253, 7221},
  { ARM::BI__builtin_arm_mve_vcmulq_rot180_m_f16, 7287, 7271},
  { ARM::BI__builtin_arm_mve_vcmulq_rot180_m_f32, 7307, 7271},
  { ARM::BI__builtin_arm_mve_vcmulq_rot180_x_f16, 7343, 7327},
  { ARM::BI__builtin_arm_mve_vcmulq_rot180_x_f32, 7363, 7327},
  { ARM::BI__builtin_arm_mve_vcmulq_rot270_f16, 7397, 7383},
  { ARM::BI__builtin_arm_mve_vcmulq_rot270_f32, 7415, 7383},
  { ARM::BI__builtin_arm_mve_vcmulq_rot270_m_f16, 7449, 7433},
  { ARM::BI__builtin_arm_mve_vcmulq_rot270_m_f32, 7469, 7433},
  { ARM::BI__builtin_arm_mve_vcmulq_rot270_x_f16, 7505, 7489},
  { ARM::BI__builtin_arm_mve_vcmulq_rot270_x_f32, 7525, 7489},
  { ARM::BI__builtin_arm_mve_vcmulq_rot90_f16, 7558, 7545},
  { ARM::BI__builtin_arm_mve_vcmulq_rot90_f32, 7575, 7545},
  { ARM::BI__builtin_arm_mve_vcmulq_rot90_m_f16, 7607, 7592},
  { ARM::BI__builtin_arm_mve_vcmulq_rot90_m_f32, 7626, 7592},
  { ARM::BI__builtin_arm_mve_vcmulq_rot90_x_f16, 7660, 7645},
  { ARM::BI__builtin_arm_mve_vcmulq_rot90_x_f32, 7679, 7645},
  { ARM::BI__builtin_arm_mve_vcmulq_x_f16, 7707, 7698},
  { ARM::BI__builtin_arm_mve_vcmulq_x_f32, 7720, 7698},
  { ARM::BI__builtin_arm_mve_vcreateq_f16, 7733, -1},
  { ARM::BI__builtin_arm_mve_vcreateq_f32, 7746, -1},
  { ARM::BI__builtin_arm_mve_vcreateq_s16, 7759, -1},
  { ARM::BI__builtin_arm_mve_vcreateq_s32, 7772, -1},
  { ARM::BI__builtin_arm_mve_vcreateq_s64, 7785, -1},
  { ARM::BI__builtin_arm_mve_vcreateq_s8, 7798, -1},
  { ARM::BI__builtin_arm_mve_vcreateq_u16, 7810, -1},
  { ARM::BI__builtin_arm_mve_vcreateq_u32, 7823, -1},
  { ARM::BI__builtin_arm_mve_vcreateq_u64, 7836, -1},
  { ARM::BI__builtin_arm_mve_vcreateq_u8, 7849, -1},
  { ARM::BI__builtin_arm_mve_vctp16q, 7861, -1},
  { ARM::BI__builtin_arm_mve_vctp16q_m, 7869, -1},
  { ARM::BI__builtin_arm_mve_vctp32q, 7879, -1},
  { ARM::BI__builtin_arm_mve_vctp32q_m, 7887, -1},
  { ARM::BI__builtin_arm_mve_vctp64q, 7897, -1},
  { ARM::BI__builtin_arm_mve_vctp64q_m, 7905, -1},
  { ARM::BI__builtin_arm_mve_vctp8q, 7915, -1},
  { ARM::BI__builtin_arm_mve_vctp8q_m, 7922, -1},
  { ARM::BI__builtin_arm_mve_vcvtaq_m_s16_f16, 7940, 7931},
  { ARM::BI__builtin_arm_mve_vcvtaq_m_s32_f32, 7957, 7931},
  { ARM::BI__builtin_arm_mve_vcvtaq_m_u16_f16, 7974, 7931},
  { ARM::BI__builtin_arm_mve_vcvtaq_m_u32_f32, 7991, 7931},
  { ARM::BI__builtin_arm_mve_vcvtaq_s16_f16, 8008, -1},
  { ARM::BI__builtin_arm_mve_vcvtaq_s32_f32, 8023, -1},
  { ARM::BI__builtin_arm_mve_vcvtaq_u16_f16, 8038, -1},
  { ARM::BI__builtin_arm_mve_vcvtaq_u32_f32, 8053, -1},
  { ARM::BI__builtin_arm_mve_vcvtaq_x_s16_f16, 8068, -1},
  { ARM::BI__builtin_arm_mve_vcvtaq_x_s32_f32, 8085, -1},
  { ARM::BI__builtin_arm_mve_vcvtaq_x_u16_f16, 8102, -1},
  { ARM::BI__builtin_arm_mve_vcvtaq_x_u32_f32, 8119, -1},
  { ARM::BI__builtin_arm_mve_vcvtbq_f16_f32, 8136, -1},
  { ARM::BI__builtin_arm_mve_vcvtbq_f32_f16, 8151, -1},
  { ARM::BI__builtin_arm_mve_vcvtbq_m_f16_f32, 8166, -1},
  { ARM::BI__builtin_arm_mve_vcvtbq_m_f32_f16, 8183, -1},
  { ARM::BI__builtin_arm_mve_vcvtbq_x_f32_f16, 8200, -1},
  { ARM::BI__builtin_arm_mve_vcvtmq_m_s16_f16, 8226, 8217},
  { ARM::BI__builtin_arm_mve_vcvtmq_m_s32_f32, 8243, 8217},
  { ARM::BI__builtin_arm_mve_vcvtmq_m_u16_f16, 8260, 8217},
  { ARM::BI__builtin_arm_mve_vcvtmq_m_u32_f32, 8277, 8217},
  { ARM::BI__builtin_arm_mve_vcvtmq_s16_f16, 8294, -1},
  { ARM::BI__builtin_arm_mve_vcvtmq_s32_f32, 8309, -1},
  { ARM::BI__builtin_arm_mve_vcvtmq_u16_f16, 8324, -1},
  { ARM::BI__builtin_arm_mve_vcvtmq_u32_f32, 8339, -1},
  { ARM::BI__builtin_arm_mve_vcvtmq_x_s16_f16, 8354, -1},
  { ARM::BI__builtin_arm_mve_vcvtmq_x_s32_f32, 8371, -1},
  { ARM::BI__builtin_arm_mve_vcvtmq_x_u16_f16, 8388, -1},
  { ARM::BI__builtin_arm_mve_vcvtmq_x_u32_f32, 8405, -1},
  { ARM::BI__builtin_arm_mve_vcvtnq_m_s16_f16, 8431, 8422},
  { ARM::BI__builtin_arm_mve_vcvtnq_m_s32_f32, 8448, 8422},
  { ARM::BI__builtin_arm_mve_vcvtnq_m_u16_f16, 8465, 8422},
  { ARM::BI__builtin_arm_mve_vcvtnq_m_u32_f32, 8482, 8422},
  { ARM::BI__builtin_arm_mve_vcvtnq_s16_f16, 8499, -1},
  { ARM::BI__builtin_arm_mve_vcvtnq_s32_f32, 8514, -1},
  { ARM::BI__builtin_arm_mve_vcvtnq_u16_f16, 8529, -1},
  { ARM::BI__builtin_arm_mve_vcvtnq_u32_f32, 8544, -1},
  { ARM::BI__builtin_arm_mve_vcvtnq_x_s16_f16, 8559, -1},
  { ARM::BI__builtin_arm_mve_vcvtnq_x_s32_f32, 8576, -1},
  { ARM::BI__builtin_arm_mve_vcvtnq_x_u16_f16, 8593, -1},
  { ARM::BI__builtin_arm_mve_vcvtnq_x_u32_f32, 8610, -1},
  { ARM::BI__builtin_arm_mve_vcvtpq_m_s16_f16, 8636, 8627},
  { ARM::BI__builtin_arm_mve_vcvtpq_m_s32_f32, 8653, 8627},
  { ARM::BI__builtin_arm_mve_vcvtpq_m_u16_f16, 8670, 8627},
  { ARM::BI__builtin_arm_mve_vcvtpq_m_u32_f32, 8687, 8627},
  { ARM::BI__builtin_arm_mve_vcvtpq_s16_f16, 8704, -1},
  { ARM::BI__builtin_arm_mve_vcvtpq_s32_f32, 8719, -1},
  { ARM::BI__builtin_arm_mve_vcvtpq_u16_f16, 8734, -1},
  { ARM::BI__builtin_arm_mve_vcvtpq_u32_f32, 8749, -1},
  { ARM::BI__builtin_arm_mve_vcvtpq_x_s16_f16, 8764, -1},
  { ARM::BI__builtin_arm_mve_vcvtpq_x_s32_f32, 8781, -1},
  { ARM::BI__builtin_arm_mve_vcvtpq_x_u16_f16, 8798, -1},
  { ARM::BI__builtin_arm_mve_vcvtpq_x_u32_f32, 8815, -1},
  { ARM::BI__builtin_arm_mve_vcvtq_f16_s16, 8838, 8832},
  { ARM::BI__builtin_arm_mve_vcvtq_f16_u16, 8852, 8832},
  { ARM::BI__builtin_arm_mve_vcvtq_f32_s32, 8866, 8832},
  { ARM::BI__builtin_arm_mve_vcvtq_f32_u32, 8880, 8832},
  { ARM::BI__builtin_arm_mve_vcvtq_m_f16_s16, 8902, 8894},
  { ARM::BI__builtin_arm_mve_vcvtq_m_f16_u16, 8918, 8894},
  { ARM::BI__builtin_arm_mve_vcvtq_m_f32_s32, 8934, 8894},
  { ARM::BI__builtin_arm_mve_vcvtq_m_f32_u32, 8950, 8894},
  { ARM::BI__builtin_arm_mve_vcvtq_m_n_f16_s16, 8976, 8966},
  { ARM::BI__builtin_arm_mve_vcvtq_m_n_f16_u16, 8994, 8966},
  { ARM::BI__builtin_arm_mve_vcvtq_m_n_f32_s32, 9012, 8966},
  { ARM::BI__builtin_arm_mve_vcvtq_m_n_f32_u32, 9030, 8966},
  { ARM::BI__builtin_arm_mve_vcvtq_m_n_s16_f16, 9048, 8966},
  { ARM::BI__builtin_arm_mve_vcvtq_m_n_s32_f32, 9066, 8966},
  { ARM::BI__builtin_arm_mve_vcvtq_m_n_u16_f16, 9084, 8966},
  { ARM::BI__builtin_arm_mve_vcvtq_m_n_u32_f32, 9102, 8966},
  { ARM::BI__builtin_arm_mve_vcvtq_m_s16_f16, 9120, 8894},
  { ARM::BI__builtin_arm_mve_vcvtq_m_s32_f32, 9136, 8894},
  { ARM::BI__builtin_arm_mve_vcvtq_m_u16_f16, 9152, 8894},
  { ARM::BI__builtin_arm_mve_vcvtq_m_u32_f32, 9168, 8894},
  { ARM::BI__builtin_arm_mve_vcvtq_n_f16_s16, 9192, 9184},
  { ARM::BI__builtin_arm_mve_vcvtq_n_f16_u16, 9208, 9184},
  { ARM::BI__builtin_arm_mve_vcvtq_n_f32_s32, 9224, 9184},
  { ARM::BI__builtin_arm_mve_vcvtq_n_f32_u32, 9240, 9184},
  { ARM::BI__builtin_arm_mve_vcvtq_n_s16_f16, 9256, -1},
  { ARM::BI__builtin_arm_mve_vcvtq_n_s32_f32, 9272, -1},
  { ARM::BI__builtin_arm_mve_vcvtq_n_u16_f16, 9288, -1},
  { ARM::BI__builtin_arm_mve_vcvtq_n_u32_f32, 9304, -1},
  { ARM::BI__builtin_arm_mve_vcvtq_s16_f16, 9320, -1},
  { ARM::BI__builtin_arm_mve_vcvtq_s32_f32, 9334, -1},
  { ARM::BI__builtin_arm_mve_vcvtq_u16_f16, 9348, -1},
  { ARM::BI__builtin_arm_mve_vcvtq_u32_f32, 9362, -1},
  { ARM::BI__builtin_arm_mve_vcvtq_x_f16_s16, 9384, 9376},
  { ARM::BI__builtin_arm_mve_vcvtq_x_f16_u16, 9400, 9376},
  { ARM::BI__builtin_arm_mve_vcvtq_x_f32_s32, 9416, 9376},
  { ARM::BI__builtin_arm_mve_vcvtq_x_f32_u32, 9432, 9376},
  { ARM::BI__builtin_arm_mve_vcvtq_x_n_f16_s16, 9458, 9448},
  { ARM::BI__builtin_arm_mve_vcvtq_x_n_f16_u16, 9476, 9448},
  { ARM::BI__builtin_arm_mve_vcvtq_x_n_f32_s32, 9494, 9448},
  { ARM::BI__builtin_arm_mve_vcvtq_x_n_f32_u32, 9512, 9448},
  { ARM::BI__builtin_arm_mve_vcvtq_x_n_s16_f16, 9530, -1},
  { ARM::BI__builtin_arm_mve_vcvtq_x_n_s32_f32, 9548, -1},
  { ARM::BI__builtin_arm_mve_vcvtq_x_n_u16_f16, 9566, -1},
  { ARM::BI__builtin_arm_mve_vcvtq_x_n_u32_f32, 9584, -1},
  { ARM::BI__builtin_arm_mve_vcvtq_x_s16_f16, 9602, -1},
  { ARM::BI__builtin_arm_mve_vcvtq_x_s32_f32, 9618, -1},
  { ARM::BI__builtin_arm_mve_vcvtq_x_u16_f16, 9634, -1},
  { ARM::BI__builtin_arm_mve_vcvtq_x_u32_f32, 9650, -1},
  { ARM::BI__builtin_arm_mve_vcvttq_f16_f32, 9666, -1},
  { ARM::BI__builtin_arm_mve_vcvttq_f32_f16, 9681, -1},
  { ARM::BI__builtin_arm_mve_vcvttq_m_f16_f32, 9696, -1},
  { ARM::BI__builtin_arm_mve_vcvttq_m_f32_f16, 9713, -1},
  { ARM::BI__builtin_arm_mve_vcvttq_x_f32_f16, 9730, -1},
  { ARM::BI__builtin_arm_mve_vddupq_m_n_u16, 9756, 9747},
  { ARM::BI__builtin_arm_mve_vddupq_m_n_u32, 9771, 9747},
  { ARM::BI__builtin_arm_mve_vddupq_m_n_u8, 9786, 9747},
  { ARM::BI__builtin_arm_mve_vddupq_m_wb_u16, 9800, 9747},
  { ARM::BI__builtin_arm_mve_vddupq_m_wb_u32, 9816, 9747},
  { ARM::BI__builtin_arm_mve_vddupq_m_wb_u8, 9832, 9747},
  { ARM::BI__builtin_arm_mve_vddupq_n_u16, 9858, 9847},
  { ARM::BI__builtin_arm_mve_vddupq_n_u32, 9882, 9871},
  { ARM::BI__builtin_arm_mve_vddupq_n_u8, 9905, 9895},
  { ARM::BI__builtin_arm_mve_vddupq_wb_u16, 9917, 9847},
  { ARM::BI__builtin_arm_mve_vddupq_wb_u32, 9931, 9871},
  { ARM::BI__builtin_arm_mve_vddupq_wb_u8, 9945, 9895},
  { ARM::BI__builtin_arm_mve_vddupq_x_n_u16, 9971, 9958},
  { ARM::BI__builtin_arm_mve_vddupq_x_n_u32, 9999, 9986},
  { ARM::BI__builtin_arm_mve_vddupq_x_n_u8, 10026, 10014},
  { ARM::BI__builtin_arm_mve_vddupq_x_wb_u16, 10040, 9958},
  { ARM::BI__builtin_arm_mve_vddupq_x_wb_u32, 10056, 9986},
  { ARM::BI__builtin_arm_mve_vddupq_x_wb_u8, 10072, 10014},
  { ARM::BI__builtin_arm_mve_vdupq_m_n_f16, 10095, 10087},
  { ARM::BI__builtin_arm_mve_vdupq_m_n_f32, 10109, 10087},
  { ARM::BI__builtin_arm_mve_vdupq_m_n_s16, 10123, 10087},
  { ARM::BI__builtin_arm_mve_vdupq_m_n_s32, 10137, 10087},
  { ARM::BI__builtin_arm_mve_vdupq_m_n_s8, 10151, 10087},
  { ARM::BI__builtin_arm_mve_vdupq_m_n_u16, 10164, 10087},
  { ARM::BI__builtin_arm_mve_vdupq_m_n_u32, 10178, 10087},
  { ARM::BI__builtin_arm_mve_vdupq_m_n_u8, 10192, 10087},
  { ARM::BI__builtin_arm_mve_vdupq_n_f16, 10205, -1},
  { ARM::BI__builtin_arm_mve_vdupq_n_f32, 10217, -1},
  { ARM::BI__builtin_arm_mve_vdupq_n_s16, 10229, -1},
  { ARM::BI__builtin_arm_mve_vdupq_n_s32, 10241, -1},
  { ARM::BI__builtin_arm_mve_vdupq_n_s8, 10253, -1},
  { ARM::BI__builtin_arm_mve_vdupq_n_u16, 10264, -1},
  { ARM::BI__builtin_arm_mve_vdupq_n_u32, 10276, -1},
  { ARM::BI__builtin_arm_mve_vdupq_n_u8, 10288, -1},
  { ARM::BI__builtin_arm_mve_vdupq_x_n_f16, 10299, -1},
  { ARM::BI__builtin_arm_mve_vdupq_x_n_f32, 10313, -1},
  { ARM::BI__builtin_arm_mve_vdupq_x_n_s16, 10327, -1},
  { ARM::BI__builtin_arm_mve_vdupq_x_n_s32, 10341, -1},
  { ARM::BI__builtin_arm_mve_vdupq_x_n_s8, 10355, -1},
  { ARM::BI__builtin_arm_mve_vdupq_x_n_u16, 10368, -1},
  { ARM::BI__builtin_arm_mve_vdupq_x_n_u32, 10382, -1},
  { ARM::BI__builtin_arm_mve_vdupq_x_n_u8, 10396, -1},
  { ARM::BI__builtin_arm_mve_vdwdupq_m_n_u16, 10419, 10409},
  { ARM::BI__builtin_arm_mve_vdwdupq_m_n_u32, 10435, 10409},
  { ARM::BI__builtin_arm_mve_vdwdupq_m_n_u8, 10451, 10409},
  { ARM::BI__builtin_arm_mve_vdwdupq_m_wb_u16, 10466, 10409},
  { ARM::BI__builtin_arm_mve_vdwdupq_m_wb_u32, 10483, 10409},
  { ARM::BI__builtin_arm_mve_vdwdupq_m_wb_u8, 10500, 10409},
  { ARM::BI__builtin_arm_mve_vdwdupq_n_u16, 10528, 10516},
  { ARM::BI__builtin_arm_mve_vdwdupq_n_u32, 10554, 10542},
  { ARM::BI__builtin_arm_mve_vdwdupq_n_u8, 10579, 10568},
  { ARM::BI__builtin_arm_mve_vdwdupq_wb_u16, 10592, 10516},
  { ARM::BI__builtin_arm_mve_vdwdupq_wb_u32, 10607, 10542},
  { ARM::BI__builtin_arm_mve_vdwdupq_wb_u8, 10622, 10568},
  { ARM::BI__builtin_arm_mve_vdwdupq_x_n_u16, 10650, 10636},
  { ARM::BI__builtin_arm_mve_vdwdupq_x_n_u32, 10680, 10666},
  { ARM::BI__builtin_arm_mve_vdwdupq_x_n_u8, 10709, 10696},
  { ARM::BI__builtin_arm_mve_vdwdupq_x_wb_u16, 10724, 10636},
  { ARM::BI__builtin_arm_mve_vdwdupq_x_wb_u32, 10741, 10666},
  { ARM::BI__builtin_arm_mve_vdwdupq_x_wb_u8, 10758, 10696},
  { ARM::BI__builtin_arm_mve_veorq_f16, 10780, 10774},
  { ARM::BI__builtin_arm_mve_veorq_f32, 10790, 10774},
  { ARM::BI__builtin_arm_mve_veorq_m_f16, 10808, 10800},
  { ARM::BI__builtin_arm_mve_veorq_m_f32, 10820, 10800},
  { ARM::BI__builtin_arm_mve_veorq_m_s16, 10832, 10800},
  { ARM::BI__builtin_arm_mve_veorq_m_s32, 10844, 10800},
  { ARM::BI__builtin_arm_mve_veorq_m_s8, 10856, 10800},
  { ARM::BI__builtin_arm_mve_veorq_m_u16, 10867, 10800},
  { ARM::BI__builtin_arm_mve_veorq_m_u32, 10879, 10800},
  { ARM::BI__builtin_arm_mve_veorq_m_u8, 10891, 10800},
  { ARM::BI__builtin_arm_mve_veorq_s16, 10902, 10774},
  { ARM::BI__builtin_arm_mve_veorq_s32, 10912, 10774},
  { ARM::BI__builtin_arm_mve_veorq_s8, 10922, 10774},
  { ARM::BI__builtin_arm_mve_veorq_u16, 10931, 10774},
  { ARM::BI__builtin_arm_mve_veorq_u32, 10941, 10774},
  { ARM::BI__builtin_arm_mve_veorq_u8, 10951, 10774},
  { ARM::BI__builtin_arm_mve_veorq_x_f16, 10968, 10960},
  { ARM::BI__builtin_arm_mve_veorq_x_f32, 10980, 10960},
  { ARM::BI__builtin_arm_mve_veorq_x_s16, 10992, 10960},
  { ARM::BI__builtin_arm_mve_veorq_x_s32, 11004, 10960},
  { ARM::BI__builtin_arm_mve_veorq_x_s8, 11016, 10960},
  { ARM::BI__builtin_arm_mve_veorq_x_u16, 11027, 10960},
  { ARM::BI__builtin_arm_mve_veorq_x_u32, 11039, 10960},
  { ARM::BI__builtin_arm_mve_veorq_x_u8, 11051, 10960},
  { ARM::BI__builtin_arm_mve_vfmaq_f16, 11068, 11062},
  { ARM::BI__builtin_arm_mve_vfmaq_f32, 11078, 11062},
  { ARM::BI__builtin_arm_mve_vfmaq_m_f16, 11096, 11088},
  { ARM::BI__builtin_arm_mve_vfmaq_m_f32, 11108, 11088},
  { ARM::BI__builtin_arm_mve_vfmaq_m_n_f16, 11120, 11088},
  { ARM::BI__builtin_arm_mve_vfmaq_m_n_f32, 11134, 11088},
  { ARM::BI__builtin_arm_mve_vfmaq_n_f16, 11148, 11062},
  { ARM::BI__builtin_arm_mve_vfmaq_n_f32, 11160, 11062},
  { ARM::BI__builtin_arm_mve_vfmasq_m_n_f16, 11181, 11172},
  { ARM::BI__builtin_arm_mve_vfmasq_m_n_f32, 11196, 11172},
  { ARM::BI__builtin_arm_mve_vfmasq_n_f16, 11218, 11211},
  { ARM::BI__builtin_arm_mve_vfmasq_n_f32, 11231, 11211},
  { ARM::BI__builtin_arm_mve_vfmsq_f16, 11250, 11244},
  { ARM::BI__builtin_arm_mve_vfmsq_f32, 11260, 11244},
  { ARM::BI__builtin_arm_mve_vfmsq_m_f16, 11278, 11270},
  { ARM::BI__builtin_arm_mve_vfmsq_m_f32, 11290, 11270},
  { ARM::BI__builtin_arm_mve_vgetq_lane_f16, 11313, 11302},
  { ARM::BI__builtin_arm_mve_vgetq_lane_f32, 11328, 11302},
  { ARM::BI__builtin_arm_mve_vgetq_lane_s16, 11343, 11302},
  { ARM::BI__builtin_arm_mve_vgetq_lane_s32, 11358, 11302},
  { ARM::BI__builtin_arm_mve_vgetq_lane_s64, 11373, 11302},
  { ARM::BI__builtin_arm_mve_vgetq_lane_s8, 11388, 11302},
  { ARM::BI__builtin_arm_mve_vgetq_lane_u16, 11402, 11302},
  { ARM::BI__builtin_arm_mve_vgetq_lane_u32, 11417, 11302},
  { ARM::BI__builtin_arm_mve_vgetq_lane_u64, 11432, 11302},
  { ARM::BI__builtin_arm_mve_vgetq_lane_u8, 11447, 11302},
  { ARM::BI__builtin_arm_mve_vhaddq_m_n_s16, 11470, 11461},
  { ARM::BI__builtin_arm_mve_vhaddq_m_n_s32, 11485, 11461},
  { ARM::BI__builtin_arm_mve_vhaddq_m_n_s8, 11500, 11461},
  { ARM::BI__builtin_arm_mve_vhaddq_m_n_u16, 11514, 11461},
  { ARM::BI__builtin_arm_mve_vhaddq_m_n_u32, 11529, 11461},
  { ARM::BI__builtin_arm_mve_vhaddq_m_n_u8, 11544, 11461},
  { ARM::BI__builtin_arm_mve_vhaddq_m_s16, 11558, 11461},
  { ARM::BI__builtin_arm_mve_vhaddq_m_s32, 11571, 11461},
  { ARM::BI__builtin_arm_mve_vhaddq_m_s8, 11584, 11461},
  { ARM::BI__builtin_arm_mve_vhaddq_m_u16, 11596, 11461},
  { ARM::BI__builtin_arm_mve_vhaddq_m_u32, 11609, 11461},
  { ARM::BI__builtin_arm_mve_vhaddq_m_u8, 11622, 11461},
  { ARM::BI__builtin_arm_mve_vhaddq_n_s16, 11641, 11634},
  { ARM::BI__builtin_arm_mve_vhaddq_n_s32, 11654, 11634},
  { ARM::BI__builtin_arm_mve_vhaddq_n_s8, 11667, 11634},
  { ARM::BI__builtin_arm_mve_vhaddq_n_u16, 11679, 11634},
  { ARM::BI__builtin_arm_mve_vhaddq_n_u32, 11692, 11634},
  { ARM::BI__builtin_arm_mve_vhaddq_n_u8, 11705, 11634},
  { ARM::BI__builtin_arm_mve_vhaddq_s16, 11717, 11634},
  { ARM::BI__builtin_arm_mve_vhaddq_s32, 11728, 11634},
  { ARM::BI__builtin_arm_mve_vhaddq_s8, 11739, 11634},
  { ARM::BI__builtin_arm_mve_vhaddq_u16, 11749, 11634},
  { ARM::BI__builtin_arm_mve_vhaddq_u32, 11760, 11634},
  { ARM::BI__builtin_arm_mve_vhaddq_u8, 11771, 11634},
  { ARM::BI__builtin_arm_mve_vhaddq_x_n_s16, 11790, 11781},
  { ARM::BI__builtin_arm_mve_vhaddq_x_n_s32, 11805, 11781},
  { ARM::BI__builtin_arm_mve_vhaddq_x_n_s8, 11820, 11781},
  { ARM::BI__builtin_arm_mve_vhaddq_x_n_u16, 11834, 11781},
  { ARM::BI__builtin_arm_mve_vhaddq_x_n_u32, 11849, 11781},
  { ARM::BI__builtin_arm_mve_vhaddq_x_n_u8, 11864, 11781},
  { ARM::BI__builtin_arm_mve_vhaddq_x_s16, 11878, 11781},
  { ARM::BI__builtin_arm_mve_vhaddq_x_s32, 11891, 11781},
  { ARM::BI__builtin_arm_mve_vhaddq_x_s8, 11904, 11781},
  { ARM::BI__builtin_arm_mve_vhaddq_x_u16, 11916, 11781},
  { ARM::BI__builtin_arm_mve_vhaddq_x_u32, 11929, 11781},
  { ARM::BI__builtin_arm_mve_vhaddq_x_u8, 11942, 11781},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot270_m_s16, 11971, 11954},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot270_m_s32, 11992, 11954},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot270_m_s8, 12013, 11954},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot270_s16, 12048, 12033},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot270_s32, 12067, 12033},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot270_s8, 12086, 12033},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot270_x_s16, 12121, 12104},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot270_x_s32, 12142, 12104},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot270_x_s8, 12163, 12104},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot90_m_s16, 12199, 12183},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot90_m_s32, 12219, 12183},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot90_m_s8, 12239, 12183},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot90_s16, 12272, 12258},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot90_s32, 12290, 12258},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot90_s8, 12308, 12258},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot90_x_s16, 12341, 12325},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot90_x_s32, 12361, 12325},
  { ARM::BI__builtin_arm_mve_vhcaddq_rot90_x_s8, 12381, 12325},
  { ARM::BI__builtin_arm_mve_vhsubq_m_n_s16, 12409, 12400},
  { ARM::BI__builtin_arm_mve_vhsubq_m_n_s32, 12424, 12400},
  { ARM::BI__builtin_arm_mve_vhsubq_m_n_s8, 12439, 12400},
  { ARM::BI__builtin_arm_mve_vhsubq_m_n_u16, 12453, 12400},
  { ARM::BI__builtin_arm_mve_vhsubq_m_n_u32, 12468, 12400},
  { ARM::BI__builtin_arm_mve_vhsubq_m_n_u8, 12483, 12400},
  { ARM::BI__builtin_arm_mve_vhsubq_m_s16, 12497, 12400},
  { ARM::BI__builtin_arm_mve_vhsubq_m_s32, 12510, 12400},
  { ARM::BI__builtin_arm_mve_vhsubq_m_s8, 12523, 12400},
  { ARM::BI__builtin_arm_mve_vhsubq_m_u16, 12535, 12400},
  { ARM::BI__builtin_arm_mve_vhsubq_m_u32, 12548, 12400},
  { ARM::BI__builtin_arm_mve_vhsubq_m_u8, 12561, 12400},
  { ARM::BI__builtin_arm_mve_vhsubq_n_s16, 12580, 12573},
  { ARM::BI__builtin_arm_mve_vhsubq_n_s32, 12593, 12573},
  { ARM::BI__builtin_arm_mve_vhsubq_n_s8, 12606, 12573},
  { ARM::BI__builtin_arm_mve_vhsubq_n_u16, 12618, 12573},
  { ARM::BI__builtin_arm_mve_vhsubq_n_u32, 12631, 12573},
  { ARM::BI__builtin_arm_mve_vhsubq_n_u8, 12644, 12573},
  { ARM::BI__builtin_arm_mve_vhsubq_s16, 12656, 12573},
  { ARM::BI__builtin_arm_mve_vhsubq_s32, 12667, 12573},
  { ARM::BI__builtin_arm_mve_vhsubq_s8, 12678, 12573},
  { ARM::BI__builtin_arm_mve_vhsubq_u16, 12688, 12573},
  { ARM::BI__builtin_arm_mve_vhsubq_u32, 12699, 12573},
  { ARM::BI__builtin_arm_mve_vhsubq_u8, 12710, 12573},
  { ARM::BI__builtin_arm_mve_vhsubq_x_n_s16, 12729, 12720},
  { ARM::BI__builtin_arm_mve_vhsubq_x_n_s32, 12744, 12720},
  { ARM::BI__builtin_arm_mve_vhsubq_x_n_s8, 12759, 12720},
  { ARM::BI__builtin_arm_mve_vhsubq_x_n_u16, 12773, 12720},
  { ARM::BI__builtin_arm_mve_vhsubq_x_n_u32, 12788, 12720},
  { ARM::BI__builtin_arm_mve_vhsubq_x_n_u8, 12803, 12720},
  { ARM::BI__builtin_arm_mve_vhsubq_x_s16, 12817, 12720},
  { ARM::BI__builtin_arm_mve_vhsubq_x_s32, 12830, 12720},
  { ARM::BI__builtin_arm_mve_vhsubq_x_s8, 12843, 12720},
  { ARM::BI__builtin_arm_mve_vhsubq_x_u16, 12855, 12720},
  { ARM::BI__builtin_arm_mve_vhsubq_x_u32, 12868, 12720},
  { ARM::BI__builtin_arm_mve_vhsubq_x_u8, 12881, 12720},
  { ARM::BI__builtin_arm_mve_vidupq_m_n_u16, 12902, 12893},
  { ARM::BI__builtin_arm_mve_vidupq_m_n_u32, 12917, 12893},
  { ARM::BI__builtin_arm_mve_vidupq_m_n_u8, 12932, 12893},
  { ARM::BI__builtin_arm_mve_vidupq_m_wb_u16, 12946, 12893},
  { ARM::BI__builtin_arm_mve_vidupq_m_wb_u32, 12962, 12893},
  { ARM::BI__builtin_arm_mve_vidupq_m_wb_u8, 12978, 12893},
  { ARM::BI__builtin_arm_mve_vidupq_n_u16, 13004, 12993},
  { ARM::BI__builtin_arm_mve_vidupq_n_u32, 13028, 13017},
  { ARM::BI__builtin_arm_mve_vidupq_n_u8, 13051, 13041},
  { ARM::BI__builtin_arm_mve_vidupq_wb_u16, 13063, 12993},
  { ARM::BI__builtin_arm_mve_vidupq_wb_u32, 13077, 13017},
  { ARM::BI__builtin_arm_mve_vidupq_wb_u8, 13091, 13041},
  { ARM::BI__builtin_arm_mve_vidupq_x_n_u16, 13117, 13104},
  { ARM::BI__builtin_arm_mve_vidupq_x_n_u32, 13145, 13132},
  { ARM::BI__builtin_arm_mve_vidupq_x_n_u8, 13172, 13160},
  { ARM::BI__builtin_arm_mve_vidupq_x_wb_u16, 13186, 13104},
  { ARM::BI__builtin_arm_mve_vidupq_x_wb_u32, 13202, 13132},
  { ARM::BI__builtin_arm_mve_vidupq_x_wb_u8, 13218, 13160},
  { ARM::BI__builtin_arm_mve_viwdupq_m_n_u16, 13243, 13233},
  { ARM::BI__builtin_arm_mve_viwdupq_m_n_u32, 13259, 13233},
  { ARM::BI__builtin_arm_mve_viwdupq_m_n_u8, 13275, 13233},
  { ARM::BI__builtin_arm_mve_viwdupq_m_wb_u16, 13290, 13233},
  { ARM::BI__builtin_arm_mve_viwdupq_m_wb_u32, 13307, 13233},
  { ARM::BI__builtin_arm_mve_viwdupq_m_wb_u8, 13324, 13233},
  { ARM::BI__builtin_arm_mve_viwdupq_n_u16, 13352, 13340},
  { ARM::BI__builtin_arm_mve_viwdupq_n_u32, 13378, 13366},
  { ARM::BI__builtin_arm_mve_viwdupq_n_u8, 13403, 13392},
  { ARM::BI__builtin_arm_mve_viwdupq_wb_u16, 13416, 13340},
  { ARM::BI__builtin_arm_mve_viwdupq_wb_u32, 13431, 13366},
  { ARM::BI__builtin_arm_mve_viwdupq_wb_u8, 13446, 13392},
  { ARM::BI__builtin_arm_mve_viwdupq_x_n_u16, 13474, 13460},
  { ARM::BI__builtin_arm_mve_viwdupq_x_n_u32, 13504, 13490},
  { ARM::BI__builtin_arm_mve_viwdupq_x_n_u8, 13533, 13520},
  { ARM::BI__builtin_arm_mve_viwdupq_x_wb_u16, 13548, 13460},
  { ARM::BI__builtin_arm_mve_viwdupq_x_wb_u32, 13565, 13490},
  { ARM::BI__builtin_arm_mve_viwdupq_x_wb_u8, 13582, 13520},
  { ARM::BI__builtin_arm_mve_vld1q_f16, 13604, 13598},
  { ARM::BI__builtin_arm_mve_vld1q_f32, 13614, 13598},
  { ARM::BI__builtin_arm_mve_vld1q_s16, 13624, 13598},
  { ARM::BI__builtin_arm_mve_vld1q_s32, 13634, 13598},
  { ARM::BI__builtin_arm_mve_vld1q_s8, 13644, 13598},
  { ARM::BI__builtin_arm_mve_vld1q_u16, 13653, 13598},
  { ARM::BI__builtin_arm_mve_vld1q_u32, 13663, 13598},
  { ARM::BI__builtin_arm_mve_vld1q_u8, 13673, 13598},
  { ARM::BI__builtin_arm_mve_vld1q_z_f16, 13690, 13682},
  { ARM::BI__builtin_arm_mve_vld1q_z_f32, 13702, 13682},
  { ARM::BI__builtin_arm_mve_vld1q_z_s16, 13714, 13682},
  { ARM::BI__builtin_arm_mve_vld1q_z_s32, 13726, 13682},
  { ARM::BI__builtin_arm_mve_vld1q_z_s8, 13738, 13682},
  { ARM::BI__builtin_arm_mve_vld1q_z_u16, 13749, 13682},
  { ARM::BI__builtin_arm_mve_vld1q_z_u32, 13761, 13682},
  { ARM::BI__builtin_arm_mve_vld1q_z_u8, 13773, 13682},
  { ARM::BI__builtin_arm_mve_vld2q_f16, 13790, 13784},
  { ARM::BI__builtin_arm_mve_vld2q_f32, 13800, 13784},
  { ARM::BI__builtin_arm_mve_vld2q_s16, 13810, 13784},
  { ARM::BI__builtin_arm_mve_vld2q_s32, 13820, 13784},
  { ARM::BI__builtin_arm_mve_vld2q_s8, 13830, 13784},
  { ARM::BI__builtin_arm_mve_vld2q_u16, 13839, 13784},
  { ARM::BI__builtin_arm_mve_vld2q_u32, 13849, 13784},
  { ARM::BI__builtin_arm_mve_vld2q_u8, 13859, 13784},
  { ARM::BI__builtin_arm_mve_vld4q_f16, 13874, 13868},
  { ARM::BI__builtin_arm_mve_vld4q_f32, 13884, 13868},
  { ARM::BI__builtin_arm_mve_vld4q_s16, 13894, 13868},
  { ARM::BI__builtin_arm_mve_vld4q_s32, 13904, 13868},
  { ARM::BI__builtin_arm_mve_vld4q_s8, 13914, 13868},
  { ARM::BI__builtin_arm_mve_vld4q_u16, 13923, 13868},
  { ARM::BI__builtin_arm_mve_vld4q_u32, 13933, 13868},
  { ARM::BI__builtin_arm_mve_vld4q_u8, 13943, 13868},
  { ARM::BI__builtin_arm_mve_vldrbq_gather_offset_s16, 13973, 13952},
  { ARM::BI__builtin_arm_mve_vldrbq_gather_offset_s32, 13998, 13952},
  { ARM::BI__builtin_arm_mve_vldrbq_gather_offset_s8, 14023, 13952},
  { ARM::BI__builtin_arm_mve_vldrbq_gather_offset_u16, 14047, 13952},
  { ARM::BI__builtin_arm_mve_vldrbq_gather_offset_u32, 14072, 13952},
  { ARM::BI__builtin_arm_mve_vldrbq_gather_offset_u8, 14097, 13952},
  { ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_s16, 14144, 14121},
  { ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_s32, 14171, 14121},
  { ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_s8, 14198, 14121},
  { ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_u16, 14224, 14121},
  { ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_u32, 14251, 14121},
  { ARM::BI__builtin_arm_mve_vldrbq_gather_offset_z_u8, 14278, 14121},
  { ARM::BI__builtin_arm_mve_vldrbq_s16, 14304, -1},
  { ARM::BI__builtin_arm_mve_vldrbq_s32, 14315, -1},
  { ARM::BI__builtin_arm_mve_vldrbq_s8, 14326, -1},
  { ARM::BI__builtin_arm_mve_vldrbq_u16, 14336, -1},
  { ARM::BI__builtin_arm_mve_vldrbq_u32, 14347, -1},
  { ARM::BI__builtin_arm_mve_vldrbq_u8, 14358, -1},
  { ARM::BI__builtin_arm_mve_vldrbq_z_s16, 14368, -1},
  { ARM::BI__builtin_arm_mve_vldrbq_z_s32, 14381, -1},
  { ARM::BI__builtin_arm_mve_vldrbq_z_s8, 14394, -1},
  { ARM::BI__builtin_arm_mve_vldrbq_z_u16, 14406, -1},
  { ARM::BI__builtin_arm_mve_vldrbq_z_u32, 14419, -1},
  { ARM::BI__builtin_arm_mve_vldrbq_z_u8, 14432, -1},
  { ARM::BI__builtin_arm_mve_vldrdq_gather_base_s64, 14444, -1},
  { ARM::BI__builtin_arm_mve_vldrdq_gather_base_u64, 14467, -1},
  { ARM::BI__builtin_arm_mve_vldrdq_gather_base_wb_s64, 14490, -1},
  { ARM::BI__builtin_arm_mve_vldrdq_gather_base_wb_u64, 14516, -1},
  { ARM::BI__builtin_arm_mve_vldrdq_gather_base_wb_z_s64, 14542, -1},
  { ARM::BI__builtin_arm_mve_vldrdq_gather_base_wb_z_u64, 14570, -1},
  { ARM::BI__builtin_arm_mve_vldrdq_gather_base_z_s64, 14598, -1},
  { ARM::BI__builtin_arm_mve_vldrdq_gather_base_z_u64, 14623, -1},
  { ARM::BI__builtin_arm_mve_vldrdq_gather_offset_s64, 14669, 14648},
  { ARM::BI__builtin_arm_mve_vldrdq_gather_offset_u64, 14694, 14648},
  { ARM::BI__builtin_arm_mve_vldrdq_gather_offset_z_s64, 14742, 14719},
  { ARM::BI__builtin_arm_mve_vldrdq_gather_offset_z_u64, 14769, 14719},
  { ARM::BI__builtin_arm_mve_vldrdq_gather_shifted_offset_s64, 14825, 14796},
  { ARM::BI__builtin_arm_mve_vldrdq_gather_shifted_offset_u64, 14858, 14796},
  { ARM::BI__builtin_arm_mve_vldrdq_gather_shifted_offset_z_s64, 14922, 14891},
  { ARM::BI__builtin_arm_mve_vldrdq_gather_shifted_offset_z_u64, 14957, 14891},
  { ARM::BI__builtin_arm_mve_vldrhq_f16, 14992, -1},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_offset_f16, 15024, 15003},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_offset_s16, 15049, 15003},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_offset_s32, 15074, 15003},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_offset_u16, 15099, 15003},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_offset_u32, 15124, 15003},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_offset_z_f16, 15172, 15149},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_offset_z_s16, 15199, 15149},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_offset_z_s32, 15226, 15149},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_offset_z_u16, 15253, 15149},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_offset_z_u32, 15280, 15149},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_f16, 15336, 15307},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_s16, 15369, 15307},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_s32, 15402, 15307},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_u16, 15435, 15307},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_u32, 15468, 15307},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_z_f16, 15532, 15501},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_z_s16, 15567, 15501},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_z_s32, 15602, 15501},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_z_u16, 15637, 15501},
  { ARM::BI__builtin_arm_mve_vldrhq_gather_shifted_offset_z_u32, 15672, 15501},
  { ARM::BI__builtin_arm_mve_vldrhq_s16, 15707, -1},
  { ARM::BI__builtin_arm_mve_vldrhq_s32, 15718, -1},
  { ARM::BI__builtin_arm_mve_vldrhq_u16, 15729, -1},
  { ARM::BI__builtin_arm_mve_vldrhq_u32, 15740, -1},
  { ARM::BI__builtin_arm_mve_vldrhq_z_f16, 15751, -1},
  { ARM::BI__builtin_arm_mve_vldrhq_z_s16, 15764, -1},
  { ARM::BI__builtin_arm_mve_vldrhq_z_s32, 15777, -1},
  { ARM::BI__builtin_arm_mve_vldrhq_z_u16, 15790, -1},
  { ARM::BI__builtin_arm_mve_vldrhq_z_u32, 15803, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_f32, 15816, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_base_f32, 15827, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_base_s32, 15850, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_base_u32, 15873, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_f32, 15896, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_s32, 15922, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_u32, 15948, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_z_f32, 15974, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_z_s32, 16002, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_base_wb_z_u32, 16030, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_base_z_f32, 16058, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_base_z_s32, 16083, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_base_z_u32, 16108, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_offset_f32, 16154, 16133},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_offset_s32, 16179, 16133},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_offset_u32, 16204, 16133},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_offset_z_f32, 16252, 16229},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_offset_z_s32, 16279, 16229},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_offset_z_u32, 16306, 16229},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_f32, 16362, 16333},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_s32, 16395, 16333},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_u32, 16428, 16333},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_z_f32, 16492, 16461},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_z_s32, 16527, 16461},
  { ARM::BI__builtin_arm_mve_vldrwq_gather_shifted_offset_z_u32, 16562, 16461},
  { ARM::BI__builtin_arm_mve_vldrwq_s32, 16597, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_u32, 16608, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_z_f32, 16619, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_z_s32, 16632, -1},
  { ARM::BI__builtin_arm_mve_vldrwq_z_u32, 16645, -1},
  { ARM::BI__builtin_arm_mve_vmaxaq_m_s16, 16667, 16658},
  { ARM::BI__builtin_arm_mve_vmaxaq_m_s32, 16680, 16658},
  { ARM::BI__builtin_arm_mve_vmaxaq_m_s8, 16693, 16658},
  { ARM::BI__builtin_arm_mve_vmaxaq_s16, 16712, 16705},
  { ARM::BI__builtin_arm_mve_vmaxaq_s32, 16723, 16705},
  { ARM::BI__builtin_arm_mve_vmaxaq_s8, 16734, 16705},
  { ARM::BI__builtin_arm_mve_vmaxavq_p_s16, 16754, 16744},
  { ARM::BI__builtin_arm_mve_vmaxavq_p_s32, 16768, 16744},
  { ARM::BI__builtin_arm_mve_vmaxavq_p_s8, 16782, 16744},
  { ARM::BI__builtin_arm_mve_vmaxavq_s16, 16803, 16795},
  { ARM::BI__builtin_arm_mve_vmaxavq_s32, 16815, 16795},
  { ARM::BI__builtin_arm_mve_vmaxavq_s8, 16827, 16795},
  { ARM::BI__builtin_arm_mve_vmaxnmaq_f16, 16847, 16838},
  { ARM::BI__builtin_arm_mve_vmaxnmaq_f32, 16860, 16838},
  { ARM::BI__builtin_arm_mve_vmaxnmaq_m_f16, 16884, 16873},
  { ARM::BI__builtin_arm_mve_vmaxnmaq_m_f32, 16899, 16873},
  { ARM::BI__builtin_arm_mve_vmaxnmavq_f16, 16924, 16914},
  { ARM::BI__builtin_arm_mve_vmaxnmavq_f32, 16938, 16914},
  { ARM::BI__builtin_arm_mve_vmaxnmavq_p_f16, 16964, 16952},
  { ARM::BI__builtin_arm_mve_vmaxnmavq_p_f32, 16980, 16952},
  { ARM::BI__builtin_arm_mve_vmaxnmq_f16, 17004, 16996},
  { ARM::BI__builtin_arm_mve_vmaxnmq_f32, 17016, 16996},
  { ARM::BI__builtin_arm_mve_vmaxnmq_m_f16, 17038, 17028},
  { ARM::BI__builtin_arm_mve_vmaxnmq_m_f32, 17052, 17028},
  { ARM::BI__builtin_arm_mve_vmaxnmq_x_f16, 17076, 17066},
  { ARM::BI__builtin_arm_mve_vmaxnmq_x_f32, 17090, 17066},
  { ARM::BI__builtin_arm_mve_vmaxnmvq_f16, 17113, 17104},
  { ARM::BI__builtin_arm_mve_vmaxnmvq_f32, 17126, 17104},
  { ARM::BI__builtin_arm_mve_vmaxnmvq_p_f16, 17150, 17139},
  { ARM::BI__builtin_arm_mve_vmaxnmvq_p_f32, 17165, 17139},
  { ARM::BI__builtin_arm_mve_vmaxq_m_s16, 17188, 17180},
  { ARM::BI__builtin_arm_mve_vmaxq_m_s32, 17200, 17180},
  { ARM::BI__builtin_arm_mve_vmaxq_m_s8, 17212, 17180},
  { ARM::BI__builtin_arm_mve_vmaxq_m_u16, 17223, 17180},
  { ARM::BI__builtin_arm_mve_vmaxq_m_u32, 17235, 17180},
  { ARM::BI__builtin_arm_mve_vmaxq_m_u8, 17247, 17180},
  { ARM::BI__builtin_arm_mve_vmaxq_s16, 17264, 17258},
  { ARM::BI__builtin_arm_mve_vmaxq_s32, 17274, 17258},
  { ARM::BI__builtin_arm_mve_vmaxq_s8, 17284, 17258},
  { ARM::BI__builtin_arm_mve_vmaxq_u16, 17293, 17258},
  { ARM::BI__builtin_arm_mve_vmaxq_u32, 17303, 17258},
  { ARM::BI__builtin_arm_mve_vmaxq_u8, 17313, 17258},
  { ARM::BI__builtin_arm_mve_vmaxq_x_s16, 17330, 17322},
  { ARM::BI__builtin_arm_mve_vmaxq_x_s32, 17342, 17322},
  { ARM::BI__builtin_arm_mve_vmaxq_x_s8, 17354, 17322},
  { ARM::BI__builtin_arm_mve_vmaxq_x_u16, 17365, 17322},
  { ARM::BI__builtin_arm_mve_vmaxq_x_u32, 17377, 17322},
  { ARM::BI__builtin_arm_mve_vmaxq_x_u8, 17389, 17322},
  { ARM::BI__builtin_arm_mve_vmaxvq_p_s16, 17409, 17400},
  { ARM::BI__builtin_arm_mve_vmaxvq_p_s32, 17422, 17400},
  { ARM::BI__builtin_arm_mve_vmaxvq_p_s8, 17435, 17400},
  { ARM::BI__builtin_arm_mve_vmaxvq_p_u16, 17447, 17400},
  { ARM::BI__builtin_arm_mve_vmaxvq_p_u32, 17460, 17400},
  { ARM::BI__builtin_arm_mve_vmaxvq_p_u8, 17473, 17400},
  { ARM::BI__builtin_arm_mve_vmaxvq_s16, 17492, 17485},
  { ARM::BI__builtin_arm_mve_vmaxvq_s32, 17503, 17485},
  { ARM::BI__builtin_arm_mve_vmaxvq_s8, 17514, 17485},
  { ARM::BI__builtin_arm_mve_vmaxvq_u16, 17524, 17485},
  { ARM::BI__builtin_arm_mve_vmaxvq_u32, 17535, 17485},
  { ARM::BI__builtin_arm_mve_vmaxvq_u8, 17546, 17485},
  { ARM::BI__builtin_arm_mve_vminaq_m_s16, 17565, 17556},
  { ARM::BI__builtin_arm_mve_vminaq_m_s32, 17578, 17556},
  { ARM::BI__builtin_arm_mve_vminaq_m_s8, 17591, 17556},
  { ARM::BI__builtin_arm_mve_vminaq_s16, 17610, 17603},
  { ARM::BI__builtin_arm_mve_vminaq_s32, 17621, 17603},
  { ARM::BI__builtin_arm_mve_vminaq_s8, 17632, 17603},
  { ARM::BI__builtin_arm_mve_vminavq_p_s16, 17652, 17642},
  { ARM::BI__builtin_arm_mve_vminavq_p_s32, 17666, 17642},
  { ARM::BI__builtin_arm_mve_vminavq_p_s8, 17680, 17642},
  { ARM::BI__builtin_arm_mve_vminavq_s16, 17701, 17693},
  { ARM::BI__builtin_arm_mve_vminavq_s32, 17713, 17693},
  { ARM::BI__builtin_arm_mve_vminavq_s8, 17725, 17693},
  { ARM::BI__builtin_arm_mve_vminnmaq_f16, 17745, 17736},
  { ARM::BI__builtin_arm_mve_vminnmaq_f32, 17758, 17736},
  { ARM::BI__builtin_arm_mve_vminnmaq_m_f16, 17782, 17771},
  { ARM::BI__builtin_arm_mve_vminnmaq_m_f32, 17797, 17771},
  { ARM::BI__builtin_arm_mve_vminnmavq_f16, 17822, 17812},
  { ARM::BI__builtin_arm_mve_vminnmavq_f32, 17836, 17812},
  { ARM::BI__builtin_arm_mve_vminnmavq_p_f16, 17862, 17850},
  { ARM::BI__builtin_arm_mve_vminnmavq_p_f32, 17878, 17850},
  { ARM::BI__builtin_arm_mve_vminnmq_f16, 17902, 17894},
  { ARM::BI__builtin_arm_mve_vminnmq_f32, 17914, 17894},
  { ARM::BI__builtin_arm_mve_vminnmq_m_f16, 17936, 17926},
  { ARM::BI__builtin_arm_mve_vminnmq_m_f32, 17950, 17926},
  { ARM::BI__builtin_arm_mve_vminnmq_x_f16, 17974, 17964},
  { ARM::BI__builtin_arm_mve_vminnmq_x_f32, 17988, 17964},
  { ARM::BI__builtin_arm_mve_vminnmvq_f16, 18011, 18002},
  { ARM::BI__builtin_arm_mve_vminnmvq_f32, 18024, 18002},
  { ARM::BI__builtin_arm_mve_vminnmvq_p_f16, 18048, 18037},
  { ARM::BI__builtin_arm_mve_vminnmvq_p_f32, 18063, 18037},
  { ARM::BI__builtin_arm_mve_vminq_m_s16, 18086, 18078},
  { ARM::BI__builtin_arm_mve_vminq_m_s32, 18098, 18078},
  { ARM::BI__builtin_arm_mve_vminq_m_s8, 18110, 18078},
  { ARM::BI__builtin_arm_mve_vminq_m_u16, 18121, 18078},
  { ARM::BI__builtin_arm_mve_vminq_m_u32, 18133, 18078},
  { ARM::BI__builtin_arm_mve_vminq_m_u8, 18145, 18078},
  { ARM::BI__builtin_arm_mve_vminq_s16, 18162, 18156},
  { ARM::BI__builtin_arm_mve_vminq_s32, 18172, 18156},
  { ARM::BI__builtin_arm_mve_vminq_s8, 18182, 18156},
  { ARM::BI__builtin_arm_mve_vminq_u16, 18191, 18156},
  { ARM::BI__builtin_arm_mve_vminq_u32, 18201, 18156},
  { ARM::BI__builtin_arm_mve_vminq_u8, 18211, 18156},
  { ARM::BI__builtin_arm_mve_vminq_x_s16, 18228, 18220},
  { ARM::BI__builtin_arm_mve_vminq_x_s32, 18240, 18220},
  { ARM::BI__builtin_arm_mve_vminq_x_s8, 18252, 18220},
  { ARM::BI__builtin_arm_mve_vminq_x_u16, 18263, 18220},
  { ARM::BI__builtin_arm_mve_vminq_x_u32, 18275, 18220},
  { ARM::BI__builtin_arm_mve_vminq_x_u8, 18287, 18220},
  { ARM::BI__builtin_arm_mve_vminvq_p_s16, 18307, 18298},
  { ARM::BI__builtin_arm_mve_vminvq_p_s32, 18320, 18298},
  { ARM::BI__builtin_arm_mve_vminvq_p_s8, 18333, 18298},
  { ARM::BI__builtin_arm_mve_vminvq_p_u16, 18345, 18298},
  { ARM::BI__builtin_arm_mve_vminvq_p_u32, 18358, 18298},
  { ARM::BI__builtin_arm_mve_vminvq_p_u8, 18371, 18298},
  { ARM::BI__builtin_arm_mve_vminvq_s16, 18390, 18383},
  { ARM::BI__builtin_arm_mve_vminvq_s32, 18401, 18383},
  { ARM::BI__builtin_arm_mve_vminvq_s8, 18412, 18383},
  { ARM::BI__builtin_arm_mve_vminvq_u16, 18422, 18383},
  { ARM::BI__builtin_arm_mve_vminvq_u32, 18433, 18383},
  { ARM::BI__builtin_arm_mve_vminvq_u8, 18444, 18383},
  { ARM::BI__builtin_arm_mve_vmladavaq_p_s16, 18466, 18454},
  { ARM::BI__builtin_arm_mve_vmladavaq_p_s32, 18482, 18454},
  { ARM::BI__builtin_arm_mve_vmladavaq_p_s8, 18498, 18454},
  { ARM::BI__builtin_arm_mve_vmladavaq_p_u16, 18513, 18454},
  { ARM::BI__builtin_arm_mve_vmladavaq_p_u32, 18529, 18454},
  { ARM::BI__builtin_arm_mve_vmladavaq_p_u8, 18545, 18454},
  { ARM::BI__builtin_arm_mve_vmladavaq_s16, 18570, 18560},
  { ARM::BI__builtin_arm_mve_vmladavaq_s32, 18584, 18560},
  { ARM::BI__builtin_arm_mve_vmladavaq_s8, 18598, 18560},
  { ARM::BI__builtin_arm_mve_vmladavaq_u16, 18611, 18560},
  { ARM::BI__builtin_arm_mve_vmladavaq_u32, 18625, 18560},
  { ARM::BI__builtin_arm_mve_vmladavaq_u8, 18639, 18560},
  { ARM::BI__builtin_arm_mve_vmladavaxq_p_s16, 18665, 18652},
  { ARM::BI__builtin_arm_mve_vmladavaxq_p_s32, 18682, 18652},
  { ARM::BI__builtin_arm_mve_vmladavaxq_p_s8, 18699, 18652},
  { ARM::BI__builtin_arm_mve_vmladavaxq_s16, 18726, 18715},
  { ARM::BI__builtin_arm_mve_vmladavaxq_s32, 18741, 18715},
  { ARM::BI__builtin_arm_mve_vmladavaxq_s8, 18756, 18715},
  { ARM::BI__builtin_arm_mve_vmladavq_p_s16, 18781, 18770},
  { ARM::BI__builtin_arm_mve_vmladavq_p_s32, 18796, 18770},
  { ARM::BI__builtin_arm_mve_vmladavq_p_s8, 18811, 18770},
  { ARM::BI__builtin_arm_mve_vmladavq_p_u16, 18825, 18770},
  { ARM::BI__builtin_arm_mve_vmladavq_p_u32, 18840, 18770},
  { ARM::BI__builtin_arm_mve_vmladavq_p_u8, 18855, 18770},
  { ARM::BI__builtin_arm_mve_vmladavq_s16, 18878, 18869},
  { ARM::BI__builtin_arm_mve_vmladavq_s32, 18891, 18869},
  { ARM::BI__builtin_arm_mve_vmladavq_s8, 18904, 18869},
  { ARM::BI__builtin_arm_mve_vmladavq_u16, 18916, 18869},
  { ARM::BI__builtin_arm_mve_vmladavq_u32, 18929, 18869},
  { ARM::BI__builtin_arm_mve_vmladavq_u8, 18942, 18869},
  { ARM::BI__builtin_arm_mve_vmladavxq_p_s16, 18966, 18954},
  { ARM::BI__builtin_arm_mve_vmladavxq_p_s32, 18982, 18954},
  { ARM::BI__builtin_arm_mve_vmladavxq_p_s8, 18998, 18954},
  { ARM::BI__builtin_arm_mve_vmladavxq_s16, 19023, 19013},
  { ARM::BI__builtin_arm_mve_vmladavxq_s32, 19037, 19013},
  { ARM::BI__builtin_arm_mve_vmladavxq_s8, 19051, 19013},
  { ARM::BI__builtin_arm_mve_vmlaldavaq_p_s16, 19077, 19064},
  { ARM::BI__builtin_arm_mve_vmlaldavaq_p_s32, 19094, 19064},
  { ARM::BI__builtin_arm_mve_vmlaldavaq_p_u16, 19111, 19064},
  { ARM::BI__builtin_arm_mve_vmlaldavaq_p_u32, 19128, 19064},
  { ARM::BI__builtin_arm_mve_vmlaldavaq_s16, 19156, 19145},
  { ARM::BI__builtin_arm_mve_vmlaldavaq_s32, 19171, 19145},
  { ARM::BI__builtin_arm_mve_vmlaldavaq_u16, 19186, 19145},
  { ARM::BI__builtin_arm_mve_vmlaldavaq_u32, 19201, 19145},
  { ARM::BI__builtin_arm_mve_vmlaldavaxq_p_s16, 19230, 19216},
  { ARM::BI__builtin_arm_mve_vmlaldavaxq_p_s32, 19248, 19216},
  { ARM::BI__builtin_arm_mve_vmlaldavaxq_s16, 19278, 19266},
  { ARM::BI__builtin_arm_mve_vmlaldavaxq_s32, 19294, 19266},
  { ARM::BI__builtin_arm_mve_vmlaldavq_p_s16, 19322, 19310},
  { ARM::BI__builtin_arm_mve_vmlaldavq_p_s32, 19338, 19310},
  { ARM::BI__builtin_arm_mve_vmlaldavq_p_u16, 19354, 19310},
  { ARM::BI__builtin_arm_mve_vmlaldavq_p_u32, 19370, 19310},
  { ARM::BI__builtin_arm_mve_vmlaldavq_s16, 19396, 19386},
  { ARM::BI__builtin_arm_mve_vmlaldavq_s32, 19410, 19386},
  { ARM::BI__builtin_arm_mve_vmlaldavq_u16, 19424, 19386},
  { ARM::BI__builtin_arm_mve_vmlaldavq_u32, 19438, 19386},
  { ARM::BI__builtin_arm_mve_vmlaldavxq_p_s16, 19465, 19452},
  { ARM::BI__builtin_arm_mve_vmlaldavxq_p_s32, 19482, 19452},
  { ARM::BI__builtin_arm_mve_vmlaldavxq_s16, 19510, 19499},
  { ARM::BI__builtin_arm_mve_vmlaldavxq_s32, 19525, 19499},
  { ARM::BI__builtin_arm_mve_vmlaq_m_n_s16, 19548, 19540},
  { ARM::BI__builtin_arm_mve_vmlaq_m_n_s32, 19562, 19540},
  { ARM::BI__builtin_arm_mve_vmlaq_m_n_s8, 19576, 19540},
  { ARM::BI__builtin_arm_mve_vmlaq_m_n_u16, 19589, 19540},
  { ARM::BI__builtin_arm_mve_vmlaq_m_n_u32, 19603, 19540},
  { ARM::BI__builtin_arm_mve_vmlaq_m_n_u8, 19617, 19540},
  { ARM::BI__builtin_arm_mve_vmlaq_n_s16, 19636, 19630},
  { ARM::BI__builtin_arm_mve_vmlaq_n_s32, 19648, 19630},
  { ARM::BI__builtin_arm_mve_vmlaq_n_s8, 19660, 19630},
  { ARM::BI__builtin_arm_mve_vmlaq_n_u16, 19671, 19630},
  { ARM::BI__builtin_arm_mve_vmlaq_n_u32, 19683, 19630},
  { ARM::BI__builtin_arm_mve_vmlaq_n_u8, 19695, 19630},
  { ARM::BI__builtin_arm_mve_vmlasq_m_n_s16, 19715, 19706},
  { ARM::BI__builtin_arm_mve_vmlasq_m_n_s32, 19730, 19706},
  { ARM::BI__builtin_arm_mve_vmlasq_m_n_s8, 19745, 19706},
  { ARM::BI__builtin_arm_mve_vmlasq_m_n_u16, 19759, 19706},
  { ARM::BI__builtin_arm_mve_vmlasq_m_n_u32, 19774, 19706},
  { ARM::BI__builtin_arm_mve_vmlasq_m_n_u8, 19789, 19706},
  { ARM::BI__builtin_arm_mve_vmlasq_n_s16, 19810, 19803},
  { ARM::BI__builtin_arm_mve_vmlasq_n_s32, 19823, 19803},
  { ARM::BI__builtin_arm_mve_vmlasq_n_s8, 19836, 19803},
  { ARM::BI__builtin_arm_mve_vmlasq_n_u16, 19848, 19803},
  { ARM::BI__builtin_arm_mve_vmlasq_n_u32, 19861, 19803},
  { ARM::BI__builtin_arm_mve_vmlasq_n_u8, 19874, 19803},
  { ARM::BI__builtin_arm_mve_vmlsdavaq_p_s16, 19898, 19886},
  { ARM::BI__builtin_arm_mve_vmlsdavaq_p_s32, 19914, 19886},
  { ARM::BI__builtin_arm_mve_vmlsdavaq_p_s8, 19930, 19886},
  { ARM::BI__builtin_arm_mve_vmlsdavaq_s16, 19955, 19945},
  { ARM::BI__builtin_arm_mve_vmlsdavaq_s32, 19969, 19945},
  { ARM::BI__builtin_arm_mve_vmlsdavaq_s8, 19983, 19945},
  { ARM::BI__builtin_arm_mve_vmlsdavaxq_p_s16, 20009, 19996},
  { ARM::BI__builtin_arm_mve_vmlsdavaxq_p_s32, 20026, 19996},
  { ARM::BI__builtin_arm_mve_vmlsdavaxq_p_s8, 20043, 19996},
  { ARM::BI__builtin_arm_mve_vmlsdavaxq_s16, 20070, 20059},
  { ARM::BI__builtin_arm_mve_vmlsdavaxq_s32, 20085, 20059},
  { ARM::BI__builtin_arm_mve_vmlsdavaxq_s8, 20100, 20059},
  { ARM::BI__builtin_arm_mve_vmlsdavq_p_s16, 20125, 20114},
  { ARM::BI__builtin_arm_mve_vmlsdavq_p_s32, 20140, 20114},
  { ARM::BI__builtin_arm_mve_vmlsdavq_p_s8, 20155, 20114},
  { ARM::BI__builtin_arm_mve_vmlsdavq_s16, 20178, 20169},
  { ARM::BI__builtin_arm_mve_vmlsdavq_s32, 20191, 20169},
  { ARM::BI__builtin_arm_mve_vmlsdavq_s8, 20204, 20169},
  { ARM::BI__builtin_arm_mve_vmlsdavxq_p_s16, 20228, 20216},
  { ARM::BI__builtin_arm_mve_vmlsdavxq_p_s32, 20244, 20216},
  { ARM::BI__builtin_arm_mve_vmlsdavxq_p_s8, 20260, 20216},
  { ARM::BI__builtin_arm_mve_vmlsdavxq_s16, 20285, 20275},
  { ARM::BI__builtin_arm_mve_vmlsdavxq_s32, 20299, 20275},
  { ARM::BI__builtin_arm_mve_vmlsdavxq_s8, 20313, 20275},
  { ARM::BI__builtin_arm_mve_vmlsldavaq_p_s16, 20339, 20326},
  { ARM::BI__builtin_arm_mve_vmlsldavaq_p_s32, 20356, 20326},
  { ARM::BI__builtin_arm_mve_vmlsldavaq_s16, 20384, 20373},
  { ARM::BI__builtin_arm_mve_vmlsldavaq_s32, 20399, 20373},
  { ARM::BI__builtin_arm_mve_vmlsldavaxq_p_s16, 20428, 20414},
  { ARM::BI__builtin_arm_mve_vmlsldavaxq_p_s32, 20446, 20414},
  { ARM::BI__builtin_arm_mve_vmlsldavaxq_s16, 20476, 20464},
  { ARM::BI__builtin_arm_mve_vmlsldavaxq_s32, 20492, 20464},
  { ARM::BI__builtin_arm_mve_vmlsldavq_p_s16, 20520, 20508},
  { ARM::BI__builtin_arm_mve_vmlsldavq_p_s32, 20536, 20508},
  { ARM::BI__builtin_arm_mve_vmlsldavq_s16, 20562, 20552},
  { ARM::BI__builtin_arm_mve_vmlsldavq_s32, 20576, 20552},
  { ARM::BI__builtin_arm_mve_vmlsldavxq_p_s16, 20603, 20590},
  { ARM::BI__builtin_arm_mve_vmlsldavxq_p_s32, 20620, 20590},
  { ARM::BI__builtin_arm_mve_vmlsldavxq_s16, 20648, 20637},
  { ARM::BI__builtin_arm_mve_vmlsldavxq_s32, 20663, 20637},
  { ARM::BI__builtin_arm_mve_vmovlbq_m_s16, 20688, 20678},
  { ARM::BI__builtin_arm_mve_vmovlbq_m_s8, 20702, 20678},
  { ARM::BI__builtin_arm_mve_vmovlbq_m_u16, 20715, 20678},
  { ARM::BI__builtin_arm_mve_vmovlbq_m_u8, 20729, 20678},
  { ARM::BI__builtin_arm_mve_vmovlbq_s16, 20750, 20742},
  { ARM::BI__builtin_arm_mve_vmovlbq_s8, 20762, 20742},
  { ARM::BI__builtin_arm_mve_vmovlbq_u16, 20773, 20742},
  { ARM::BI__builtin_arm_mve_vmovlbq_u8, 20785, 20742},
  { ARM::BI__builtin_arm_mve_vmovlbq_x_s16, 20806, 20796},
  { ARM::BI__builtin_arm_mve_vmovlbq_x_s8, 20820, 20796},
  { ARM::BI__builtin_arm_mve_vmovlbq_x_u16, 20833, 20796},
  { ARM::BI__builtin_arm_mve_vmovlbq_x_u8, 20847, 20796},
  { ARM::BI__builtin_arm_mve_vmovltq_m_s16, 20870, 20860},
  { ARM::BI__builtin_arm_mve_vmovltq_m_s8, 20884, 20860},
  { ARM::BI__builtin_arm_mve_vmovltq_m_u16, 20897, 20860},
  { ARM::BI__builtin_arm_mve_vmovltq_m_u8, 20911, 20860},
  { ARM::BI__builtin_arm_mve_vmovltq_s16, 20932, 20924},
  { ARM::BI__builtin_arm_mve_vmovltq_s8, 20944, 20924},
  { ARM::BI__builtin_arm_mve_vmovltq_u16, 20955, 20924},
  { ARM::BI__builtin_arm_mve_vmovltq_u8, 20967, 20924},
  { ARM::BI__builtin_arm_mve_vmovltq_x_s16, 20988, 20978},
  { ARM::BI__builtin_arm_mve_vmovltq_x_s8, 21002, 20978},
  { ARM::BI__builtin_arm_mve_vmovltq_x_u16, 21015, 20978},
  { ARM::BI__builtin_arm_mve_vmovltq_x_u8, 21029, 20978},
  { ARM::BI__builtin_arm_mve_vmovnbq_m_s16, 21052, 21042},
  { ARM::BI__builtin_arm_mve_vmovnbq_m_s32, 21066, 21042},
  { ARM::BI__builtin_arm_mve_vmovnbq_m_u16, 21080, 21042},
  { ARM::BI__builtin_arm_mve_vmovnbq_m_u32, 21094, 21042},
  { ARM::BI__builtin_arm_mve_vmovnbq_s16, 21116, 21108},
  { ARM::BI__builtin_arm_mve_vmovnbq_s32, 21128, 21108},
  { ARM::BI__builtin_arm_mve_vmovnbq_u16, 21140, 21108},
  { ARM::BI__builtin_arm_mve_vmovnbq_u32, 21152, 21108},
  { ARM::BI__builtin_arm_mve_vmovntq_m_s16, 21174, 21164},
  { ARM::BI__builtin_arm_mve_vmovntq_m_s32, 21188, 21164},
  { ARM::BI__builtin_arm_mve_vmovntq_m_u16, 21202, 21164},
  { ARM::BI__builtin_arm_mve_vmovntq_m_u32, 21216, 21164},
  { ARM::BI__builtin_arm_mve_vmovntq_s16, 21238, 21230},
  { ARM::BI__builtin_arm_mve_vmovntq_s32, 21250, 21230},
  { ARM::BI__builtin_arm_mve_vmovntq_u16, 21262, 21230},
  { ARM::BI__builtin_arm_mve_vmovntq_u32, 21274, 21230},
  { ARM::BI__builtin_arm_mve_vmulhq_m_s16, 21295, 21286},
  { ARM::BI__builtin_arm_mve_vmulhq_m_s32, 21308, 21286},
  { ARM::BI__builtin_arm_mve_vmulhq_m_s8, 21321, 21286},
  { ARM::BI__builtin_arm_mve_vmulhq_m_u16, 21333, 21286},
  { ARM::BI__builtin_arm_mve_vmulhq_m_u32, 21346, 21286},
  { ARM::BI__builtin_arm_mve_vmulhq_m_u8, 21359, 21286},
  { ARM::BI__builtin_arm_mve_vmulhq_s16, 21378, 21371},
  { ARM::BI__builtin_arm_mve_vmulhq_s32, 21389, 21371},
  { ARM::BI__builtin_arm_mve_vmulhq_s8, 21400, 21371},
  { ARM::BI__builtin_arm_mve_vmulhq_u16, 21410, 21371},
  { ARM::BI__builtin_arm_mve_vmulhq_u32, 21421, 21371},
  { ARM::BI__builtin_arm_mve_vmulhq_u8, 21432, 21371},
  { ARM::BI__builtin_arm_mve_vmulhq_x_s16, 21451, 21442},
  { ARM::BI__builtin_arm_mve_vmulhq_x_s32, 21464, 21442},
  { ARM::BI__builtin_arm_mve_vmulhq_x_s8, 21477, 21442},
  { ARM::BI__builtin_arm_mve_vmulhq_x_u16, 21489, 21442},
  { ARM::BI__builtin_arm_mve_vmulhq_x_u32, 21502, 21442},
  { ARM::BI__builtin_arm_mve_vmulhq_x_u8, 21515, 21442},
  { ARM::BI__builtin_arm_mve_vmullbq_int_m_s16, 21541, 21527},
  { ARM::BI__builtin_arm_mve_vmullbq_int_m_s32, 21559, 21527},
  { ARM::BI__builtin_arm_mve_vmullbq_int_m_s8, 21577, 21527},
  { ARM::BI__builtin_arm_mve_vmullbq_int_m_u16, 21594, 21527},
  { ARM::BI__builtin_arm_mve_vmullbq_int_m_u32, 21612, 21527},
  { ARM::BI__builtin_arm_mve_vmullbq_int_m_u8, 21630, 21527},
  { ARM::BI__builtin_arm_mve_vmullbq_int_s16, 21659, 21647},
  { ARM::BI__builtin_arm_mve_vmullbq_int_s32, 21675, 21647},
  { ARM::BI__builtin_arm_mve_vmullbq_int_s8, 21691, 21647},
  { ARM::BI__builtin_arm_mve_vmullbq_int_u16, 21706, 21647},
  { ARM::BI__builtin_arm_mve_vmullbq_int_u32, 21722, 21647},
  { ARM::BI__builtin_arm_mve_vmullbq_int_u8, 21738, 21647},
  { ARM::BI__builtin_arm_mve_vmullbq_int_x_s16, 21767, 21753},
  { ARM::BI__builtin_arm_mve_vmullbq_int_x_s32, 21785, 21753},
  { ARM::BI__builtin_arm_mve_vmullbq_int_x_s8, 21803, 21753},
  { ARM::BI__builtin_arm_mve_vmullbq_int_x_u16, 21820, 21753},
  { ARM::BI__builtin_arm_mve_vmullbq_int_x_u32, 21838, 21753},
  { ARM::BI__builtin_arm_mve_vmullbq_int_x_u8, 21856, 21753},
  { ARM::BI__builtin_arm_mve_vmullbq_poly_m_p16, 21888, 21873},
  { ARM::BI__builtin_arm_mve_vmullbq_poly_m_p8, 21907, 21873},
  { ARM::BI__builtin_arm_mve_vmullbq_poly_p16, 21938, 21925},
  { ARM::BI__builtin_arm_mve_vmullbq_poly_p8, 21955, 21925},
  { ARM::BI__builtin_arm_mve_vmullbq_poly_x_p16, 21986, 21971},
  { ARM::BI__builtin_arm_mve_vmullbq_poly_x_p8, 22005, 21971},
  { ARM::BI__builtin_arm_mve_vmulltq_int_m_s16, 22037, 22023},
  { ARM::BI__builtin_arm_mve_vmulltq_int_m_s32, 22055, 22023},
  { ARM::BI__builtin_arm_mve_vmulltq_int_m_s8, 22073, 22023},
  { ARM::BI__builtin_arm_mve_vmulltq_int_m_u16, 22090, 22023},
  { ARM::BI__builtin_arm_mve_vmulltq_int_m_u32, 22108, 22023},
  { ARM::BI__builtin_arm_mve_vmulltq_int_m_u8, 22126, 22023},
  { ARM::BI__builtin_arm_mve_vmulltq_int_s16, 22155, 22143},
  { ARM::BI__builtin_arm_mve_vmulltq_int_s32, 22171, 22143},
  { ARM::BI__builtin_arm_mve_vmulltq_int_s8, 22187, 22143},
  { ARM::BI__builtin_arm_mve_vmulltq_int_u16, 22202, 22143},
  { ARM::BI__builtin_arm_mve_vmulltq_int_u32, 22218, 22143},
  { ARM::BI__builtin_arm_mve_vmulltq_int_u8, 22234, 22143},
  { ARM::BI__builtin_arm_mve_vmulltq_int_x_s16, 22263, 22249},
  { ARM::BI__builtin_arm_mve_vmulltq_int_x_s32, 22281, 22249},
  { ARM::BI__builtin_arm_mve_vmulltq_int_x_s8, 22299, 22249},
  { ARM::BI__builtin_arm_mve_vmulltq_int_x_u16, 22316, 22249},
  { ARM::BI__builtin_arm_mve_vmulltq_int_x_u32, 22334, 22249},
  { ARM::BI__builtin_arm_mve_vmulltq_int_x_u8, 22352, 22249},
  { ARM::BI__builtin_arm_mve_vmulltq_poly_m_p16, 22384, 22369},
  { ARM::BI__builtin_arm_mve_vmulltq_poly_m_p8, 22403, 22369},
  { ARM::BI__builtin_arm_mve_vmulltq_poly_p16, 22434, 22421},
  { ARM::BI__builtin_arm_mve_vmulltq_poly_p8, 22451, 22421},
  { ARM::BI__builtin_arm_mve_vmulltq_poly_x_p16, 22482, 22467},
  { ARM::BI__builtin_arm_mve_vmulltq_poly_x_p8, 22501, 22467},
  { ARM::BI__builtin_arm_mve_vmulq_f16, 22525, 22519},
  { ARM::BI__builtin_arm_mve_vmulq_f32, 22535, 22519},
  { ARM::BI__builtin_arm_mve_vmulq_m_f16, 22553, 22545},
  { ARM::BI__builtin_arm_mve_vmulq_m_f32, 22565, 22545},
  { ARM::BI__builtin_arm_mve_vmulq_m_n_f16, 22577, 22545},
  { ARM::BI__builtin_arm_mve_vmulq_m_n_f32, 22591, 22545},
  { ARM::BI__builtin_arm_mve_vmulq_m_n_s16, 22605, 22545},
  { ARM::BI__builtin_arm_mve_vmulq_m_n_s32, 22619, 22545},
  { ARM::BI__builtin_arm_mve_vmulq_m_n_s8, 22633, 22545},
  { ARM::BI__builtin_arm_mve_vmulq_m_n_u16, 22646, 22545},
  { ARM::BI__builtin_arm_mve_vmulq_m_n_u32, 22660, 22545},
  { ARM::BI__builtin_arm_mve_vmulq_m_n_u8, 22674, 22545},
  { ARM::BI__builtin_arm_mve_vmulq_m_s16, 22687, 22545},
  { ARM::BI__builtin_arm_mve_vmulq_m_s32, 22699, 22545},
  { ARM::BI__builtin_arm_mve_vmulq_m_s8, 22711, 22545},
  { ARM::BI__builtin_arm_mve_vmulq_m_u16, 22722, 22545},
  { ARM::BI__builtin_arm_mve_vmulq_m_u32, 22734, 22545},
  { ARM::BI__builtin_arm_mve_vmulq_m_u8, 22746, 22545},
  { ARM::BI__builtin_arm_mve_vmulq_n_f16, 22757, 22519},
  { ARM::BI__builtin_arm_mve_vmulq_n_f32, 22769, 22519},
  { ARM::BI__builtin_arm_mve_vmulq_n_s16, 22781, 22519},
  { ARM::BI__builtin_arm_mve_vmulq_n_s32, 22793, 22519},
  { ARM::BI__builtin_arm_mve_vmulq_n_s8, 22805, 22519},
  { ARM::BI__builtin_arm_mve_vmulq_n_u16, 22816, 22519},
  { ARM::BI__builtin_arm_mve_vmulq_n_u32, 22828, 22519},
  { ARM::BI__builtin_arm_mve_vmulq_n_u8, 22840, 22519},
  { ARM::BI__builtin_arm_mve_vmulq_s16, 22851, 22519},
  { ARM::BI__builtin_arm_mve_vmulq_s32, 22861, 22519},
  { ARM::BI__builtin_arm_mve_vmulq_s8, 22871, 22519},
  { ARM::BI__builtin_arm_mve_vmulq_u16, 22880, 22519},
  { ARM::BI__builtin_arm_mve_vmulq_u32, 22890, 22519},
  { ARM::BI__builtin_arm_mve_vmulq_u8, 22900, 22519},
  { ARM::BI__builtin_arm_mve_vmulq_x_f16, 22917, 22909},
  { ARM::BI__builtin_arm_mve_vmulq_x_f32, 22929, 22909},
  { ARM::BI__builtin_arm_mve_vmulq_x_n_f16, 22941, 22909},
  { ARM::BI__builtin_arm_mve_vmulq_x_n_f32, 22955, 22909},
  { ARM::BI__builtin_arm_mve_vmulq_x_n_s16, 22969, 22909},
  { ARM::BI__builtin_arm_mve_vmulq_x_n_s32, 22983, 22909},
  { ARM::BI__builtin_arm_mve_vmulq_x_n_s8, 22997, 22909},
  { ARM::BI__builtin_arm_mve_vmulq_x_n_u16, 23010, 22909},
  { ARM::BI__builtin_arm_mve_vmulq_x_n_u32, 23024, 22909},
  { ARM::BI__builtin_arm_mve_vmulq_x_n_u8, 23038, 22909},
  { ARM::BI__builtin_arm_mve_vmulq_x_s16, 23051, 22909},
  { ARM::BI__builtin_arm_mve_vmulq_x_s32, 23063, 22909},
  { ARM::BI__builtin_arm_mve_vmulq_x_s8, 23075, 22909},
  { ARM::BI__builtin_arm_mve_vmulq_x_u16, 23086, 22909},
  { ARM::BI__builtin_arm_mve_vmulq_x_u32, 23098, 22909},
  { ARM::BI__builtin_arm_mve_vmulq_x_u8, 23110, 22909},
  { ARM::BI__builtin_arm_mve_vmvnq_m_n_s16, 23129, 23121},
  { ARM::BI__builtin_arm_mve_vmvnq_m_n_s32, 23143, 23121},
  { ARM::BI__builtin_arm_mve_vmvnq_m_n_u16, 23157, 23121},
  { ARM::BI__builtin_arm_mve_vmvnq_m_n_u32, 23171, 23121},
  { ARM::BI__builtin_arm_mve_vmvnq_m_s16, 23185, 23121},
  { ARM::BI__builtin_arm_mve_vmvnq_m_s32, 23197, 23121},
  { ARM::BI__builtin_arm_mve_vmvnq_m_s8, 23209, 23121},
  { ARM::BI__builtin_arm_mve_vmvnq_m_u16, 23220, 23121},
  { ARM::BI__builtin_arm_mve_vmvnq_m_u32, 23232, 23121},
  { ARM::BI__builtin_arm_mve_vmvnq_m_u8, 23244, 23121},
  { ARM::BI__builtin_arm_mve_vmvnq_n_s16, 23255, -1},
  { ARM::BI__builtin_arm_mve_vmvnq_n_s32, 23267, -1},
  { ARM::BI__builtin_arm_mve_vmvnq_n_u16, 23279, -1},
  { ARM::BI__builtin_arm_mve_vmvnq_n_u32, 23291, -1},
  { ARM::BI__builtin_arm_mve_vmvnq_s16, 23309, 23303},
  { ARM::BI__builtin_arm_mve_vmvnq_s32, 23319, 23303},
  { ARM::BI__builtin_arm_mve_vmvnq_s8, 23329, 23303},
  { ARM::BI__builtin_arm_mve_vmvnq_u16, 23338, 23303},
  { ARM::BI__builtin_arm_mve_vmvnq_u32, 23348, 23303},
  { ARM::BI__builtin_arm_mve_vmvnq_u8, 23358, 23303},
  { ARM::BI__builtin_arm_mve_vmvnq_x_n_s16, 23367, -1},
  { ARM::BI__builtin_arm_mve_vmvnq_x_n_s32, 23381, -1},
  { ARM::BI__builtin_arm_mve_vmvnq_x_n_u16, 23395, -1},
  { ARM::BI__builtin_arm_mve_vmvnq_x_n_u32, 23409, -1},
  { ARM::BI__builtin_arm_mve_vmvnq_x_s16, 23431, 23423},
  { ARM::BI__builtin_arm_mve_vmvnq_x_s32, 23443, 23423},
  { ARM::BI__builtin_arm_mve_vmvnq_x_s8, 23455, 23423},
  { ARM::BI__builtin_arm_mve_vmvnq_x_u16, 23466, 23423},
  { ARM::BI__builtin_arm_mve_vmvnq_x_u32, 23478, 23423},
  { ARM::BI__builtin_arm_mve_vmvnq_x_u8, 23490, 23423},
  { ARM::BI__builtin_arm_mve_vnegq_f16, 23507, 23501},
  { ARM::BI__builtin_arm_mve_vnegq_f32, 23517, 23501},
  { ARM::BI__builtin_arm_mve_vnegq_m_f16, 23535, 23527},
  { ARM::BI__builtin_arm_mve_vnegq_m_f32, 23547, 23527},
  { ARM::BI__builtin_arm_mve_vnegq_m_s16, 23559, 23527},
  { ARM::BI__builtin_arm_mve_vnegq_m_s32, 23571, 23527},
  { ARM::BI__builtin_arm_mve_vnegq_m_s8, 23583, 23527},
  { ARM::BI__builtin_arm_mve_vnegq_s16, 23594, 23501},
  { ARM::BI__builtin_arm_mve_vnegq_s32, 23604, 23501},
  { ARM::BI__builtin_arm_mve_vnegq_s8, 23614, 23501},
  { ARM::BI__builtin_arm_mve_vnegq_x_f16, 23631, 23623},
  { ARM::BI__builtin_arm_mve_vnegq_x_f32, 23643, 23623},
  { ARM::BI__builtin_arm_mve_vnegq_x_s16, 23655, 23623},
  { ARM::BI__builtin_arm_mve_vnegq_x_s32, 23667, 23623},
  { ARM::BI__builtin_arm_mve_vnegq_x_s8, 23679, 23623},
  { ARM::BI__builtin_arm_mve_vornq_f16, 23696, 23690},
  { ARM::BI__builtin_arm_mve_vornq_f32, 23706, 23690},
  { ARM::BI__builtin_arm_mve_vornq_m_f16, 23724, 23716},
  { ARM::BI__builtin_arm_mve_vornq_m_f32, 23736, 23716},
  { ARM::BI__builtin_arm_mve_vornq_m_s16, 23748, 23716},
  { ARM::BI__builtin_arm_mve_vornq_m_s32, 23760, 23716},
  { ARM::BI__builtin_arm_mve_vornq_m_s8, 23772, 23716},
  { ARM::BI__builtin_arm_mve_vornq_m_u16, 23783, 23716},
  { ARM::BI__builtin_arm_mve_vornq_m_u32, 23795, 23716},
  { ARM::BI__builtin_arm_mve_vornq_m_u8, 23807, 23716},
  { ARM::BI__builtin_arm_mve_vornq_s16, 23818, 23690},
  { ARM::BI__builtin_arm_mve_vornq_s32, 23828, 23690},
  { ARM::BI__builtin_arm_mve_vornq_s8, 23838, 23690},
  { ARM::BI__builtin_arm_mve_vornq_u16, 23847, 23690},
  { ARM::BI__builtin_arm_mve_vornq_u32, 23857, 23690},
  { ARM::BI__builtin_arm_mve_vornq_u8, 23867, 23690},
  { ARM::BI__builtin_arm_mve_vornq_x_f16, 23884, 23876},
  { ARM::BI__builtin_arm_mve_vornq_x_f32, 23896, 23876},
  { ARM::BI__builtin_arm_mve_vornq_x_s16, 23908, 23876},
  { ARM::BI__builtin_arm_mve_vornq_x_s32, 23920, 23876},
  { ARM::BI__builtin_arm_mve_vornq_x_s8, 23932, 23876},
  { ARM::BI__builtin_arm_mve_vornq_x_u16, 23943, 23876},
  { ARM::BI__builtin_arm_mve_vornq_x_u32, 23955, 23876},
  { ARM::BI__builtin_arm_mve_vornq_x_u8, 23967, 23876},
  { ARM::BI__builtin_arm_mve_vorrq_f16, 23984, 23978},
  { ARM::BI__builtin_arm_mve_vorrq_f32, 23994, 23978},
  { ARM::BI__builtin_arm_mve_vorrq_m_f16, 24012, 24004},
  { ARM::BI__builtin_arm_mve_vorrq_m_f32, 24024, 24004},
  { ARM::BI__builtin_arm_mve_vorrq_m_n_s16, 24046, 24036},
  { ARM::BI__builtin_arm_mve_vorrq_m_n_s32, 24060, 24036},
  { ARM::BI__builtin_arm_mve_vorrq_m_n_u16, 24074, 24036},
  { ARM::BI__builtin_arm_mve_vorrq_m_n_u32, 24088, 24036},
  { ARM::BI__builtin_arm_mve_vorrq_m_s16, 24102, 24004},
  { ARM::BI__builtin_arm_mve_vorrq_m_s32, 24114, 24004},
  { ARM::BI__builtin_arm_mve_vorrq_m_s8, 24126, 24004},
  { ARM::BI__builtin_arm_mve_vorrq_m_u16, 24137, 24004},
  { ARM::BI__builtin_arm_mve_vorrq_m_u32, 24149, 24004},
  { ARM::BI__builtin_arm_mve_vorrq_m_u8, 24161, 24004},
  { ARM::BI__builtin_arm_mve_vorrq_n_s16, 24172, 23978},
  { ARM::BI__builtin_arm_mve_vorrq_n_s32, 24184, 23978},
  { ARM::BI__builtin_arm_mve_vorrq_n_u16, 24196, 23978},
  { ARM::BI__builtin_arm_mve_vorrq_n_u32, 24208, 23978},
  { ARM::BI__builtin_arm_mve_vorrq_s16, 24220, 23978},
  { ARM::BI__builtin_arm_mve_vorrq_s32, 24230, 23978},
  { ARM::BI__builtin_arm_mve_vorrq_s8, 24240, 23978},
  { ARM::BI__builtin_arm_mve_vorrq_u16, 24249, 23978},
  { ARM::BI__builtin_arm_mve_vorrq_u32, 24259, 23978},
  { ARM::BI__builtin_arm_mve_vorrq_u8, 24269, 23978},
  { ARM::BI__builtin_arm_mve_vorrq_x_f16, 24286, 24278},
  { ARM::BI__builtin_arm_mve_vorrq_x_f32, 24298, 24278},
  { ARM::BI__builtin_arm_mve_vorrq_x_s16, 24310, 24278},
  { ARM::BI__builtin_arm_mve_vorrq_x_s32, 24322, 24278},
  { ARM::BI__builtin_arm_mve_vorrq_x_s8, 24334, 24278},
  { ARM::BI__builtin_arm_mve_vorrq_x_u16, 24345, 24278},
  { ARM::BI__builtin_arm_mve_vorrq_x_u32, 24357, 24278},
  { ARM::BI__builtin_arm_mve_vorrq_x_u8, 24369, 24278},
  { ARM::BI__builtin_arm_mve_vpnot, 24380, -1},
  { ARM::BI__builtin_arm_mve_vpselq_f16, 24393, 24386},
  { ARM::BI__builtin_arm_mve_vpselq_f32, 24404, 24386},
  { ARM::BI__builtin_arm_mve_vpselq_s16, 24415, 24386},
  { ARM::BI__builtin_arm_mve_vpselq_s32, 24426, 24386},
  { ARM::BI__builtin_arm_mve_vpselq_s64, 24437, 24386},
  { ARM::BI__builtin_arm_mve_vpselq_s8, 24448, 24386},
  { ARM::BI__builtin_arm_mve_vpselq_u16, 24458, 24386},
  { ARM::BI__builtin_arm_mve_vpselq_u32, 24469, 24386},
  { ARM::BI__builtin_arm_mve_vpselq_u64, 24480, 24386},
  { ARM::BI__builtin_arm_mve_vpselq_u8, 24491, 24386},
  { ARM::BI__builtin_arm_mve_vqabsq_m_s16, 24510, 24501},
  { ARM::BI__builtin_arm_mve_vqabsq_m_s32, 24523, 24501},
  { ARM::BI__builtin_arm_mve_vqabsq_m_s8, 24536, 24501},
  { ARM::BI__builtin_arm_mve_vqabsq_s16, 24555, 24548},
  { ARM::BI__builtin_arm_mve_vqabsq_s32, 24566, 24548},
  { ARM::BI__builtin_arm_mve_vqabsq_s8, 24577, 24548},
  { ARM::BI__builtin_arm_mve_vqaddq_m_n_s16, 24596, 24587},
  { ARM::BI__builtin_arm_mve_vqaddq_m_n_s32, 24611, 24587},
  { ARM::BI__builtin_arm_mve_vqaddq_m_n_s8, 24626, 24587},
  { ARM::BI__builtin_arm_mve_vqaddq_m_n_u16, 24640, 24587},
  { ARM::BI__builtin_arm_mve_vqaddq_m_n_u32, 24655, 24587},
  { ARM::BI__builtin_arm_mve_vqaddq_m_n_u8, 24670, 24587},
  { ARM::BI__builtin_arm_mve_vqaddq_m_s16, 24684, 24587},
  { ARM::BI__builtin_arm_mve_vqaddq_m_s32, 24697, 24587},
  { ARM::BI__builtin_arm_mve_vqaddq_m_s8, 24710, 24587},
  { ARM::BI__builtin_arm_mve_vqaddq_m_u16, 24722, 24587},
  { ARM::BI__builtin_arm_mve_vqaddq_m_u32, 24735, 24587},
  { ARM::BI__builtin_arm_mve_vqaddq_m_u8, 24748, 24587},
  { ARM::BI__builtin_arm_mve_vqaddq_n_s16, 24767, 24760},
  { ARM::BI__builtin_arm_mve_vqaddq_n_s32, 24780, 24760},
  { ARM::BI__builtin_arm_mve_vqaddq_n_s8, 24793, 24760},
  { ARM::BI__builtin_arm_mve_vqaddq_n_u16, 24805, 24760},
  { ARM::BI__builtin_arm_mve_vqaddq_n_u32, 24818, 24760},
  { ARM::BI__builtin_arm_mve_vqaddq_n_u8, 24831, 24760},
  { ARM::BI__builtin_arm_mve_vqaddq_s16, 24843, 24760},
  { ARM::BI__builtin_arm_mve_vqaddq_s32, 24854, 24760},
  { ARM::BI__builtin_arm_mve_vqaddq_s8, 24865, 24760},
  { ARM::BI__builtin_arm_mve_vqaddq_u16, 24875, 24760},
  { ARM::BI__builtin_arm_mve_vqaddq_u32, 24886, 24760},
  { ARM::BI__builtin_arm_mve_vqaddq_u8, 24897, 24760},
  { ARM::BI__builtin_arm_mve_vqdmladhq_m_s16, 24919, 24907},
  { ARM::BI__builtin_arm_mve_vqdmladhq_m_s32, 24935, 24907},
  { ARM::BI__builtin_arm_mve_vqdmladhq_m_s8, 24951, 24907},
  { ARM::BI__builtin_arm_mve_vqdmladhq_s16, 24976, 24966},
  { ARM::BI__builtin_arm_mve_vqdmladhq_s32, 24990, 24966},
  { ARM::BI__builtin_arm_mve_vqdmladhq_s8, 25004, 24966},
  { ARM::BI__builtin_arm_mve_vqdmladhxq_m_s16, 25030, 25017},
  { ARM::BI__builtin_arm_mve_vqdmladhxq_m_s32, 25047, 25017},
  { ARM::BI__builtin_arm_mve_vqdmladhxq_m_s8, 25064, 25017},
  { ARM::BI__builtin_arm_mve_vqdmladhxq_s16, 25091, 25080},
  { ARM::BI__builtin_arm_mve_vqdmladhxq_s32, 25106, 25080},
  { ARM::BI__builtin_arm_mve_vqdmladhxq_s8, 25121, 25080},
  { ARM::BI__builtin_arm_mve_vqdmlahq_m_n_s16, 25146, 25135},
  { ARM::BI__builtin_arm_mve_vqdmlahq_m_n_s32, 25163, 25135},
  { ARM::BI__builtin_arm_mve_vqdmlahq_m_n_s8, 25180, 25135},
  { ARM::BI__builtin_arm_mve_vqdmlahq_n_s16, 25205, 25196},
  { ARM::BI__builtin_arm_mve_vqdmlahq_n_s32, 25220, 25196},
  { ARM::BI__builtin_arm_mve_vqdmlahq_n_s8, 25235, 25196},
  { ARM::BI__builtin_arm_mve_vqdmlashq_m_n_s16, 25261, 25249},
  { ARM::BI__builtin_arm_mve_vqdmlashq_m_n_s32, 25279, 25249},
  { ARM::BI__builtin_arm_mve_vqdmlashq_m_n_s8, 25297, 25249},
  { ARM::BI__builtin_arm_mve_vqdmlashq_n_s16, 25324, 25314},
  { ARM::BI__builtin_arm_mve_vqdmlashq_n_s32, 25340, 25314},
  { ARM::BI__builtin_arm_mve_vqdmlashq_n_s8, 25356, 25314},
  { ARM::BI__builtin_arm_mve_vqdmlsdhq_m_s16, 25383, 25371},
  { ARM::BI__builtin_arm_mve_vqdmlsdhq_m_s32, 25399, 25371},
  { ARM::BI__builtin_arm_mve_vqdmlsdhq_m_s8, 25415, 25371},
  { ARM::BI__builtin_arm_mve_vqdmlsdhq_s16, 25440, 25430},
  { ARM::BI__builtin_arm_mve_vqdmlsdhq_s32, 25454, 25430},
  { ARM::BI__builtin_arm_mve_vqdmlsdhq_s8, 25468, 25430},
  { ARM::BI__builtin_arm_mve_vqdmlsdhxq_m_s16, 25494, 25481},
  { ARM::BI__builtin_arm_mve_vqdmlsdhxq_m_s32, 25511, 25481},
  { ARM::BI__builtin_arm_mve_vqdmlsdhxq_m_s8, 25528, 25481},
  { ARM::BI__builtin_arm_mve_vqdmlsdhxq_s16, 25555, 25544},
  { ARM::BI__builtin_arm_mve_vqdmlsdhxq_s32, 25570, 25544},
  { ARM::BI__builtin_arm_mve_vqdmlsdhxq_s8, 25585, 25544},
  { ARM::BI__builtin_arm_mve_vqdmulhq_m_n_s16, 25610, 25599},
  { ARM::BI__builtin_arm_mve_vqdmulhq_m_n_s32, 25627, 25599},
  { ARM::BI__builtin_arm_mve_vqdmulhq_m_n_s8, 25644, 25599},
  { ARM::BI__builtin_arm_mve_vqdmulhq_m_s16, 25660, 25599},
  { ARM::BI__builtin_arm_mve_vqdmulhq_m_s32, 25675, 25599},
  { ARM::BI__builtin_arm_mve_vqdmulhq_m_s8, 25690, 25599},
  { ARM::BI__builtin_arm_mve_vqdmulhq_n_s16, 25713, 25704},
  { ARM::BI__builtin_arm_mve_vqdmulhq_n_s32, 25728, 25704},
  { ARM::BI__builtin_arm_mve_vqdmulhq_n_s8, 25743, 25704},
  { ARM::BI__builtin_arm_mve_vqdmulhq_s16, 25757, 25704},
  { ARM::BI__builtin_arm_mve_vqdmulhq_s32, 25770, 25704},
  { ARM::BI__builtin_arm_mve_vqdmulhq_s8, 25783, 25704},
  { ARM::BI__builtin_arm_mve_vqdmullbq_m_n_s16, 25807, 25795},
  { ARM::BI__builtin_arm_mve_vqdmullbq_m_n_s32, 25825, 25795},
  { ARM::BI__builtin_arm_mve_vqdmullbq_m_s16, 25843, 25795},
  { ARM::BI__builtin_arm_mve_vqdmullbq_m_s32, 25859, 25795},
  { ARM::BI__builtin_arm_mve_vqdmullbq_n_s16, 25885, 25875},
  { ARM::BI__builtin_arm_mve_vqdmullbq_n_s32, 25901, 25875},
  { ARM::BI__builtin_arm_mve_vqdmullbq_s16, 25917, 25875},
  { ARM::BI__builtin_arm_mve_vqdmullbq_s32, 25931, 25875},
  { ARM::BI__builtin_arm_mve_vqdmulltq_m_n_s16, 25957, 25945},
  { ARM::BI__builtin_arm_mve_vqdmulltq_m_n_s32, 25975, 25945},
  { ARM::BI__builtin_arm_mve_vqdmulltq_m_s16, 25993, 25945},
  { ARM::BI__builtin_arm_mve_vqdmulltq_m_s32, 26009, 25945},
  { ARM::BI__builtin_arm_mve_vqdmulltq_n_s16, 26035, 26025},
  { ARM::BI__builtin_arm_mve_vqdmulltq_n_s32, 26051, 26025},
  { ARM::BI__builtin_arm_mve_vqdmulltq_s16, 26067, 26025},
  { ARM::BI__builtin_arm_mve_vqdmulltq_s32, 26081, 26025},
  { ARM::BI__builtin_arm_mve_vqmovnbq_m_s16, 26106, 26095},
  { ARM::BI__builtin_arm_mve_vqmovnbq_m_s32, 26121, 26095},
  { ARM::BI__builtin_arm_mve_vqmovnbq_m_u16, 26136, 26095},
  { ARM::BI__builtin_arm_mve_vqmovnbq_m_u32, 26151, 26095},
  { ARM::BI__builtin_arm_mve_vqmovnbq_s16, 26175, 26166},
  { ARM::BI__builtin_arm_mve_vqmovnbq_s32, 26188, 26166},
  { ARM::BI__builtin_arm_mve_vqmovnbq_u16, 26201, 26166},
  { ARM::BI__builtin_arm_mve_vqmovnbq_u32, 26214, 26166},
  { ARM::BI__builtin_arm_mve_vqmovntq_m_s16, 26238, 26227},
  { ARM::BI__builtin_arm_mve_vqmovntq_m_s32, 26253, 26227},
  { ARM::BI__builtin_arm_mve_vqmovntq_m_u16, 26268, 26227},
  { ARM::BI__builtin_arm_mve_vqmovntq_m_u32, 26283, 26227},
  { ARM::BI__builtin_arm_mve_vqmovntq_s16, 26307, 26298},
  { ARM::BI__builtin_arm_mve_vqmovntq_s32, 26320, 26298},
  { ARM::BI__builtin_arm_mve_vqmovntq_u16, 26333, 26298},
  { ARM::BI__builtin_arm_mve_vqmovntq_u32, 26346, 26298},
  { ARM::BI__builtin_arm_mve_vqmovunbq_m_s16, 26371, 26359},
  { ARM::BI__builtin_arm_mve_vqmovunbq_m_s32, 26387, 26359},
  { ARM::BI__builtin_arm_mve_vqmovunbq_s16, 26413, 26403},
  { ARM::BI__builtin_arm_mve_vqmovunbq_s32, 26427, 26403},
  { ARM::BI__builtin_arm_mve_vqmovuntq_m_s16, 26453, 26441},
  { ARM::BI__builtin_arm_mve_vqmovuntq_m_s32, 26469, 26441},
  { ARM::BI__builtin_arm_mve_vqmovuntq_s16, 26495, 26485},
  { ARM::BI__builtin_arm_mve_vqmovuntq_s32, 26509, 26485},
  { ARM::BI__builtin_arm_mve_vqnegq_m_s16, 26532, 26523},
  { ARM::BI__builtin_arm_mve_vqnegq_m_s32, 26545, 26523},
  { ARM::BI__builtin_arm_mve_vqnegq_m_s8, 26558, 26523},
  { ARM::BI__builtin_arm_mve_vqnegq_s16, 26577, 26570},
  { ARM::BI__builtin_arm_mve_vqnegq_s32, 26588, 26570},
  { ARM::BI__builtin_arm_mve_vqnegq_s8, 26599, 26570},
  { ARM::BI__builtin_arm_mve_vqrdmladhq_m_s16, 26622, 26609},
  { ARM::BI__builtin_arm_mve_vqrdmladhq_m_s32, 26639, 26609},
  { ARM::BI__builtin_arm_mve_vqrdmladhq_m_s8, 26656, 26609},
  { ARM::BI__builtin_arm_mve_vqrdmladhq_s16, 26683, 26672},
  { ARM::BI__builtin_arm_mve_vqrdmladhq_s32, 26698, 26672},
  { ARM::BI__builtin_arm_mve_vqrdmladhq_s8, 26713, 26672},
  { ARM::BI__builtin_arm_mve_vqrdmladhxq_m_s16, 26741, 26727},
  { ARM::BI__builtin_arm_mve_vqrdmladhxq_m_s32, 26759, 26727},
  { ARM::BI__builtin_arm_mve_vqrdmladhxq_m_s8, 26777, 26727},
  { ARM::BI__builtin_arm_mve_vqrdmladhxq_s16, 26806, 26794},
  { ARM::BI__builtin_arm_mve_vqrdmladhxq_s32, 26822, 26794},
  { ARM::BI__builtin_arm_mve_vqrdmladhxq_s8, 26838, 26794},
  { ARM::BI__builtin_arm_mve_vqrdmlahq_m_n_s16, 26865, 26853},
  { ARM::BI__builtin_arm_mve_vqrdmlahq_m_n_s32, 26883, 26853},
  { ARM::BI__builtin_arm_mve_vqrdmlahq_m_n_s8, 26901, 26853},
  { ARM::BI__builtin_arm_mve_vqrdmlahq_n_s16, 26928, 26918},
  { ARM::BI__builtin_arm_mve_vqrdmlahq_n_s32, 26944, 26918},
  { ARM::BI__builtin_arm_mve_vqrdmlahq_n_s8, 26960, 26918},
  { ARM::BI__builtin_arm_mve_vqrdmlashq_m_n_s16, 26988, 26975},
  { ARM::BI__builtin_arm_mve_vqrdmlashq_m_n_s32, 27007, 26975},
  { ARM::BI__builtin_arm_mve_vqrdmlashq_m_n_s8, 27026, 26975},
  { ARM::BI__builtin_arm_mve_vqrdmlashq_n_s16, 27055, 27044},
  { ARM::BI__builtin_arm_mve_vqrdmlashq_n_s32, 27072, 27044},
  { ARM::BI__builtin_arm_mve_vqrdmlashq_n_s8, 27089, 27044},
  { ARM::BI__builtin_arm_mve_vqrdmlsdhq_m_s16, 27118, 27105},
  { ARM::BI__builtin_arm_mve_vqrdmlsdhq_m_s32, 27135, 27105},
  { ARM::BI__builtin_arm_mve_vqrdmlsdhq_m_s8, 27152, 27105},
  { ARM::BI__builtin_arm_mve_vqrdmlsdhq_s16, 27179, 27168},
  { ARM::BI__builtin_arm_mve_vqrdmlsdhq_s32, 27194, 27168},
  { ARM::BI__builtin_arm_mve_vqrdmlsdhq_s8, 27209, 27168},
  { ARM::BI__builtin_arm_mve_vqrdmlsdhxq_m_s16, 27237, 27223},
  { ARM::BI__builtin_arm_mve_vqrdmlsdhxq_m_s32, 27255, 27223},
  { ARM::BI__builtin_arm_mve_vqrdmlsdhxq_m_s8, 27273, 27223},
  { ARM::BI__builtin_arm_mve_vqrdmlsdhxq_s16, 27302, 27290},
  { ARM::BI__builtin_arm_mve_vqrdmlsdhxq_s32, 27318, 27290},
  { ARM::BI__builtin_arm_mve_vqrdmlsdhxq_s8, 27334, 27290},
  { ARM::BI__builtin_arm_mve_vqrdmulhq_m_n_s16, 27361, 27349},
  { ARM::BI__builtin_arm_mve_vqrdmulhq_m_n_s32, 27379, 27349},
  { ARM::BI__builtin_arm_mve_vqrdmulhq_m_n_s8, 27397, 27349},
  { ARM::BI__builtin_arm_mve_vqrdmulhq_m_s16, 27414, 27349},
  { ARM::BI__builtin_arm_mve_vqrdmulhq_m_s32, 27430, 27349},
  { ARM::BI__builtin_arm_mve_vqrdmulhq_m_s8, 27446, 27349},
  { ARM::BI__builtin_arm_mve_vqrdmulhq_n_s16, 27471, 27461},
  { ARM::BI__builtin_arm_mve_vqrdmulhq_n_s32, 27487, 27461},
  { ARM::BI__builtin_arm_mve_vqrdmulhq_n_s8, 27503, 27461},
  { ARM::BI__builtin_arm_mve_vqrdmulhq_s16, 27518, 27461},
  { ARM::BI__builtin_arm_mve_vqrdmulhq_s32, 27532, 27461},
  { ARM::BI__builtin_arm_mve_vqrdmulhq_s8, 27546, 27461},
  { ARM::BI__builtin_arm_mve_vqrshlq_m_n_s16, 27571, 27559},
  { ARM::BI__builtin_arm_mve_vqrshlq_m_n_s32, 27587, 27559},
  { ARM::BI__builtin_arm_mve_vqrshlq_m_n_s8, 27603, 27559},
  { ARM::BI__builtin_arm_mve_vqrshlq_m_n_u16, 27618, 27559},
  { ARM::BI__builtin_arm_mve_vqrshlq_m_n_u32, 27634, 27559},
  { ARM::BI__builtin_arm_mve_vqrshlq_m_n_u8, 27650, 27559},
  { ARM::BI__builtin_arm_mve_vqrshlq_m_s16, 27675, 27665},
  { ARM::BI__builtin_arm_mve_vqrshlq_m_s32, 27689, 27665},
  { ARM::BI__builtin_arm_mve_vqrshlq_m_s8, 27703, 27665},
  { ARM::BI__builtin_arm_mve_vqrshlq_m_u16, 27716, 27665},
  { ARM::BI__builtin_arm_mve_vqrshlq_m_u32, 27730, 27665},
  { ARM::BI__builtin_arm_mve_vqrshlq_m_u8, 27744, 27665},
  { ARM::BI__builtin_arm_mve_vqrshlq_n_s16, 27765, 27757},
  { ARM::BI__builtin_arm_mve_vqrshlq_n_s32, 27779, 27757},
  { ARM::BI__builtin_arm_mve_vqrshlq_n_s8, 27793, 27757},
  { ARM::BI__builtin_arm_mve_vqrshlq_n_u16, 27806, 27757},
  { ARM::BI__builtin_arm_mve_vqrshlq_n_u32, 27820, 27757},
  { ARM::BI__builtin_arm_mve_vqrshlq_n_u8, 27834, 27757},
  { ARM::BI__builtin_arm_mve_vqrshlq_s16, 27847, 27757},
  { ARM::BI__builtin_arm_mve_vqrshlq_s32, 27859, 27757},
  { ARM::BI__builtin_arm_mve_vqrshlq_s8, 27871, 27757},
  { ARM::BI__builtin_arm_mve_vqrshlq_u16, 27882, 27757},
  { ARM::BI__builtin_arm_mve_vqrshlq_u32, 27894, 27757},
  { ARM::BI__builtin_arm_mve_vqrshlq_u8, 27906, 27757},
  { ARM::BI__builtin_arm_mve_vqrshrnbq_m_n_s16, 27929, 27917},
  { ARM::BI__builtin_arm_mve_vqrshrnbq_m_n_s32, 27947, 27917},
  { ARM::BI__builtin_arm_mve_vqrshrnbq_m_n_u16, 27965, 27917},
  { ARM::BI__builtin_arm_mve_vqrshrnbq_m_n_u32, 27983, 27917},
  { ARM::BI__builtin_arm_mve_vqrshrnbq_n_s16, 28011, 28001},
  { ARM::BI__builtin_arm_mve_vqrshrnbq_n_s32, 28027, 28001},
  { ARM::BI__builtin_arm_mve_vqrshrnbq_n_u16, 28043, 28001},
  { ARM::BI__builtin_arm_mve_vqrshrnbq_n_u32, 28059, 28001},
  { ARM::BI__builtin_arm_mve_vqrshrntq_m_n_s16, 28087, 28075},
  { ARM::BI__builtin_arm_mve_vqrshrntq_m_n_s32, 28105, 28075},
  { ARM::BI__builtin_arm_mve_vqrshrntq_m_n_u16, 28123, 28075},
  { ARM::BI__builtin_arm_mve_vqrshrntq_m_n_u32, 28141, 28075},
  { ARM::BI__builtin_arm_mve_vqrshrntq_n_s16, 28169, 28159},
  { ARM::BI__builtin_arm_mve_vqrshrntq_n_s32, 28185, 28159},
  { ARM::BI__builtin_arm_mve_vqrshrntq_n_u16, 28201, 28159},
  { ARM::BI__builtin_arm_mve_vqrshrntq_n_u32, 28217, 28159},
  { ARM::BI__builtin_arm_mve_vqrshrunbq_m_n_s16, 28246, 28233},
  { ARM::BI__builtin_arm_mve_vqrshrunbq_m_n_s32, 28265, 28233},
  { ARM::BI__builtin_arm_mve_vqrshrunbq_n_s16, 28295, 28284},
  { ARM::BI__builtin_arm_mve_vqrshrunbq_n_s32, 28312, 28284},
  { ARM::BI__builtin_arm_mve_vqrshruntq_m_n_s16, 28342, 28329},
  { ARM::BI__builtin_arm_mve_vqrshruntq_m_n_s32, 28361, 28329},
  { ARM::BI__builtin_arm_mve_vqrshruntq_n_s16, 28391, 28380},
  { ARM::BI__builtin_arm_mve_vqrshruntq_n_s32, 28408, 28380},
  { ARM::BI__builtin_arm_mve_vqshlq_m_n_s16, 28436, 28425},
  { ARM::BI__builtin_arm_mve_vqshlq_m_n_s32, 28451, 28425},
  { ARM::BI__builtin_arm_mve_vqshlq_m_n_s8, 28466, 28425},
  { ARM::BI__builtin_arm_mve_vqshlq_m_n_u16, 28480, 28425},
  { ARM::BI__builtin_arm_mve_vqshlq_m_n_u32, 28495, 28425},
  { ARM::BI__builtin_arm_mve_vqshlq_m_n_u8, 28510, 28425},
  { ARM::BI__builtin_arm_mve_vqshlq_m_r_s16, 28535, 28524},
  { ARM::BI__builtin_arm_mve_vqshlq_m_r_s32, 28550, 28524},
  { ARM::BI__builtin_arm_mve_vqshlq_m_r_s8, 28565, 28524},
  { ARM::BI__builtin_arm_mve_vqshlq_m_r_u16, 28579, 28524},
  { ARM::BI__builtin_arm_mve_vqshlq_m_r_u32, 28594, 28524},
  { ARM::BI__builtin_arm_mve_vqshlq_m_r_u8, 28609, 28524},
  { ARM::BI__builtin_arm_mve_vqshlq_m_s16, 28632, 28623},
  { ARM::BI__builtin_arm_mve_vqshlq_m_s32, 28645, 28623},
  { ARM::BI__builtin_arm_mve_vqshlq_m_s8, 28658, 28623},
  { ARM::BI__builtin_arm_mve_vqshlq_m_u16, 28670, 28623},
  { ARM::BI__builtin_arm_mve_vqshlq_m_u32, 28683, 28623},
  { ARM::BI__builtin_arm_mve_vqshlq_m_u8, 28696, 28623},
  { ARM::BI__builtin_arm_mve_vqshlq_n_s16, 28717, 28708},
  { ARM::BI__builtin_arm_mve_vqshlq_n_s32, 28730, 28708},
  { ARM::BI__builtin_arm_mve_vqshlq_n_s8, 28743, 28708},
  { ARM::BI__builtin_arm_mve_vqshlq_n_u16, 28755, 28708},
  { ARM::BI__builtin_arm_mve_vqshlq_n_u32, 28768, 28708},
  { ARM::BI__builtin_arm_mve_vqshlq_n_u8, 28781, 28708},
  { ARM::BI__builtin_arm_mve_vqshlq_r_s16, 28802, 28793},
  { ARM::BI__builtin_arm_mve_vqshlq_r_s32, 28815, 28793},
  { ARM::BI__builtin_arm_mve_vqshlq_r_s8, 28828, 28793},
  { ARM::BI__builtin_arm_mve_vqshlq_r_u16, 28840, 28793},
  { ARM::BI__builtin_arm_mve_vqshlq_r_u32, 28853, 28793},
  { ARM::BI__builtin_arm_mve_vqshlq_r_u8, 28866, 28793},
  { ARM::BI__builtin_arm_mve_vqshlq_s16, 28885, 28878},
  { ARM::BI__builtin_arm_mve_vqshlq_s32, 28896, 28878},
  { ARM::BI__builtin_arm_mve_vqshlq_s8, 28907, 28878},
  { ARM::BI__builtin_arm_mve_vqshlq_u16, 28917, 28878},
  { ARM::BI__builtin_arm_mve_vqshlq_u32, 28928, 28878},
  { ARM::BI__builtin_arm_mve_vqshlq_u8, 28939, 28878},
  { ARM::BI__builtin_arm_mve_vqshluq_m_n_s16, 28959, 28949},
  { ARM::BI__builtin_arm_mve_vqshluq_m_n_s32, 28975, 28949},
  { ARM::BI__builtin_arm_mve_vqshluq_m_n_s8, 28991, 28949},
  { ARM::BI__builtin_arm_mve_vqshluq_n_s16, 29014, 29006},
  { ARM::BI__builtin_arm_mve_vqshluq_n_s32, 29028, 29006},
  { ARM::BI__builtin_arm_mve_vqshluq_n_s8, 29042, 29006},
  { ARM::BI__builtin_arm_mve_vqshrnbq_m_n_s16, 29066, 29055},
  { ARM::BI__builtin_arm_mve_vqshrnbq_m_n_s32, 29083, 29055},
  { ARM::BI__builtin_arm_mve_vqshrnbq_m_n_u16, 29100, 29055},
  { ARM::BI__builtin_arm_mve_vqshrnbq_m_n_u32, 29117, 29055},
  { ARM::BI__builtin_arm_mve_vqshrnbq_n_s16, 29143, 29134},
  { ARM::BI__builtin_arm_mve_vqshrnbq_n_s32, 29158, 29134},
  { ARM::BI__builtin_arm_mve_vqshrnbq_n_u16, 29173, 29134},
  { ARM::BI__builtin_arm_mve_vqshrnbq_n_u32, 29188, 29134},
  { ARM::BI__builtin_arm_mve_vqshrntq_m_n_s16, 29214, 29203},
  { ARM::BI__builtin_arm_mve_vqshrntq_m_n_s32, 29231, 29203},
  { ARM::BI__builtin_arm_mve_vqshrntq_m_n_u16, 29248, 29203},
  { ARM::BI__builtin_arm_mve_vqshrntq_m_n_u32, 29265, 29203},
  { ARM::BI__builtin_arm_mve_vqshrntq_n_s16, 29291, 29282},
  { ARM::BI__builtin_arm_mve_vqshrntq_n_s32, 29306, 29282},
  { ARM::BI__builtin_arm_mve_vqshrntq_n_u16, 29321, 29282},
  { ARM::BI__builtin_arm_mve_vqshrntq_n_u32, 29336, 29282},
  { ARM::BI__builtin_arm_mve_vqshrunbq_m_n_s16, 29363, 29351},
  { ARM::BI__builtin_arm_mve_vqshrunbq_m_n_s32, 29381, 29351},
  { ARM::BI__builtin_arm_mve_vqshrunbq_n_s16, 29409, 29399},
  { ARM::BI__builtin_arm_mve_vqshrunbq_n_s32, 29425, 29399},
  { ARM::BI__builtin_arm_mve_vqshruntq_m_n_s16, 29453, 29441},
  { ARM::BI__builtin_arm_mve_vqshruntq_m_n_s32, 29471, 29441},
  { ARM::BI__builtin_arm_mve_vqshruntq_n_s16, 29499, 29489},
  { ARM::BI__builtin_arm_mve_vqshruntq_n_s32, 29515, 29489},
  { ARM::BI__builtin_arm_mve_vqsubq_m_n_s16, 29540, 29531},
  { ARM::BI__builtin_arm_mve_vqsubq_m_n_s32, 29555, 29531},
  { ARM::BI__builtin_arm_mve_vqsubq_m_n_s8, 29570, 29531},
  { ARM::BI__builtin_arm_mve_vqsubq_m_n_u16, 29584, 29531},
  { ARM::BI__builtin_arm_mve_vqsubq_m_n_u32, 29599, 29531},
  { ARM::BI__builtin_arm_mve_vqsubq_m_n_u8, 29614, 29531},
  { ARM::BI__builtin_arm_mve_vqsubq_m_s16, 29628, 29531},
  { ARM::BI__builtin_arm_mve_vqsubq_m_s32, 29641, 29531},
  { ARM::BI__builtin_arm_mve_vqsubq_m_s8, 29654, 29531},
  { ARM::BI__builtin_arm_mve_vqsubq_m_u16, 29666, 29531},
  { ARM::BI__builtin_arm_mve_vqsubq_m_u32, 29679, 29531},
  { ARM::BI__builtin_arm_mve_vqsubq_m_u8, 29692, 29531},
  { ARM::BI__builtin_arm_mve_vqsubq_n_s16, 29711, 29704},
  { ARM::BI__builtin_arm_mve_vqsubq_n_s32, 29724, 29704},
  { ARM::BI__builtin_arm_mve_vqsubq_n_s8, 29737, 29704},
  { ARM::BI__builtin_arm_mve_vqsubq_n_u16, 29749, 29704},
  { ARM::BI__builtin_arm_mve_vqsubq_n_u32, 29762, 29704},
  { ARM::BI__builtin_arm_mve_vqsubq_n_u8, 29775, 29704},
  { ARM::BI__builtin_arm_mve_vqsubq_s16, 29787, 29704},
  { ARM::BI__builtin_arm_mve_vqsubq_s32, 29798, 29704},
  { ARM::BI__builtin_arm_mve_vqsubq_s8, 29809, 29704},
  { ARM::BI__builtin_arm_mve_vqsubq_u16, 29819, 29704},
  { ARM::BI__builtin_arm_mve_vqsubq_u32, 29830, 29704},
  { ARM::BI__builtin_arm_mve_vqsubq_u8, 29841, 29704},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f16_f32, 29869, 29851},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f16_s16, 29891, 29851},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f16_s32, 29913, 29851},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f16_s64, 29935, 29851},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f16_s8, 29957, 29851},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f16_u16, 29978, 29851},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f16_u32, 30000, 29851},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f16_u64, 30022, 29851},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f16_u8, 30044, 29851},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f32_f16, 30083, 30065},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f32_s16, 30105, 30065},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f32_s32, 30127, 30065},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f32_s64, 30149, 30065},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f32_s8, 30171, 30065},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f32_u16, 30192, 30065},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f32_u32, 30214, 30065},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f32_u64, 30236, 30065},
  { ARM::BI__builtin_arm_mve_vreinterpretq_f32_u8, 30258, 30065},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s16_f16, 30297, 30279},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s16_f32, 30319, 30279},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s16_s32, 30341, 30279},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s16_s64, 30363, 30279},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s16_s8, 30385, 30279},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s16_u16, 30406, 30279},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s16_u32, 30428, 30279},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s16_u64, 30450, 30279},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s16_u8, 30472, 30279},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s32_f16, 30511, 30493},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s32_f32, 30533, 30493},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s32_s16, 30555, 30493},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s32_s64, 30577, 30493},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s32_s8, 30599, 30493},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s32_u16, 30620, 30493},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s32_u32, 30642, 30493},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s32_u64, 30664, 30493},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s32_u8, 30686, 30493},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s64_f16, 30725, 30707},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s64_f32, 30747, 30707},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s64_s16, 30769, 30707},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s64_s32, 30791, 30707},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s64_s8, 30813, 30707},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s64_u16, 30834, 30707},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s64_u32, 30856, 30707},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s64_u64, 30878, 30707},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s64_u8, 30900, 30707},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s8_f16, 30938, 30921},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s8_f32, 30959, 30921},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s8_s16, 30980, 30921},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s8_s32, 31001, 30921},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s8_s64, 31022, 30921},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s8_u16, 31043, 30921},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s8_u32, 31064, 30921},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s8_u64, 31085, 30921},
  { ARM::BI__builtin_arm_mve_vreinterpretq_s8_u8, 31106, 30921},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u16_f16, 31144, 31126},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u16_f32, 31166, 31126},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u16_s16, 31188, 31126},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u16_s32, 31210, 31126},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u16_s64, 31232, 31126},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u16_s8, 31254, 31126},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u16_u32, 31275, 31126},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u16_u64, 31297, 31126},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u16_u8, 31319, 31126},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u32_f16, 31358, 31340},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u32_f32, 31380, 31340},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u32_s16, 31402, 31340},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u32_s32, 31424, 31340},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u32_s64, 31446, 31340},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u32_s8, 31468, 31340},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u32_u16, 31489, 31340},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u32_u64, 31511, 31340},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u32_u8, 31533, 31340},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u64_f16, 31572, 31554},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u64_f32, 31594, 31554},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u64_s16, 31616, 31554},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u64_s32, 31638, 31554},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u64_s64, 31660, 31554},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u64_s8, 31682, 31554},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u64_u16, 31703, 31554},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u64_u32, 31725, 31554},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u64_u8, 31747, 31554},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u8_f16, 31785, 31768},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u8_f32, 31806, 31768},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u8_s16, 31827, 31768},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u8_s32, 31848, 31768},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u8_s64, 31869, 31768},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u8_s8, 31890, 31768},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u8_u16, 31910, 31768},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u8_u32, 31931, 31768},
  { ARM::BI__builtin_arm_mve_vreinterpretq_u8_u64, 31952, 31768},
  { ARM::BI__builtin_arm_mve_vrev16q_m_s8, 31983, 31973},
  { ARM::BI__builtin_arm_mve_vrev16q_m_u8, 31996, 31973},
  { ARM::BI__builtin_arm_mve_vrev16q_s8, 32017, 32009},
  { ARM::BI__builtin_arm_mve_vrev16q_u8, 32028, 32009},
  { ARM::BI__builtin_arm_mve_vrev16q_x_s8, 32049, 32039},
  { ARM::BI__builtin_arm_mve_vrev16q_x_u8, 32062, 32039},
  { ARM::BI__builtin_arm_mve_vrev32q_f16, 32083, 32075},
  { ARM::BI__builtin_arm_mve_vrev32q_m_f16, 32105, 32095},
  { ARM::BI__builtin_arm_mve_vrev32q_m_s16, 32119, 32095},
  { ARM::BI__builtin_arm_mve_vrev32q_m_s8, 32133, 32095},
  { ARM::BI__builtin_arm_mve_vrev32q_m_u16, 32146, 32095},
  { ARM::BI__builtin_arm_mve_vrev32q_m_u8, 32160, 32095},
  { ARM::BI__builtin_arm_mve_vrev32q_s16, 32173, 32075},
  { ARM::BI__builtin_arm_mve_vrev32q_s8, 32185, 32075},
  { ARM::BI__builtin_arm_mve_vrev32q_u16, 32196, 32075},
  { ARM::BI__builtin_arm_mve_vrev32q_u8, 32208, 32075},
  { ARM::BI__builtin_arm_mve_vrev32q_x_f16, 32229, 32219},
  { ARM::BI__builtin_arm_mve_vrev32q_x_s16, 32243, 32219},
  { ARM::BI__builtin_arm_mve_vrev32q_x_s8, 32257, 32219},
  { ARM::BI__builtin_arm_mve_vrev32q_x_u16, 32270, 32219},
  { ARM::BI__builtin_arm_mve_vrev32q_x_u8, 32284, 32219},
  { ARM::BI__builtin_arm_mve_vrev64q_f16, 32305, 32297},
  { ARM::BI__builtin_arm_mve_vrev64q_f32, 32317, 32297},
  { ARM::BI__builtin_arm_mve_vrev64q_m_f16, 32339, 32329},
  { ARM::BI__builtin_arm_mve_vrev64q_m_f32, 32353, 32329},
  { ARM::BI__builtin_arm_mve_vrev64q_m_s16, 32367, 32329},
  { ARM::BI__builtin_arm_mve_vrev64q_m_s32, 32381, 32329},
  { ARM::BI__builtin_arm_mve_vrev64q_m_s8, 32395, 32329},
  { ARM::BI__builtin_arm_mve_vrev64q_m_u16, 32408, 32329},
  { ARM::BI__builtin_arm_mve_vrev64q_m_u32, 32422, 32329},
  { ARM::BI__builtin_arm_mve_vrev64q_m_u8, 32436, 32329},
  { ARM::BI__builtin_arm_mve_vrev64q_s16, 32449, 32297},
  { ARM::BI__builtin_arm_mve_vrev64q_s32, 32461, 32297},
  { ARM::BI__builtin_arm_mve_vrev64q_s8, 32473, 32297},
  { ARM::BI__builtin_arm_mve_vrev64q_u16, 32484, 32297},
  { ARM::BI__builtin_arm_mve_vrev64q_u32, 32496, 32297},
  { ARM::BI__builtin_arm_mve_vrev64q_u8, 32508, 32297},
  { ARM::BI__builtin_arm_mve_vrev64q_x_f16, 32529, 32519},
  { ARM::BI__builtin_arm_mve_vrev64q_x_f32, 32543, 32519},
  { ARM::BI__builtin_arm_mve_vrev64q_x_s16, 32557, 32519},
  { ARM::BI__builtin_arm_mve_vrev64q_x_s32, 32571, 32519},
  { ARM::BI__builtin_arm_mve_vrev64q_x_s8, 32585, 32519},
  { ARM::BI__builtin_arm_mve_vrev64q_x_u16, 32598, 32519},
  { ARM::BI__builtin_arm_mve_vrev64q_x_u32, 32612, 32519},
  { ARM::BI__builtin_arm_mve_vrev64q_x_u8, 32626, 32519},
  { ARM::BI__builtin_arm_mve_vrhaddq_m_s16, 32649, 32639},
  { ARM::BI__builtin_arm_mve_vrhaddq_m_s32, 32663, 32639},
  { ARM::BI__builtin_arm_mve_vrhaddq_m_s8, 32677, 32639},
  { ARM::BI__builtin_arm_mve_vrhaddq_m_u16, 32690, 32639},
  { ARM::BI__builtin_arm_mve_vrhaddq_m_u32, 32704, 32639},
  { ARM::BI__builtin_arm_mve_vrhaddq_m_u8, 32718, 32639},
  { ARM::BI__builtin_arm_mve_vrhaddq_s16, 32739, 32731},
  { ARM::BI__builtin_arm_mve_vrhaddq_s32, 32751, 32731},
  { ARM::BI__builtin_arm_mve_vrhaddq_s8, 32763, 32731},
  { ARM::BI__builtin_arm_mve_vrhaddq_u16, 32774, 32731},
  { ARM::BI__builtin_arm_mve_vrhaddq_u32, 32786, 32731},
  { ARM::BI__builtin_arm_mve_vrhaddq_u8, 32798, 32731},
  { ARM::BI__builtin_arm_mve_vrhaddq_x_s16, 32819, 32809},
  { ARM::BI__builtin_arm_mve_vrhaddq_x_s32, 32833, 32809},
  { ARM::BI__builtin_arm_mve_vrhaddq_x_s8, 32847, 32809},
  { ARM::BI__builtin_arm_mve_vrhaddq_x_u16, 32860, 32809},
  { ARM::BI__builtin_arm_mve_vrhaddq_x_u32, 32874, 32809},
  { ARM::BI__builtin_arm_mve_vrhaddq_x_u8, 32888, 32809},
  { ARM::BI__builtin_arm_mve_vrmlaldavhaq_p_s32, 32916, 32901},
  { ARM::BI__builtin_arm_mve_vrmlaldavhaq_p_u32, 32935, 32901},
  { ARM::BI__builtin_arm_mve_vrmlaldavhaq_s32, 32967, 32954},
  { ARM::BI__builtin_arm_mve_vrmlaldavhaq_u32, 32984, 32954},
  { ARM::BI__builtin_arm_mve_vrmlaldavhaxq_p_s32, 33017, 33001},
  { ARM::BI__builtin_arm_mve_vrmlaldavhaxq_s32, 33051, 33037},
  { ARM::BI__builtin_arm_mve_vrmlaldavhq_p_s32, 33083, 33069},
  { ARM::BI__builtin_arm_mve_vrmlaldavhq_p_u32, 33101, 33069},
  { ARM::BI__builtin_arm_mve_vrmlaldavhq_s32, 33131, 33119},
  { ARM::BI__builtin_arm_mve_vrmlaldavhq_u32, 33147, 33119},
  { ARM::BI__builtin_arm_mve_vrmlaldavhxq_p_s32, 33178, 33163},
  { ARM::BI__builtin_arm_mve_vrmlaldavhxq_s32, 33210, 33197},
  { ARM::BI__builtin_arm_mve_vrmlsldavhaq_p_s32, 33242, 33227},
  { ARM::BI__builtin_arm_mve_vrmlsldavhaq_s32, 33274, 33261},
  { ARM::BI__builtin_arm_mve_vrmlsldavhaxq_p_s32, 33307, 33291},
  { ARM::BI__builtin_arm_mve_vrmlsldavhaxq_s32, 33341, 33327},
  { ARM::BI__builtin_arm_mve_vrmlsldavhq_p_s32, 33373, 33359},
  { ARM::BI__builtin_arm_mve_vrmlsldavhq_s32, 33403, 33391},
  { ARM::BI__builtin_arm_mve_vrmlsldavhxq_p_s32, 33434, 33419},
  { ARM::BI__builtin_arm_mve_vrmlsldavhxq_s32, 33466, 33453},
  { ARM::BI__builtin_arm_mve_vrmulhq_m_s16, 33493, 33483},
  { ARM::BI__builtin_arm_mve_vrmulhq_m_s32, 33507, 33483},
  { ARM::BI__builtin_arm_mve_vrmulhq_m_s8, 33521, 33483},
  { ARM::BI__builtin_arm_mve_vrmulhq_m_u16, 33534, 33483},
  { ARM::BI__builtin_arm_mve_vrmulhq_m_u32, 33548, 33483},
  { ARM::BI__builtin_arm_mve_vrmulhq_m_u8, 33562, 33483},
  { ARM::BI__builtin_arm_mve_vrmulhq_s16, 33583, 33575},
  { ARM::BI__builtin_arm_mve_vrmulhq_s32, 33595, 33575},
  { ARM::BI__builtin_arm_mve_vrmulhq_s8, 33607, 33575},
  { ARM::BI__builtin_arm_mve_vrmulhq_u16, 33618, 33575},
  { ARM::BI__builtin_arm_mve_vrmulhq_u32, 33630, 33575},
  { ARM::BI__builtin_arm_mve_vrmulhq_u8, 33642, 33575},
  { ARM::BI__builtin_arm_mve_vrmulhq_x_s16, 33663, 33653},
  { ARM::BI__builtin_arm_mve_vrmulhq_x_s32, 33677, 33653},
  { ARM::BI__builtin_arm_mve_vrmulhq_x_s8, 33691, 33653},
  { ARM::BI__builtin_arm_mve_vrmulhq_x_u16, 33704, 33653},
  { ARM::BI__builtin_arm_mve_vrmulhq_x_u32, 33718, 33653},
  { ARM::BI__builtin_arm_mve_vrmulhq_x_u8, 33732, 33653},
  { ARM::BI__builtin_arm_mve_vrndaq_f16, 33752, 33745},
  { ARM::BI__builtin_arm_mve_vrndaq_f32, 33763, 33745},
  { ARM::BI__builtin_arm_mve_vrndaq_m_f16, 33783, 33774},
  { ARM::BI__builtin_arm_mve_vrndaq_m_f32, 33796, 33774},
  { ARM::BI__builtin_arm_mve_vrndaq_x_f16, 33818, 33809},
  { ARM::BI__builtin_arm_mve_vrndaq_x_f32, 33831, 33809},
  { ARM::BI__builtin_arm_mve_vrndmq_f16, 33851, 33844},
  { ARM::BI__builtin_arm_mve_vrndmq_f32, 33862, 33844},
  { ARM::BI__builtin_arm_mve_vrndmq_m_f16, 33882, 33873},
  { ARM::BI__builtin_arm_mve_vrndmq_m_f32, 33895, 33873},
  { ARM::BI__builtin_arm_mve_vrndmq_x_f16, 33917, 33908},
  { ARM::BI__builtin_arm_mve_vrndmq_x_f32, 33930, 33908},
  { ARM::BI__builtin_arm_mve_vrndnq_f16, 33950, 33943},
  { ARM::BI__builtin_arm_mve_vrndnq_f32, 33961, 33943},
  { ARM::BI__builtin_arm_mve_vrndnq_m_f16, 33981, 33972},
  { ARM::BI__builtin_arm_mve_vrndnq_m_f32, 33994, 33972},
  { ARM::BI__builtin_arm_mve_vrndnq_x_f16, 34016, 34007},
  { ARM::BI__builtin_arm_mve_vrndnq_x_f32, 34029, 34007},
  { ARM::BI__builtin_arm_mve_vrndpq_f16, 34049, 34042},
  { ARM::BI__builtin_arm_mve_vrndpq_f32, 34060, 34042},
  { ARM::BI__builtin_arm_mve_vrndpq_m_f16, 34080, 34071},
  { ARM::BI__builtin_arm_mve_vrndpq_m_f32, 34093, 34071},
  { ARM::BI__builtin_arm_mve_vrndpq_x_f16, 34115, 34106},
  { ARM::BI__builtin_arm_mve_vrndpq_x_f32, 34128, 34106},
  { ARM::BI__builtin_arm_mve_vrndq_f16, 34147, 34141},
  { ARM::BI__builtin_arm_mve_vrndq_f32, 34157, 34141},
  { ARM::BI__builtin_arm_mve_vrndq_m_f16, 34175, 34167},
  { ARM::BI__builtin_arm_mve_vrndq_m_f32, 34187, 34167},
  { ARM::BI__builtin_arm_mve_vrndq_x_f16, 34207, 34199},
  { ARM::BI__builtin_arm_mve_vrndq_x_f32, 34219, 34199},
  { ARM::BI__builtin_arm_mve_vrndxq_f16, 34238, 34231},
  { ARM::BI__builtin_arm_mve_vrndxq_f32, 34249, 34231},
  { ARM::BI__builtin_arm_mve_vrndxq_m_f16, 34269, 34260},
  { ARM::BI__builtin_arm_mve_vrndxq_m_f32, 34282, 34260},
  { ARM::BI__builtin_arm_mve_vrndxq_x_f16, 34304, 34295},
  { ARM::BI__builtin_arm_mve_vrndxq_x_f32, 34317, 34295},
  { ARM::BI__builtin_arm_mve_vrshlq_m_n_s16, 34341, 34330},
  { ARM::BI__builtin_arm_mve_vrshlq_m_n_s32, 34356, 34330},
  { ARM::BI__builtin_arm_mve_vrshlq_m_n_s8, 34371, 34330},
  { ARM::BI__builtin_arm_mve_vrshlq_m_n_u16, 34385, 34330},
  { ARM::BI__builtin_arm_mve_vrshlq_m_n_u32, 34400, 34330},
  { ARM::BI__builtin_arm_mve_vrshlq_m_n_u8, 34415, 34330},
  { ARM::BI__builtin_arm_mve_vrshlq_m_s16, 34438, 34429},
  { ARM::BI__builtin_arm_mve_vrshlq_m_s32, 34451, 34429},
  { ARM::BI__builtin_arm_mve_vrshlq_m_s8, 34464, 34429},
  { ARM::BI__builtin_arm_mve_vrshlq_m_u16, 34476, 34429},
  { ARM::BI__builtin_arm_mve_vrshlq_m_u32, 34489, 34429},
  { ARM::BI__builtin_arm_mve_vrshlq_m_u8, 34502, 34429},
  { ARM::BI__builtin_arm_mve_vrshlq_n_s16, 34521, 34514},
  { ARM::BI__builtin_arm_mve_vrshlq_n_s32, 34534, 34514},
  { ARM::BI__builtin_arm_mve_vrshlq_n_s8, 34547, 34514},
  { ARM::BI__builtin_arm_mve_vrshlq_n_u16, 34559, 34514},
  { ARM::BI__builtin_arm_mve_vrshlq_n_u32, 34572, 34514},
  { ARM::BI__builtin_arm_mve_vrshlq_n_u8, 34585, 34514},
  { ARM::BI__builtin_arm_mve_vrshlq_s16, 34597, 34514},
  { ARM::BI__builtin_arm_mve_vrshlq_s32, 34608, 34514},
  { ARM::BI__builtin_arm_mve_vrshlq_s8, 34619, 34514},
  { ARM::BI__builtin_arm_mve_vrshlq_u16, 34629, 34514},
  { ARM::BI__builtin_arm_mve_vrshlq_u32, 34640, 34514},
  { ARM::BI__builtin_arm_mve_vrshlq_u8, 34651, 34514},
  { ARM::BI__builtin_arm_mve_vrshlq_x_s16, 34670, 34661},
  { ARM::BI__builtin_arm_mve_vrshlq_x_s32, 34683, 34661},
  { ARM::BI__builtin_arm_mve_vrshlq_x_s8, 34696, 34661},
  { ARM::BI__builtin_arm_mve_vrshlq_x_u16, 34708, 34661},
  { ARM::BI__builtin_arm_mve_vrshlq_x_u32, 34721, 34661},
  { ARM::BI__builtin_arm_mve_vrshlq_x_u8, 34734, 34661},
  { ARM::BI__builtin_arm_mve_vrshrnbq_m_n_s16, 34757, 34746},
  { ARM::BI__builtin_arm_mve_vrshrnbq_m_n_s32, 34774, 34746},
  { ARM::BI__builtin_arm_mve_vrshrnbq_m_n_u16, 34791, 34746},
  { ARM::BI__builtin_arm_mve_vrshrnbq_m_n_u32, 34808, 34746},
  { ARM::BI__builtin_arm_mve_vrshrnbq_n_s16, 34834, 34825},
  { ARM::BI__builtin_arm_mve_vrshrnbq_n_s32, 34849, 34825},
  { ARM::BI__builtin_arm_mve_vrshrnbq_n_u16, 34864, 34825},
  { ARM::BI__builtin_arm_mve_vrshrnbq_n_u32, 34879, 34825},
  { ARM::BI__builtin_arm_mve_vrshrntq_m_n_s16, 34905, 34894},
  { ARM::BI__builtin_arm_mve_vrshrntq_m_n_s32, 34922, 34894},
  { ARM::BI__builtin_arm_mve_vrshrntq_m_n_u16, 34939, 34894},
  { ARM::BI__builtin_arm_mve_vrshrntq_m_n_u32, 34956, 34894},
  { ARM::BI__builtin_arm_mve_vrshrntq_n_s16, 34982, 34973},
  { ARM::BI__builtin_arm_mve_vrshrntq_n_s32, 34997, 34973},
  { ARM::BI__builtin_arm_mve_vrshrntq_n_u16, 35012, 34973},
  { ARM::BI__builtin_arm_mve_vrshrntq_n_u32, 35027, 34973},
  { ARM::BI__builtin_arm_mve_vrshrq_m_n_s16, 35051, 35042},
  { ARM::BI__builtin_arm_mve_vrshrq_m_n_s32, 35066, 35042},
  { ARM::BI__builtin_arm_mve_vrshrq_m_n_s8, 35081, 35042},
  { ARM::BI__builtin_arm_mve_vrshrq_m_n_u16, 35095, 35042},
  { ARM::BI__builtin_arm_mve_vrshrq_m_n_u32, 35110, 35042},
  { ARM::BI__builtin_arm_mve_vrshrq_m_n_u8, 35125, 35042},
  { ARM::BI__builtin_arm_mve_vrshrq_n_s16, 35146, 35139},
  { ARM::BI__builtin_arm_mve_vrshrq_n_s32, 35159, 35139},
  { ARM::BI__builtin_arm_mve_vrshrq_n_s8, 35172, 35139},
  { ARM::BI__builtin_arm_mve_vrshrq_n_u16, 35184, 35139},
  { ARM::BI__builtin_arm_mve_vrshrq_n_u32, 35197, 35139},
  { ARM::BI__builtin_arm_mve_vrshrq_n_u8, 35210, 35139},
  { ARM::BI__builtin_arm_mve_vrshrq_x_n_s16, 35231, 35222},
  { ARM::BI__builtin_arm_mve_vrshrq_x_n_s32, 35246, 35222},
  { ARM::BI__builtin_arm_mve_vrshrq_x_n_s8, 35261, 35222},
  { ARM::BI__builtin_arm_mve_vrshrq_x_n_u16, 35275, 35222},
  { ARM::BI__builtin_arm_mve_vrshrq_x_n_u32, 35290, 35222},
  { ARM::BI__builtin_arm_mve_vrshrq_x_n_u8, 35305, 35222},
  { ARM::BI__builtin_arm_mve_vsbciq_m_s32, 35328, 35319},
  { ARM::BI__builtin_arm_mve_vsbciq_m_u32, 35341, 35319},
  { ARM::BI__builtin_arm_mve_vsbciq_s32, 35361, 35354},
  { ARM::BI__builtin_arm_mve_vsbciq_u32, 35372, 35354},
  { ARM::BI__builtin_arm_mve_vsbcq_m_s32, 35391, 35383},
  { ARM::BI__builtin_arm_mve_vsbcq_m_u32, 35403, 35383},
  { ARM::BI__builtin_arm_mve_vsbcq_s32, 35421, 35415},
  { ARM::BI__builtin_arm_mve_vsbcq_u32, 35431, 35415},
  { ARM::BI__builtin_arm_mve_vsetq_lane_f16, 35452, 35441},
  { ARM::BI__builtin_arm_mve_vsetq_lane_f32, 35467, 35441},
  { ARM::BI__builtin_arm_mve_vsetq_lane_s16, 35482, 35441},
  { ARM::BI__builtin_arm_mve_vsetq_lane_s32, 35497, 35441},
  { ARM::BI__builtin_arm_mve_vsetq_lane_s64, 35512, 35441},
  { ARM::BI__builtin_arm_mve_vsetq_lane_s8, 35527, 35441},
  { ARM::BI__builtin_arm_mve_vsetq_lane_u16, 35541, 35441},
  { ARM::BI__builtin_arm_mve_vsetq_lane_u32, 35556, 35441},
  { ARM::BI__builtin_arm_mve_vsetq_lane_u64, 35571, 35441},
  { ARM::BI__builtin_arm_mve_vsetq_lane_u8, 35586, 35441},
  { ARM::BI__builtin_arm_mve_vshlcq_m_s16, 35609, 35600},
  { ARM::BI__builtin_arm_mve_vshlcq_m_s32, 35622, 35600},
  { ARM::BI__builtin_arm_mve_vshlcq_m_s8, 35635, 35600},
  { ARM::BI__builtin_arm_mve_vshlcq_m_u16, 35647, 35600},
  { ARM::BI__builtin_arm_mve_vshlcq_m_u32, 35660, 35600},
  { ARM::BI__builtin_arm_mve_vshlcq_m_u8, 35673, 35600},
  { ARM::BI__builtin_arm_mve_vshlcq_s16, 35692, 35685},
  { ARM::BI__builtin_arm_mve_vshlcq_s32, 35703, 35685},
  { ARM::BI__builtin_arm_mve_vshlcq_s8, 35714, 35685},
  { ARM::BI__builtin_arm_mve_vshlcq_u16, 35724, 35685},
  { ARM::BI__builtin_arm_mve_vshlcq_u32, 35735, 35685},
  { ARM::BI__builtin_arm_mve_vshlcq_u8, 35746, 35685},
  { ARM::BI__builtin_arm_mve_vshllbq_m_n_s16, 35766, 35756},
  { ARM::BI__builtin_arm_mve_vshllbq_m_n_s8, 35782, 35756},
  { ARM::BI__builtin_arm_mve_vshllbq_m_n_u16, 35797, 35756},
  { ARM::BI__builtin_arm_mve_vshllbq_m_n_u8, 35813, 35756},
  { ARM::BI__builtin_arm_mve_vshllbq_n_s16, 35836, 35828},
  { ARM::BI__builtin_arm_mve_vshllbq_n_s8, 35850, 35828},
  { ARM::BI__builtin_arm_mve_vshllbq_n_u16, 35863, 35828},
  { ARM::BI__builtin_arm_mve_vshllbq_n_u8, 35877, 35828},
  { ARM::BI__builtin_arm_mve_vshllbq_x_n_s16, 35900, 35890},
  { ARM::BI__builtin_arm_mve_vshllbq_x_n_s8, 35916, 35890},
  { ARM::BI__builtin_arm_mve_vshllbq_x_n_u16, 35931, 35890},
  { ARM::BI__builtin_arm_mve_vshllbq_x_n_u8, 35947, 35890},
  { ARM::BI__builtin_arm_mve_vshlltq_m_n_s16, 35972, 35962},
  { ARM::BI__builtin_arm_mve_vshlltq_m_n_s8, 35988, 35962},
  { ARM::BI__builtin_arm_mve_vshlltq_m_n_u16, 36003, 35962},
  { ARM::BI__builtin_arm_mve_vshlltq_m_n_u8, 36019, 35962},
  { ARM::BI__builtin_arm_mve_vshlltq_n_s16, 36042, 36034},
  { ARM::BI__builtin_arm_mve_vshlltq_n_s8, 36056, 36034},
  { ARM::BI__builtin_arm_mve_vshlltq_n_u16, 36069, 36034},
  { ARM::BI__builtin_arm_mve_vshlltq_n_u8, 36083, 36034},
  { ARM::BI__builtin_arm_mve_vshlltq_x_n_s16, 36106, 36096},
  { ARM::BI__builtin_arm_mve_vshlltq_x_n_s8, 36122, 36096},
  { ARM::BI__builtin_arm_mve_vshlltq_x_n_u16, 36137, 36096},
  { ARM::BI__builtin_arm_mve_vshlltq_x_n_u8, 36153, 36096},
  { ARM::BI__builtin_arm_mve_vshlq_m_n_s16, 36178, 36168},
  { ARM::BI__builtin_arm_mve_vshlq_m_n_s32, 36192, 36168},
  { ARM::BI__builtin_arm_mve_vshlq_m_n_s8, 36206, 36168},
  { ARM::BI__builtin_arm_mve_vshlq_m_n_u16, 36219, 36168},
  { ARM::BI__builtin_arm_mve_vshlq_m_n_u32, 36233, 36168},
  { ARM::BI__builtin_arm_mve_vshlq_m_n_u8, 36247, 36168},
  { ARM::BI__builtin_arm_mve_vshlq_m_r_s16, 36270, 36260},
  { ARM::BI__builtin_arm_mve_vshlq_m_r_s32, 36284, 36260},
  { ARM::BI__builtin_arm_mve_vshlq_m_r_s8, 36298, 36260},
  { ARM::BI__builtin_arm_mve_vshlq_m_r_u16, 36311, 36260},
  { ARM::BI__builtin_arm_mve_vshlq_m_r_u32, 36325, 36260},
  { ARM::BI__builtin_arm_mve_vshlq_m_r_u8, 36339, 36260},
  { ARM::BI__builtin_arm_mve_vshlq_m_s16, 36360, 36352},
  { ARM::BI__builtin_arm_mve_vshlq_m_s32, 36372, 36352},
  { ARM::BI__builtin_arm_mve_vshlq_m_s8, 36384, 36352},
  { ARM::BI__builtin_arm_mve_vshlq_m_u16, 36395, 36352},
  { ARM::BI__builtin_arm_mve_vshlq_m_u32, 36407, 36352},
  { ARM::BI__builtin_arm_mve_vshlq_m_u8, 36419, 36352},
  { ARM::BI__builtin_arm_mve_vshlq_n_s16, 36438, 36430},
  { ARM::BI__builtin_arm_mve_vshlq_n_s32, 36450, 36430},
  { ARM::BI__builtin_arm_mve_vshlq_n_s8, 36462, 36430},
  { ARM::BI__builtin_arm_mve_vshlq_n_u16, 36473, 36430},
  { ARM::BI__builtin_arm_mve_vshlq_n_u32, 36485, 36430},
  { ARM::BI__builtin_arm_mve_vshlq_n_u8, 36497, 36430},
  { ARM::BI__builtin_arm_mve_vshlq_r_s16, 36516, 36508},
  { ARM::BI__builtin_arm_mve_vshlq_r_s32, 36528, 36508},
  { ARM::BI__builtin_arm_mve_vshlq_r_s8, 36540, 36508},
  { ARM::BI__builtin_arm_mve_vshlq_r_u16, 36551, 36508},
  { ARM::BI__builtin_arm_mve_vshlq_r_u32, 36563, 36508},
  { ARM::BI__builtin_arm_mve_vshlq_r_u8, 36575, 36508},
  { ARM::BI__builtin_arm_mve_vshlq_s16, 36592, 36586},
  { ARM::BI__builtin_arm_mve_vshlq_s32, 36602, 36586},
  { ARM::BI__builtin_arm_mve_vshlq_s8, 36612, 36586},
  { ARM::BI__builtin_arm_mve_vshlq_u16, 36621, 36586},
  { ARM::BI__builtin_arm_mve_vshlq_u32, 36631, 36586},
  { ARM::BI__builtin_arm_mve_vshlq_u8, 36641, 36586},
  { ARM::BI__builtin_arm_mve_vshlq_x_n_s16, 36660, 36650},
  { ARM::BI__builtin_arm_mve_vshlq_x_n_s32, 36674, 36650},
  { ARM::BI__builtin_arm_mve_vshlq_x_n_s8, 36688, 36650},
  { ARM::BI__builtin_arm_mve_vshlq_x_n_u16, 36701, 36650},
  { ARM::BI__builtin_arm_mve_vshlq_x_n_u32, 36715, 36650},
  { ARM::BI__builtin_arm_mve_vshlq_x_n_u8, 36729, 36650},
  { ARM::BI__builtin_arm_mve_vshlq_x_s16, 36750, 36742},
  { ARM::BI__builtin_arm_mve_vshlq_x_s32, 36762, 36742},
  { ARM::BI__builtin_arm_mve_vshlq_x_s8, 36774, 36742},
  { ARM::BI__builtin_arm_mve_vshlq_x_u16, 36785, 36742},
  { ARM::BI__builtin_arm_mve_vshlq_x_u32, 36797, 36742},
  { ARM::BI__builtin_arm_mve_vshlq_x_u8, 36809, 36742},
  { ARM::BI__builtin_arm_mve_vshrnbq_m_n_s16, 36830, 36820},
  { ARM::BI__builtin_arm_mve_vshrnbq_m_n_s32, 36846, 36820},
  { ARM::BI__builtin_arm_mve_vshrnbq_m_n_u16, 36862, 36820},
  { ARM::BI__builtin_arm_mve_vshrnbq_m_n_u32, 36878, 36820},
  { ARM::BI__builtin_arm_mve_vshrnbq_n_s16, 36902, 36894},
  { ARM::BI__builtin_arm_mve_vshrnbq_n_s32, 36916, 36894},
  { ARM::BI__builtin_arm_mve_vshrnbq_n_u16, 36930, 36894},
  { ARM::BI__builtin_arm_mve_vshrnbq_n_u32, 36944, 36894},
  { ARM::BI__builtin_arm_mve_vshrntq_m_n_s16, 36968, 36958},
  { ARM::BI__builtin_arm_mve_vshrntq_m_n_s32, 36984, 36958},
  { ARM::BI__builtin_arm_mve_vshrntq_m_n_u16, 37000, 36958},
  { ARM::BI__builtin_arm_mve_vshrntq_m_n_u32, 37016, 36958},
  { ARM::BI__builtin_arm_mve_vshrntq_n_s16, 37040, 37032},
  { ARM::BI__builtin_arm_mve_vshrntq_n_s32, 37054, 37032},
  { ARM::BI__builtin_arm_mve_vshrntq_n_u16, 37068, 37032},
  { ARM::BI__builtin_arm_mve_vshrntq_n_u32, 37082, 37032},
  { ARM::BI__builtin_arm_mve_vshrq_m_n_s16, 37104, 37096},
  { ARM::BI__builtin_arm_mve_vshrq_m_n_s32, 37118, 37096},
  { ARM::BI__builtin_arm_mve_vshrq_m_n_s8, 37132, 37096},
  { ARM::BI__builtin_arm_mve_vshrq_m_n_u16, 37145, 37096},
  { ARM::BI__builtin_arm_mve_vshrq_m_n_u32, 37159, 37096},
  { ARM::BI__builtin_arm_mve_vshrq_m_n_u8, 37173, 37096},
  { ARM::BI__builtin_arm_mve_vshrq_n_s16, 37192, 37186},
  { ARM::BI__builtin_arm_mve_vshrq_n_s32, 37204, 37186},
  { ARM::BI__builtin_arm_mve_vshrq_n_s8, 37216, 37186},
  { ARM::BI__builtin_arm_mve_vshrq_n_u16, 37227, 37186},
  { ARM::BI__builtin_arm_mve_vshrq_n_u32, 37239, 37186},
  { ARM::BI__builtin_arm_mve_vshrq_n_u8, 37251, 37186},
  { ARM::BI__builtin_arm_mve_vshrq_x_n_s16, 37270, 37262},
  { ARM::BI__builtin_arm_mve_vshrq_x_n_s32, 37284, 37262},
  { ARM::BI__builtin_arm_mve_vshrq_x_n_s8, 37298, 37262},
  { ARM::BI__builtin_arm_mve_vshrq_x_n_u16, 37311, 37262},
  { ARM::BI__builtin_arm_mve_vshrq_x_n_u32, 37325, 37262},
  { ARM::BI__builtin_arm_mve_vshrq_x_n_u8, 37339, 37262},
  { ARM::BI__builtin_arm_mve_vsliq_m_n_s16, 37360, 37352},
  { ARM::BI__builtin_arm_mve_vsliq_m_n_s32, 37374, 37352},
  { ARM::BI__builtin_arm_mve_vsliq_m_n_s8, 37388, 37352},
  { ARM::BI__builtin_arm_mve_vsliq_m_n_u16, 37401, 37352},
  { ARM::BI__builtin_arm_mve_vsliq_m_n_u32, 37415, 37352},
  { ARM::BI__builtin_arm_mve_vsliq_m_n_u8, 37429, 37352},
  { ARM::BI__builtin_arm_mve_vsliq_n_s16, 37448, 37442},
  { ARM::BI__builtin_arm_mve_vsliq_n_s32, 37460, 37442},
  { ARM::BI__builtin_arm_mve_vsliq_n_s8, 37472, 37442},
  { ARM::BI__builtin_arm_mve_vsliq_n_u16, 37483, 37442},
  { ARM::BI__builtin_arm_mve_vsliq_n_u32, 37495, 37442},
  { ARM::BI__builtin_arm_mve_vsliq_n_u8, 37507, 37442},
  { ARM::BI__builtin_arm_mve_vsriq_m_n_s16, 37526, 37518},
  { ARM::BI__builtin_arm_mve_vsriq_m_n_s32, 37540, 37518},
  { ARM::BI__builtin_arm_mve_vsriq_m_n_s8, 37554, 37518},
  { ARM::BI__builtin_arm_mve_vsriq_m_n_u16, 37567, 37518},
  { ARM::BI__builtin_arm_mve_vsriq_m_n_u32, 37581, 37518},
  { ARM::BI__builtin_arm_mve_vsriq_m_n_u8, 37595, 37518},
  { ARM::BI__builtin_arm_mve_vsriq_n_s16, 37614, 37608},
  { ARM::BI__builtin_arm_mve_vsriq_n_s32, 37626, 37608},
  { ARM::BI__builtin_arm_mve_vsriq_n_s8, 37638, 37608},
  { ARM::BI__builtin_arm_mve_vsriq_n_u16, 37649, 37608},
  { ARM::BI__builtin_arm_mve_vsriq_n_u32, 37661, 37608},
  { ARM::BI__builtin_arm_mve_vsriq_n_u8, 37673, 37608},
  { ARM::BI__builtin_arm_mve_vst1q_f16, 37690, 37684},
  { ARM::BI__builtin_arm_mve_vst1q_f32, 37700, 37684},
  { ARM::BI__builtin_arm_mve_vst1q_p_f16, 37718, 37710},
  { ARM::BI__builtin_arm_mve_vst1q_p_f32, 37730, 37710},
  { ARM::BI__builtin_arm_mve_vst1q_p_s16, 37742, 37710},
  { ARM::BI__builtin_arm_mve_vst1q_p_s32, 37754, 37710},
  { ARM::BI__builtin_arm_mve_vst1q_p_s8, 37766, 37710},
  { ARM::BI__builtin_arm_mve_vst1q_p_u16, 37777, 37710},
  { ARM::BI__builtin_arm_mve_vst1q_p_u32, 37789, 37710},
  { ARM::BI__builtin_arm_mve_vst1q_p_u8, 37801, 37710},
  { ARM::BI__builtin_arm_mve_vst1q_s16, 37812, 37684},
  { ARM::BI__builtin_arm_mve_vst1q_s32, 37822, 37684},
  { ARM::BI__builtin_arm_mve_vst1q_s8, 37832, 37684},
  { ARM::BI__builtin_arm_mve_vst1q_u16, 37841, 37684},
  { ARM::BI__builtin_arm_mve_vst1q_u32, 37851, 37684},
  { ARM::BI__builtin_arm_mve_vst1q_u8, 37861, 37684},
  { ARM::BI__builtin_arm_mve_vst2q_f16, 37876, 37870},
  { ARM::BI__builtin_arm_mve_vst2q_f32, 37886, 37870},
  { ARM::BI__builtin_arm_mve_vst2q_s16, 37896, 37870},
  { ARM::BI__builtin_arm_mve_vst2q_s32, 37906, 37870},
  { ARM::BI__builtin_arm_mve_vst2q_s8, 37916, 37870},
  { ARM::BI__builtin_arm_mve_vst2q_u16, 37925, 37870},
  { ARM::BI__builtin_arm_mve_vst2q_u32, 37935, 37870},
  { ARM::BI__builtin_arm_mve_vst2q_u8, 37945, 37870},
  { ARM::BI__builtin_arm_mve_vst4q_f16, 37960, 37954},
  { ARM::BI__builtin_arm_mve_vst4q_f32, 37970, 37954},
  { ARM::BI__builtin_arm_mve_vst4q_s16, 37980, 37954},
  { ARM::BI__builtin_arm_mve_vst4q_s32, 37990, 37954},
  { ARM::BI__builtin_arm_mve_vst4q_s8, 38000, 37954},
  { ARM::BI__builtin_arm_mve_vst4q_u16, 38009, 37954},
  { ARM::BI__builtin_arm_mve_vst4q_u32, 38019, 37954},
  { ARM::BI__builtin_arm_mve_vst4q_u8, 38029, 37954},
  { ARM::BI__builtin_arm_mve_vstrbq_p_s16, 38047, 38038},
  { ARM::BI__builtin_arm_mve_vstrbq_p_s32, 38060, 38038},
  { ARM::BI__builtin_arm_mve_vstrbq_p_s8, 38073, 38038},
  { ARM::BI__builtin_arm_mve_vstrbq_p_u16, 38085, 38038},
  { ARM::BI__builtin_arm_mve_vstrbq_p_u32, 38098, 38038},
  { ARM::BI__builtin_arm_mve_vstrbq_p_u8, 38111, 38038},
  { ARM::BI__builtin_arm_mve_vstrbq_s16, 38130, 38123},
  { ARM::BI__builtin_arm_mve_vstrbq_s32, 38141, 38123},
  { ARM::BI__builtin_arm_mve_vstrbq_s8, 38152, 38123},
  { ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_s16, 38186, 38162},
  { ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_s32, 38214, 38162},
  { ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_s8, 38242, 38162},
  { ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_u16, 38269, 38162},
  { ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_u32, 38297, 38162},
  { ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_p_u8, 38325, 38162},
  { ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_s16, 38374, 38352},
  { ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_s32, 38400, 38352},
  { ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_s8, 38426, 38352},
  { ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_u16, 38451, 38352},
  { ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_u32, 38477, 38352},
  { ARM::BI__builtin_arm_mve_vstrbq_scatter_offset_u8, 38503, 38352},
  { ARM::BI__builtin_arm_mve_vstrbq_u16, 38528, 38123},
  { ARM::BI__builtin_arm_mve_vstrbq_u32, 38539, 38123},
  { ARM::BI__builtin_arm_mve_vstrbq_u8, 38550, 38123},
  { ARM::BI__builtin_arm_mve_vstrdq_scatter_base_p_s64, 38582, 38560},
  { ARM::BI__builtin_arm_mve_vstrdq_scatter_base_p_u64, 38608, 38560},
  { ARM::BI__builtin_arm_mve_vstrdq_scatter_base_s64, 38654, 38634},
  { ARM::BI__builtin_arm_mve_vstrdq_scatter_base_u64, 38678, 38634},
  { ARM::BI__builtin_arm_mve_vstrdq_scatter_base_wb_p_s64, 38727, 38702},
  { ARM::BI__builtin_arm_mve_vstrdq_scatter_base_wb_p_u64, 38756, 38702},
  { ARM::BI__builtin_arm_mve_vstrdq_scatter_base_wb_s64, 38808, 38785},
  { ARM::BI__builtin_arm_mve_vstrdq_scatter_base_wb_u64, 38835, 38785},
  { ARM::BI__builtin_arm_mve_vstrdq_scatter_offset_p_s64, 38886, 38862},
  { ARM::BI__builtin_arm_mve_vstrdq_scatter_offset_p_u64, 38914, 38862},
  { ARM::BI__builtin_arm_mve_vstrdq_scatter_offset_s64, 38964, 38942},
  { ARM::BI__builtin_arm_mve_vstrdq_scatter_offset_u64, 38990, 38942},
  { ARM::BI__builtin_arm_mve_vstrdq_scatter_shifted_offset_p_s64, 39048, 39016},
  { ARM::BI__builtin_arm_mve_vstrdq_scatter_shifted_offset_p_u64, 39084, 39016},
  { ARM::BI__builtin_arm_mve_vstrdq_scatter_shifted_offset_s64, 39150, 39120},
  { ARM::BI__builtin_arm_mve_vstrdq_scatter_shifted_offset_u64, 39184, 39120},
  { ARM::BI__builtin_arm_mve_vstrhq_f16, 39225, 39218},
  { ARM::BI__builtin_arm_mve_vstrhq_p_f16, 39245, 39236},
  { ARM::BI__builtin_arm_mve_vstrhq_p_s16, 39258, 39236},
  { ARM::BI__builtin_arm_mve_vstrhq_p_s32, 39271, 39236},
  { ARM::BI__builtin_arm_mve_vstrhq_p_u16, 39284, 39236},
  { ARM::BI__builtin_arm_mve_vstrhq_p_u32, 39297, 39236},
  { ARM::BI__builtin_arm_mve_vstrhq_s16, 39310, 39218},
  { ARM::BI__builtin_arm_mve_vstrhq_s32, 39321, 39218},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_f16, 39354, 39332},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_p_f16, 39404, 39380},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_p_s16, 39432, 39380},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_p_s32, 39460, 39380},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_p_u16, 39488, 39380},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_p_u32, 39516, 39380},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_s16, 39544, 39332},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_s32, 39570, 39332},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_u16, 39596, 39332},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_offset_u32, 39622, 39332},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_f16, 39678, 39648},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_p_f16, 39744, 39712},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_p_s16, 39780, 39712},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_p_s32, 39816, 39712},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_p_u16, 39852, 39712},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_p_u32, 39888, 39712},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_s16, 39924, 39648},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_s32, 39958, 39648},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_u16, 39992, 39648},
  { ARM::BI__builtin_arm_mve_vstrhq_scatter_shifted_offset_u32, 40026, 39648},
  { ARM::BI__builtin_arm_mve_vstrhq_u16, 40060, 39218},
  { ARM::BI__builtin_arm_mve_vstrhq_u32, 40071, 39218},
  { ARM::BI__builtin_arm_mve_vstrwq_f32, 40089, 40082},
  { ARM::BI__builtin_arm_mve_vstrwq_p_f32, 40109, 40100},
  { ARM::BI__builtin_arm_mve_vstrwq_p_s32, 40122, 40100},
  { ARM::BI__builtin_arm_mve_vstrwq_p_u32, 40135, 40100},
  { ARM::BI__builtin_arm_mve_vstrwq_s32, 40148, 40082},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_base_f32, 40179, 40159},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_base_p_f32, 40225, 40203},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_base_p_s32, 40251, 40203},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_base_p_u32, 40277, 40203},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_base_s32, 40303, 40159},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_base_u32, 40327, 40159},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_f32, 40374, 40351},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_p_f32, 40426, 40401},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_p_s32, 40455, 40401},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_p_u32, 40484, 40401},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_s32, 40513, 40351},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_base_wb_u32, 40540, 40351},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_f32, 40589, 40567},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_p_f32, 40639, 40615},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_p_s32, 40667, 40615},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_p_u32, 40695, 40615},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_s32, 40723, 40567},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_offset_u32, 40749, 40567},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_f32, 40805, 40775},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_p_f32, 40871, 40839},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_p_s32, 40907, 40839},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_p_u32, 40943, 40839},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_s32, 40979, 40775},
  { ARM::BI__builtin_arm_mve_vstrwq_scatter_shifted_offset_u32, 41013, 40775},
  { ARM::BI__builtin_arm_mve_vstrwq_u32, 41047, 40082},
  { ARM::BI__builtin_arm_mve_vsubq_f16, 41064, 41058},
  { ARM::BI__builtin_arm_mve_vsubq_f32, 41074, 41058},
  { ARM::BI__builtin_arm_mve_vsubq_m_f16, 41092, 41084},
  { ARM::BI__builtin_arm_mve_vsubq_m_f32, 41104, 41084},
  { ARM::BI__builtin_arm_mve_vsubq_m_n_f16, 41116, 41084},
  { ARM::BI__builtin_arm_mve_vsubq_m_n_f32, 41130, 41084},
  { ARM::BI__builtin_arm_mve_vsubq_m_n_s16, 41144, 41084},
  { ARM::BI__builtin_arm_mve_vsubq_m_n_s32, 41158, 41084},
  { ARM::BI__builtin_arm_mve_vsubq_m_n_s8, 41172, 41084},
  { ARM::BI__builtin_arm_mve_vsubq_m_n_u16, 41185, 41084},
  { ARM::BI__builtin_arm_mve_vsubq_m_n_u32, 41199, 41084},
  { ARM::BI__builtin_arm_mve_vsubq_m_n_u8, 41213, 41084},
  { ARM::BI__builtin_arm_mve_vsubq_m_s16, 41226, 41084},
  { ARM::BI__builtin_arm_mve_vsubq_m_s32, 41238, 41084},
  { ARM::BI__builtin_arm_mve_vsubq_m_s8, 41250, 41084},
  { ARM::BI__builtin_arm_mve_vsubq_m_u16, 41261, 41084},
  { ARM::BI__builtin_arm_mve_vsubq_m_u32, 41273, 41084},
  { ARM::BI__builtin_arm_mve_vsubq_m_u8, 41285, 41084},
  { ARM::BI__builtin_arm_mve_vsubq_n_f16, 41296, 41058},
  { ARM::BI__builtin_arm_mve_vsubq_n_f32, 41308, 41058},
  { ARM::BI__builtin_arm_mve_vsubq_n_s16, 41320, 41058},
  { ARM::BI__builtin_arm_mve_vsubq_n_s32, 41332, 41058},
  { ARM::BI__builtin_arm_mve_vsubq_n_s8, 41344, 41058},
  { ARM::BI__builtin_arm_mve_vsubq_n_u16, 41355, 41058},
  { ARM::BI__builtin_arm_mve_vsubq_n_u32, 41367, 41058},
  { ARM::BI__builtin_arm_mve_vsubq_n_u8, 41379, 41058},
  { ARM::BI__builtin_arm_mve_vsubq_s16, 41390, 41058},
  { ARM::BI__builtin_arm_mve_vsubq_s32, 41400, 41058},
  { ARM::BI__builtin_arm_mve_vsubq_s8, 41410, 41058},
  { ARM::BI__builtin_arm_mve_vsubq_u16, 41419, 41058},
  { ARM::BI__builtin_arm_mve_vsubq_u32, 41429, 41058},
  { ARM::BI__builtin_arm_mve_vsubq_u8, 41439, 41058},
  { ARM::BI__builtin_arm_mve_vsubq_x_f16, 41456, 41448},
  { ARM::BI__builtin_arm_mve_vsubq_x_f32, 41468, 41448},
  { ARM::BI__builtin_arm_mve_vsubq_x_n_f16, 41480, 41448},
  { ARM::BI__builtin_arm_mve_vsubq_x_n_f32, 41494, 41448},
  { ARM::BI__builtin_arm_mve_vsubq_x_n_s16, 41508, 41448},
  { ARM::BI__builtin_arm_mve_vsubq_x_n_s32, 41522, 41448},
  { ARM::BI__builtin_arm_mve_vsubq_x_n_s8, 41536, 41448},
  { ARM::BI__builtin_arm_mve_vsubq_x_n_u16, 41549, 41448},
  { ARM::BI__builtin_arm_mve_vsubq_x_n_u32, 41563, 41448},
  { ARM::BI__builtin_arm_mve_vsubq_x_n_u8, 41577, 41448},
  { ARM::BI__builtin_arm_mve_vsubq_x_s16, 41590, 41448},
  { ARM::BI__builtin_arm_mve_vsubq_x_s32, 41602, 41448},
  { ARM::BI__builtin_arm_mve_vsubq_x_s8, 41614, 41448},
  { ARM::BI__builtin_arm_mve_vsubq_x_u16, 41625, 41448},
  { ARM::BI__builtin_arm_mve_vsubq_x_u32, 41637, 41448},
  { ARM::BI__builtin_arm_mve_vsubq_x_u8, 41649, 41448},
  { ARM::BI__builtin_arm_mve_vuninitializedq_f16, 41660, -1},
  { ARM::BI__builtin_arm_mve_vuninitializedq_f32, 41680, -1},
  { ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_f16, 41716, 41700},
  { ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_f32, 41748, 41700},
  { ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_s16, 41780, 41700},
  { ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_s32, 41812, 41700},
  { ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_s64, 41844, 41700},
  { ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_s8, 41876, 41700},
  { ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_u16, 41907, 41700},
  { ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_u32, 41939, 41700},
  { ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_u64, 41971, 41700},
  { ARM::BI__builtin_arm_mve_vuninitializedq_polymorphic_u8, 42003, 41700},
  { ARM::BI__builtin_arm_mve_vuninitializedq_s16, 42034, -1},
  { ARM::BI__builtin_arm_mve_vuninitializedq_s32, 42054, -1},
  { ARM::BI__builtin_arm_mve_vuninitializedq_s64, 42074, -1},
  { ARM::BI__builtin_arm_mve_vuninitializedq_s8, 42094, -1},
  { ARM::BI__builtin_arm_mve_vuninitializedq_u16, 42113, -1},
  { ARM::BI__builtin_arm_mve_vuninitializedq_u32, 42133, -1},
  { ARM::BI__builtin_arm_mve_vuninitializedq_u64, 42153, -1},
  { ARM::BI__builtin_arm_mve_vuninitializedq_u8, 42173, -1},
};

ArrayRef<IntrinToName> Map(MapData);

static const char IntrinNames[] = {
    "\000asrl\000lsll\000sqrshr\000sqrshrl\000sqrshrl_sat48\000sqshl\000sqsh"
    "ll\000srshr\000srshrl\000uqrshl\000uqrshll\000uqrshll_sat48\000uqshl\000"
    "uqshll\000urshr\000urshrl\000vabavq_p\000vabavq_p_s16\000vabavq_p_s32\000"
    "vabavq_p_s8\000vabavq_p_u16\000vabavq_p_u32\000vabavq_p_u8\000vabavq\000"
    "vabavq_s16\000vabavq_s32\000vabavq_s8\000vabavq_u16\000vabavq_u32\000va"
    "bavq_u8\000vabdq\000vabdq_f16\000vabdq_f32\000vabdq_m\000vabdq_m_f16\000"
    "vabdq_m_f32\000vabdq_m_s16\000vabdq_m_s32\000vabdq_m_s8\000vabdq_m_u16\000"
    "vabdq_m_u32\000vabdq_m_u8\000vabdq_s16\000vabdq_s32\000vabdq_s8\000vabd"
    "q_u16\000vabdq_u32\000vabdq_u8\000vabdq_x\000vabdq_x_f16\000vabdq_x_f32"
    "\000vabdq_x_s16\000vabdq_x_s32\000vabdq_x_s8\000vabdq_x_u16\000vabdq_x_"
    "u32\000vabdq_x_u8\000vabsq\000vabsq_f16\000vabsq_f32\000vabsq_m\000vabs"
    "q_m_f16\000vabsq_m_f32\000vabsq_m_s16\000vabsq_m_s32\000vabsq_m_s8\000v"
    "absq_s16\000vabsq_s32\000vabsq_s8\000vabsq_x\000vabsq_x_f16\000vabsq_x_"
    "f32\000vabsq_x_s16\000vabsq_x_s32\000vabsq_x_s8\000vadciq_m\000vadciq_m"
    "_s32\000vadciq_m_u32\000vadciq\000vadciq_s32\000vadciq_u32\000vadcq_m\000"
    "vadcq_m_s32\000vadcq_m_u32\000vadcq\000vadcq_s32\000vadcq_u32\000vaddlv"
    "aq_p\000vaddlvaq_p_s32\000vaddlvaq_p_u32\000vaddlvaq\000vaddlvaq_s32\000"
    "vaddlvaq_u32\000vaddlvq_p\000vaddlvq_p_s32\000vaddlvq_p_u32\000vaddlvq\000"
    "vaddlvq_s32\000vaddlvq_u32\000vaddq\000vaddq_f16\000vaddq_f32\000vaddq_"
    "m\000vaddq_m_f16\000vaddq_m_f32\000vaddq_m_n_f16\000vaddq_m_n_f32\000va"
    "ddq_m_n_s16\000vaddq_m_n_s32\000vaddq_m_n_s8\000vaddq_m_n_u16\000vaddq_"
    "m_n_u32\000vaddq_m_n_u8\000vaddq_m_s16\000vaddq_m_s32\000vaddq_m_s8\000"
    "vaddq_m_u16\000vaddq_m_u32\000vaddq_m_u8\000vaddq_n_f16\000vaddq_n_f32\000"
    "vaddq_n_s16\000vaddq_n_s32\000vaddq_n_s8\000vaddq_n_u16\000vaddq_n_u32\000"
    "vaddq_n_u8\000vaddq_s16\000vaddq_s32\000vaddq_s8\000vaddq_u16\000vaddq_"
    "u32\000vaddq_u8\000vaddq_x\000vaddq_x_f16\000vaddq_x_f32\000vaddq_x_n_f"
    "16\000vaddq_x_n_f32\000vaddq_x_n_s16\000vaddq_x_n_s32\000vaddq_x_n_s8\000"
    "vaddq_x_n_u16\000vaddq_x_n_u32\000vaddq_x_n_u8\000vaddq_x_s16\000vaddq_"
    "x_s32\000vaddq_x_s8\000vaddq_x_u16\000vaddq_x_u32\000vaddq_x_u8\000vadd"
    "vaq_p\000vaddvaq_p_s16\000vaddvaq_p_s32\000vaddvaq_p_s8\000vaddvaq_p_u1"
    "6\000vaddvaq_p_u32\000vaddvaq_p_u8\000vaddvaq\000vaddvaq_s16\000vaddvaq"
    "_s32\000vaddvaq_s8\000vaddvaq_u16\000vaddvaq_u32\000vaddvaq_u8\000vaddv"
    "q_p\000vaddvq_p_s16\000vaddvq_p_s32\000vaddvq_p_s8\000vaddvq_p_u16\000v"
    "addvq_p_u32\000vaddvq_p_u8\000vaddvq\000vaddvq_s16\000vaddvq_s32\000vad"
    "dvq_s8\000vaddvq_u16\000vaddvq_u32\000vaddvq_u8\000vandq\000vandq_f16\000"
    "vandq_f32\000vandq_m\000vandq_m_f16\000vandq_m_f32\000vandq_m_s16\000va"
    "ndq_m_s32\000vandq_m_s8\000vandq_m_u16\000vandq_m_u32\000vandq_m_u8\000"
    "vandq_s16\000vandq_s32\000vandq_s8\000vandq_u16\000vandq_u32\000vandq_u"
    "8\000vandq_x\000vandq_x_f16\000vandq_x_f32\000vandq_x_s16\000vandq_x_s3"
    "2\000vandq_x_s8\000vandq_x_u16\000vandq_x_u32\000vandq_x_u8\000vbicq\000"
    "vbicq_f16\000vbicq_f32\000vbicq_m\000vbicq_m_f16\000vbicq_m_f32\000vbic"
    "q_m_n\000vbicq_m_n_s16\000vbicq_m_n_s32\000vbicq_m_n_u16\000vbicq_m_n_u"
    "32\000vbicq_m_s16\000vbicq_m_s32\000vbicq_m_s8\000vbicq_m_u16\000vbicq_"
    "m_u32\000vbicq_m_u8\000vbicq_n_s16\000vbicq_n_s32\000vbicq_n_u16\000vbi"
    "cq_n_u32\000vbicq_s16\000vbicq_s32\000vbicq_s8\000vbicq_u16\000vbicq_u3"
    "2\000vbicq_u8\000vbicq_x\000vbicq_x_f16\000vbicq_x_f32\000vbicq_x_s16\000"
    "vbicq_x_s32\000vbicq_x_s8\000vbicq_x_u16\000vbicq_x_u32\000vbicq_x_u8\000"
    "vbrsrq_m\000vbrsrq_m_n_f16\000vbrsrq_m_n_f32\000vbrsrq_m_n_s16\000vbrsr"
    "q_m_n_s32\000vbrsrq_m_n_s8\000vbrsrq_m_n_u16\000vbrsrq_m_n_u32\000vbrsr"
    "q_m_n_u8\000vbrsrq\000vbrsrq_n_f16\000vbrsrq_n_f32\000vbrsrq_n_s16\000v"
    "brsrq_n_s32\000vbrsrq_n_s8\000vbrsrq_n_u16\000vbrsrq_n_u32\000vbrsrq_n_"
    "u8\000vbrsrq_x\000vbrsrq_x_n_f16\000vbrsrq_x_n_f32\000vbrsrq_x_n_s16\000"
    "vbrsrq_x_n_s32\000vbrsrq_x_n_s8\000vbrsrq_x_n_u16\000vbrsrq_x_n_u32\000"
    "vbrsrq_x_n_u8\000vcaddq_rot270\000vcaddq_rot270_f16\000vcaddq_rot270_f3"
    "2\000vcaddq_rot270_m\000vcaddq_rot270_m_f16\000vcaddq_rot270_m_f32\000v"
    "caddq_rot270_m_s16\000vcaddq_rot270_m_s32\000vcaddq_rot270_m_s8\000vcad"
    "dq_rot270_m_u16\000vcaddq_rot270_m_u32\000vcaddq_rot270_m_u8\000vcaddq_"
    "rot270_s16\000vcaddq_rot270_s32\000vcaddq_rot270_s8\000vcaddq_rot270_u1"
    "6\000vcaddq_rot270_u32\000vcaddq_rot270_u8\000vcaddq_rot270_x\000vcaddq"
    "_rot270_x_f16\000vcaddq_rot270_x_f32\000vcaddq_rot270_x_s16\000vcaddq_r"
    "ot270_x_s32\000vcaddq_rot270_x_s8\000vcaddq_rot270_x_u16\000vcaddq_rot2"
    "70_x_u32\000vcaddq_rot270_x_u8\000vcaddq_rot90\000vcaddq_rot90_f16\000v"
    "caddq_rot90_f32\000vcaddq_rot90_m\000vcaddq_rot90_m_f16\000vcaddq_rot90"
    "_m_f32\000vcaddq_rot90_m_s16\000vcaddq_rot90_m_s32\000vcaddq_rot90_m_s8"
    "\000vcaddq_rot90_m_u16\000vcaddq_rot90_m_u32\000vcaddq_rot90_m_u8\000vc"
    "addq_rot90_s16\000vcaddq_rot90_s32\000vcaddq_rot90_s8\000vcaddq_rot90_u"
    "16\000vcaddq_rot90_u32\000vcaddq_rot90_u8\000vcaddq_rot90_x\000vcaddq_r"
    "ot90_x_f16\000vcaddq_rot90_x_f32\000vcaddq_rot90_x_s16\000vcaddq_rot90_"
    "x_s32\000vcaddq_rot90_x_s8\000vcaddq_rot90_x_u16\000vcaddq_rot90_x_u32\000"
    "vcaddq_rot90_x_u8\000vclsq_m\000vclsq_m_s16\000vclsq_m_s32\000vclsq_m_s"
    "8\000vclsq\000vclsq_s16\000vclsq_s32\000vclsq_s8\000vclsq_x\000vclsq_x_"
    "s16\000vclsq_x_s32\000vclsq_x_s8\000vclzq_m\000vclzq_m_s16\000vclzq_m_s"
    "32\000vclzq_m_s8\000vclzq_m_u16\000vclzq_m_u32\000vclzq_m_u8\000vclzq\000"
    "vclzq_s16\000vclzq_s32\000vclzq_s8\000vclzq_u16\000vclzq_u32\000vclzq_u"
    "8\000vclzq_x\000vclzq_x_s16\000vclzq_x_s32\000vclzq_x_s8\000vclzq_x_u16"
    "\000vclzq_x_u32\000vclzq_x_u8\000vcmlaq\000vcmlaq_f16\000vcmlaq_f32\000"
    "vcmlaq_m\000vcmlaq_m_f16\000vcmlaq_m_f32\000vcmlaq_rot180\000vcmlaq_rot"
    "180_f16\000vcmlaq_rot180_f32\000vcmlaq_rot180_m\000vcmlaq_rot180_m_f16\000"
    "vcmlaq_rot180_m_f32\000vcmlaq_rot270\000vcmlaq_rot270_f16\000vcmlaq_rot"
    "270_f32\000vcmlaq_rot270_m\000vcmlaq_rot270_m_f16\000vcmlaq_rot270_m_f3"
    "2\000vcmlaq_rot90\000vcmlaq_rot90_f16\000vcmlaq_rot90_f32\000vcmlaq_rot"
    "90_m\000vcmlaq_rot90_m_f16\000vcmlaq_rot90_m_f32\000vcmpcsq_m\000vcmpcs"
    "q_m_n_u16\000vcmpcsq_m_n_u32\000vcmpcsq_m_n_u8\000vcmpcsq_m_u16\000vcmp"
    "csq_m_u32\000vcmpcsq_m_u8\000vcmpcsq\000vcmpcsq_n_u16\000vcmpcsq_n_u32\000"
    "vcmpcsq_n_u8\000vcmpcsq_u16\000vcmpcsq_u32\000vcmpcsq_u8\000vcmpeqq\000"
    "vcmpeqq_f16\000vcmpeqq_f32\000vcmpeqq_m\000vcmpeqq_m_f16\000vcmpeqq_m_f"
    "32\000vcmpeqq_m_n_f16\000vcmpeqq_m_n_f32\000vcmpeqq_m_n_s16\000vcmpeqq_"
    "m_n_s32\000vcmpeqq_m_n_s8\000vcmpeqq_m_n_u16\000vcmpeqq_m_n_u32\000vcmp"
    "eqq_m_n_u8\000vcmpeqq_m_s16\000vcmpeqq_m_s32\000vcmpeqq_m_s8\000vcmpeqq"
    "_m_u16\000vcmpeqq_m_u32\000vcmpeqq_m_u8\000vcmpeqq_n_f16\000vcmpeqq_n_f"
    "32\000vcmpeqq_n_s16\000vcmpeqq_n_s32\000vcmpeqq_n_s8\000vcmpeqq_n_u16\000"
    "vcmpeqq_n_u32\000vcmpeqq_n_u8\000vcmpeqq_s16\000vcmpeqq_s32\000vcmpeqq_"
    "s8\000vcmpeqq_u16\000vcmpeqq_u32\000vcmpeqq_u8\000vcmpgeq\000vcmpgeq_f1"
    "6\000vcmpgeq_f32\000vcmpgeq_m\000vcmpgeq_m_f16\000vcmpgeq_m_f32\000vcmp"
    "geq_m_n_f16\000vcmpgeq_m_n_f32\000vcmpgeq_m_n_s16\000vcmpgeq_m_n_s32\000"
    "vcmpgeq_m_n_s8\000vcmpgeq_m_s16\000vcmpgeq_m_s32\000vcmpgeq_m_s8\000vcm"
    "pgeq_n_f16\000vcmpgeq_n_f32\000vcmpgeq_n_s16\000vcmpgeq_n_s32\000vcmpge"
    "q_n_s8\000vcmpgeq_s16\000vcmpgeq_s32\000vcmpgeq_s8\000vcmpgtq\000vcmpgt"
    "q_f16\000vcmpgtq_f32\000vcmpgtq_m\000vcmpgtq_m_f16\000vcmpgtq_m_f32\000"
    "vcmpgtq_m_n_f16\000vcmpgtq_m_n_f32\000vcmpgtq_m_n_s16\000vcmpgtq_m_n_s3"
    "2\000vcmpgtq_m_n_s8\000vcmpgtq_m_s16\000vcmpgtq_m_s32\000vcmpgtq_m_s8\000"
    "vcmpgtq_n_f16\000vcmpgtq_n_f32\000vcmpgtq_n_s16\000vcmpgtq_n_s32\000vcm"
    "pgtq_n_s8\000vcmpgtq_s16\000vcmpgtq_s32\000vcmpgtq_s8\000vcmphiq_m\000v"
    "cmphiq_m_n_u16\000vcmphiq_m_n_u32\000vcmphiq_m_n_u8\000vcmphiq_m_u16\000"
    "vcmphiq_m_u32\000vcmphiq_m_u8\000vcmphiq\000vcmphiq_n_u16\000vcmphiq_n_"
    "u32\000vcmphiq_n_u8\000vcmphiq_u16\000vcmphiq_u32\000vcmphiq_u8\000vcmp"
    "leq\000vcmpleq_f16\000vcmpleq_f32\000vcmpleq_m\000vcmpleq_m_f16\000vcmp"
    "leq_m_f32\000vcmpleq_m_n_f16\000vcmpleq_m_n_f32\000vcmpleq_m_n_s16\000v"
    "cmpleq_m_n_s32\000vcmpleq_m_n_s8\000vcmpleq_m_s16\000vcmpleq_m_s32\000v"
    "cmpleq_m_s8\000vcmpleq_n_f16\000vcmpleq_n_f32\000vcmpleq_n_s16\000vcmpl"
    "eq_n_s32\000vcmpleq_n_s8\000vcmpleq_s16\000vcmpleq_s32\000vcmpleq_s8\000"
    "vcmpltq\000vcmpltq_f16\000vcmpltq_f32\000vcmpltq_m\000vcmpltq_m_f16\000"
    "vcmpltq_m_f32\000vcmpltq_m_n_f16\000vcmpltq_m_n_f32\000vcmpltq_m_n_s16\000"
    "vcmpltq_m_n_s32\000vcmpltq_m_n_s8\000vcmpltq_m_s16\000vcmpltq_m_s32\000"
    "vcmpltq_m_s8\000vcmpltq_n_f16\000vcmpltq_n_f32\000vcmpltq_n_s16\000vcmp"
    "ltq_n_s32\000vcmpltq_n_s8\000vcmpltq_s16\000vcmpltq_s32\000vcmpltq_s8\000"
    "vcmpneq\000vcmpneq_f16\000vcmpneq_f32\000vcmpneq_m\000vcmpneq_m_f16\000"
    "vcmpneq_m_f32\000vcmpneq_m_n_f16\000vcmpneq_m_n_f32\000vcmpneq_m_n_s16\000"
    "vcmpneq_m_n_s32\000vcmpneq_m_n_s8\000vcmpneq_m_n_u16\000vcmpneq_m_n_u32"
    "\000vcmpneq_m_n_u8\000vcmpneq_m_s16\000vcmpneq_m_s32\000vcmpneq_m_s8\000"
    "vcmpneq_m_u16\000vcmpneq_m_u32\000vcmpneq_m_u8\000vcmpneq_n_f16\000vcmp"
    "neq_n_f32\000vcmpneq_n_s16\000vcmpneq_n_s32\000vcmpneq_n_s8\000vcmpneq_"
    "n_u16\000vcmpneq_n_u32\000vcmpneq_n_u8\000vcmpneq_s16\000vcmpneq_s32\000"
    "vcmpneq_s8\000vcmpneq_u16\000vcmpneq_u32\000vcmpneq_u8\000vcmulq\000vcm"
    "ulq_f16\000vcmulq_f32\000vcmulq_m\000vcmulq_m_f16\000vcmulq_m_f32\000vc"
    "mulq_rot180\000vcmulq_rot180_f16\000vcmulq_rot180_f32\000vcmulq_rot180_"
    "m\000vcmulq_rot180_m_f16\000vcmulq_rot180_m_f32\000vcmulq_rot180_x\000v"
    "cmulq_rot180_x_f16\000vcmulq_rot180_x_f32\000vcmulq_rot270\000vcmulq_ro"
    "t270_f16\000vcmulq_rot270_f32\000vcmulq_rot270_m\000vcmulq_rot270_m_f16"
    "\000vcmulq_rot270_m_f32\000vcmulq_rot270_x\000vcmulq_rot270_x_f16\000vc"
    "mulq_rot270_x_f32\000vcmulq_rot90\000vcmulq_rot90_f16\000vcmulq_rot90_f"
    "32\000vcmulq_rot90_m\000vcmulq_rot90_m_f16\000vcmulq_rot90_m_f32\000vcm"
    "ulq_rot90_x\000vcmulq_rot90_x_f16\000vcmulq_rot90_x_f32\000vcmulq_x\000"
    "vcmulq_x_f16\000vcmulq_x_f32\000vcreateq_f16\000vcreateq_f32\000vcreate"
    "q_s16\000vcreateq_s32\000vcreateq_s64\000vcreateq_s8\000vcreateq_u16\000"
    "vcreateq_u32\000vcreateq_u64\000vcreateq_u8\000vctp16q\000vctp16q_m\000"
    "vctp32q\000vctp32q_m\000vctp64q\000vctp64q_m\000vctp8q\000vctp8q_m\000v"
    "cvtaq_m\000vcvtaq_m_s16_f16\000vcvtaq_m_s32_f32\000vcvtaq_m_u16_f16\000"
    "vcvtaq_m_u32_f32\000vcvtaq_s16_f16\000vcvtaq_s32_f32\000vcvtaq_u16_f16\000"
    "vcvtaq_u32_f32\000vcvtaq_x_s16_f16\000vcvtaq_x_s32_f32\000vcvtaq_x_u16_"
    "f16\000vcvtaq_x_u32_f32\000vcvtbq_f16_f32\000vcvtbq_f32_f16\000vcvtbq_m"
    "_f16_f32\000vcvtbq_m_f32_f16\000vcvtbq_x_f32_f16\000vcvtmq_m\000vcvtmq_"
    "m_s16_f16\000vcvtmq_m_s32_f32\000vcvtmq_m_u16_f16\000vcvtmq_m_u32_f32\000"
    "vcvtmq_s16_f16\000vcvtmq_s32_f32\000vcvtmq_u16_f16\000vcvtmq_u32_f32\000"
    "vcvtmq_x_s16_f16\000vcvtmq_x_s32_f32\000vcvtmq_x_u16_f16\000vcvtmq_x_u3"
    "2_f32\000vcvtnq_m\000vcvtnq_m_s16_f16\000vcvtnq_m_s32_f32\000vcvtnq_m_u"
    "16_f16\000vcvtnq_m_u32_f32\000vcvtnq_s16_f16\000vcvtnq_s32_f32\000vcvtn"
    "q_u16_f16\000vcvtnq_u32_f32\000vcvtnq_x_s16_f16\000vcvtnq_x_s32_f32\000"
    "vcvtnq_x_u16_f16\000vcvtnq_x_u32_f32\000vcvtpq_m\000vcvtpq_m_s16_f16\000"
    "vcvtpq_m_s32_f32\000vcvtpq_m_u16_f16\000vcvtpq_m_u32_f32\000vcvtpq_s16_"
    "f16\000vcvtpq_s32_f32\000vcvtpq_u16_f16\000vcvtpq_u32_f32\000vcvtpq_x_s"
    "16_f16\000vcvtpq_x_s32_f32\000vcvtpq_x_u16_f16\000vcvtpq_x_u32_f32\000v"
    "cvtq\000vcvtq_f16_s16\000vcvtq_f16_u16\000vcvtq_f32_s32\000vcvtq_f32_u3"
    "2\000vcvtq_m\000vcvtq_m_f16_s16\000vcvtq_m_f16_u16\000vcvtq_m_f32_s32\000"
    "vcvtq_m_f32_u32\000vcvtq_m_n\000vcvtq_m_n_f16_s16\000vcvtq_m_n_f16_u16\000"
    "vcvtq_m_n_f32_s32\000vcvtq_m_n_f32_u32\000vcvtq_m_n_s16_f16\000vcvtq_m_"
    "n_s32_f32\000vcvtq_m_n_u16_f16\000vcvtq_m_n_u32_f32\000vcvtq_m_s16_f16\000"
    "vcvtq_m_s32_f32\000vcvtq_m_u16_f16\000vcvtq_m_u32_f32\000vcvtq_n\000vcv"
    "tq_n_f16_s16\000vcvtq_n_f16_u16\000vcvtq_n_f32_s32\000vcvtq_n_f32_u32\000"
    "vcvtq_n_s16_f16\000vcvtq_n_s32_f32\000vcvtq_n_u16_f16\000vcvtq_n_u32_f3"
    "2\000vcvtq_s16_f16\000vcvtq_s32_f32\000vcvtq_u16_f16\000vcvtq_u32_f32\000"
    "vcvtq_x\000vcvtq_x_f16_s16\000vcvtq_x_f16_u16\000vcvtq_x_f32_s32\000vcv"
    "tq_x_f32_u32\000vcvtq_x_n\000vcvtq_x_n_f16_s16\000vcvtq_x_n_f16_u16\000"
    "vcvtq_x_n_f32_s32\000vcvtq_x_n_f32_u32\000vcvtq_x_n_s16_f16\000vcvtq_x_"
    "n_s32_f32\000vcvtq_x_n_u16_f16\000vcvtq_x_n_u32_f32\000vcvtq_x_s16_f16\000"
    "vcvtq_x_s32_f32\000vcvtq_x_u16_f16\000vcvtq_x_u32_f32\000vcvttq_f16_f32"
    "\000vcvttq_f32_f16\000vcvttq_m_f16_f32\000vcvttq_m_f32_f16\000vcvttq_x_"
    "f32_f16\000vddupq_m\000vddupq_m_n_u16\000vddupq_m_n_u32\000vddupq_m_n_u"
    "8\000vddupq_m_wb_u16\000vddupq_m_wb_u32\000vddupq_m_wb_u8\000vddupq_u16"
    "\000vddupq_n_u16\000vddupq_u32\000vddupq_n_u32\000vddupq_u8\000vddupq_n"
    "_u8\000vddupq_wb_u16\000vddupq_wb_u32\000vddupq_wb_u8\000vddupq_x_u16\000"
    "vddupq_x_n_u16\000vddupq_x_u32\000vddupq_x_n_u32\000vddupq_x_u8\000vddu"
    "pq_x_n_u8\000vddupq_x_wb_u16\000vddupq_x_wb_u32\000vddupq_x_wb_u8\000vd"
    "upq_m\000vdupq_m_n_f16\000vdupq_m_n_f32\000vdupq_m_n_s16\000vdupq_m_n_s"
    "32\000vdupq_m_n_s8\000vdupq_m_n_u16\000vdupq_m_n_u32\000vdupq_m_n_u8\000"
    "vdupq_n_f16\000vdupq_n_f32\000vdupq_n_s16\000vdupq_n_s32\000vdupq_n_s8\000"
    "vdupq_n_u16\000vdupq_n_u32\000vdupq_n_u8\000vdupq_x_n_f16\000vdupq_x_n_"
    "f32\000vdupq_x_n_s16\000vdupq_x_n_s32\000vdupq_x_n_s8\000vdupq_x_n_u16\000"
    "vdupq_x_n_u32\000vdupq_x_n_u8\000vdwdupq_m\000vdwdupq_m_n_u16\000vdwdup"
    "q_m_n_u32\000vdwdupq_m_n_u8\000vdwdupq_m_wb_u16\000vdwdupq_m_wb_u32\000"
    "vdwdupq_m_wb_u8\000vdwdupq_u16\000vdwdupq_n_u16\000vdwdupq_u32\000vdwdu"
    "pq_n_u32\000vdwdupq_u8\000vdwdupq_n_u8\000vdwdupq_wb_u16\000vdwdupq_wb_"
    "u32\000vdwdupq_wb_u8\000vdwdupq_x_u16\000vdwdupq_x_n_u16\000vdwdupq_x_u"
    "32\000vdwdupq_x_n_u32\000vdwdupq_x_u8\000vdwdupq_x_n_u8\000vdwdupq_x_wb"
    "_u16\000vdwdupq_x_wb_u32\000vdwdupq_x_wb_u8\000veorq\000veorq_f16\000ve"
    "orq_f32\000veorq_m\000veorq_m_f16\000veorq_m_f32\000veorq_m_s16\000veor"
    "q_m_s32\000veorq_m_s8\000veorq_m_u16\000veorq_m_u32\000veorq_m_u8\000ve"
    "orq_s16\000veorq_s32\000veorq_s8\000veorq_u16\000veorq_u32\000veorq_u8\000"
    "veorq_x\000veorq_x_f16\000veorq_x_f32\000veorq_x_s16\000veorq_x_s32\000"
    "veorq_x_s8\000veorq_x_u16\000veorq_x_u32\000veorq_x_u8\000vfmaq\000vfma"
    "q_f16\000vfmaq_f32\000vfmaq_m\000vfmaq_m_f16\000vfmaq_m_f32\000vfmaq_m_"
    "n_f16\000vfmaq_m_n_f32\000vfmaq_n_f16\000vfmaq_n_f32\000vfmasq_m\000vfm"
    "asq_m_n_f16\000vfmasq_m_n_f32\000vfmasq\000vfmasq_n_f16\000vfmasq_n_f32"
    "\000vfmsq\000vfmsq_f16\000vfmsq_f32\000vfmsq_m\000vfmsq_m_f16\000vfmsq_"
    "m_f32\000vgetq_lane\000vgetq_lane_f16\000vgetq_lane_f32\000vgetq_lane_s"
    "16\000vgetq_lane_s32\000vgetq_lane_s64\000vgetq_lane_s8\000vgetq_lane_u"
    "16\000vgetq_lane_u32\000vgetq_lane_u64\000vgetq_lane_u8\000vhaddq_m\000"
    "vhaddq_m_n_s16\000vhaddq_m_n_s32\000vhaddq_m_n_s8\000vhaddq_m_n_u16\000"
    "vhaddq_m_n_u32\000vhaddq_m_n_u8\000vhaddq_m_s16\000vhaddq_m_s32\000vhad"
    "dq_m_s8\000vhaddq_m_u16\000vhaddq_m_u32\000vhaddq_m_u8\000vhaddq\000vha"
    "ddq_n_s16\000vhaddq_n_s32\000vhaddq_n_s8\000vhaddq_n_u16\000vhaddq_n_u3"
    "2\000vhaddq_n_u8\000vhaddq_s16\000vhaddq_s32\000vhaddq_s8\000vhaddq_u16"
    "\000vhaddq_u32\000vhaddq_u8\000vhaddq_x\000vhaddq_x_n_s16\000vhaddq_x_n"
    "_s32\000vhaddq_x_n_s8\000vhaddq_x_n_u16\000vhaddq_x_n_u32\000vhaddq_x_n"
    "_u8\000vhaddq_x_s16\000vhaddq_x_s32\000vhaddq_x_s8\000vhaddq_x_u16\000v"
    "haddq_x_u32\000vhaddq_x_u8\000vhcaddq_rot270_m\000vhcaddq_rot270_m_s16\000"
    "vhcaddq_rot270_m_s32\000vhcaddq_rot270_m_s8\000vhcaddq_rot270\000vhcadd"
    "q_rot270_s16\000vhcaddq_rot270_s32\000vhcaddq_rot270_s8\000vhcaddq_rot2"
    "70_x\000vhcaddq_rot270_x_s16\000vhcaddq_rot270_x_s32\000vhcaddq_rot270_"
    "x_s8\000vhcaddq_rot90_m\000vhcaddq_rot90_m_s16\000vhcaddq_rot90_m_s32\000"
    "vhcaddq_rot90_m_s8\000vhcaddq_rot90\000vhcaddq_rot90_s16\000vhcaddq_rot"
    "90_s32\000vhcaddq_rot90_s8\000vhcaddq_rot90_x\000vhcaddq_rot90_x_s16\000"
    "vhcaddq_rot90_x_s32\000vhcaddq_rot90_x_s8\000vhsubq_m\000vhsubq_m_n_s16"
    "\000vhsubq_m_n_s32\000vhsubq_m_n_s8\000vhsubq_m_n_u16\000vhsubq_m_n_u32"
    "\000vhsubq_m_n_u8\000vhsubq_m_s16\000vhsubq_m_s32\000vhsubq_m_s8\000vhs"
    "ubq_m_u16\000vhsubq_m_u32\000vhsubq_m_u8\000vhsubq\000vhsubq_n_s16\000v"
    "hsubq_n_s32\000vhsubq_n_s8\000vhsubq_n_u16\000vhsubq_n_u32\000vhsubq_n_"
    "u8\000vhsubq_s16\000vhsubq_s32\000vhsubq_s8\000vhsubq_u16\000vhsubq_u32"
    "\000vhsubq_u8\000vhsubq_x\000vhsubq_x_n_s16\000vhsubq_x_n_s32\000vhsubq"
    "_x_n_s8\000vhsubq_x_n_u16\000vhsubq_x_n_u32\000vhsubq_x_n_u8\000vhsubq_"
    "x_s16\000vhsubq_x_s32\000vhsubq_x_s8\000vhsubq_x_u16\000vhsubq_x_u32\000"
    "vhsubq_x_u8\000vidupq_m\000vidupq_m_n_u16\000vidupq_m_n_u32\000vidupq_m"
    "_n_u8\000vidupq_m_wb_u16\000vidupq_m_wb_u32\000vidupq_m_wb_u8\000vidupq"
    "_u16\000vidupq_n_u16\000vidupq_u32\000vidupq_n_u32\000vidupq_u8\000vidu"
    "pq_n_u8\000vidupq_wb_u16\000vidupq_wb_u32\000vidupq_wb_u8\000vidupq_x_u"
    "16\000vidupq_x_n_u16\000vidupq_x_u32\000vidupq_x_n_u32\000vidupq_x_u8\000"
    "vidupq_x_n_u8\000vidupq_x_wb_u16\000vidupq_x_wb_u32\000vidupq_x_wb_u8\000"
    "viwdupq_m\000viwdupq_m_n_u16\000viwdupq_m_n_u32\000viwdupq_m_n_u8\000vi"
    "wdupq_m_wb_u16\000viwdupq_m_wb_u32\000viwdupq_m_wb_u8\000viwdupq_u16\000"
    "viwdupq_n_u16\000viwdupq_u32\000viwdupq_n_u32\000viwdupq_u8\000viwdupq_"
    "n_u8\000viwdupq_wb_u16\000viwdupq_wb_u32\000viwdupq_wb_u8\000viwdupq_x_"
    "u16\000viwdupq_x_n_u16\000viwdupq_x_u32\000viwdupq_x_n_u32\000viwdupq_x"
    "_u8\000viwdupq_x_n_u8\000viwdupq_x_wb_u16\000viwdupq_x_wb_u32\000viwdup"
    "q_x_wb_u8\000vld1q\000vld1q_f16\000vld1q_f32\000vld1q_s16\000vld1q_s32\000"
    "vld1q_s8\000vld1q_u16\000vld1q_u32\000vld1q_u8\000vld1q_z\000vld1q_z_f1"
    "6\000vld1q_z_f32\000vld1q_z_s16\000vld1q_z_s32\000vld1q_z_s8\000vld1q_z"
    "_u16\000vld1q_z_u32\000vld1q_z_u8\000vld2q\000vld2q_f16\000vld2q_f32\000"
    "vld2q_s16\000vld2q_s32\000vld2q_s8\000vld2q_u16\000vld2q_u32\000vld2q_u"
    "8\000vld4q\000vld4q_f16\000vld4q_f32\000vld4q_s16\000vld4q_s32\000vld4q"
    "_s8\000vld4q_u16\000vld4q_u32\000vld4q_u8\000vldrbq_gather_offset\000vl"
    "drbq_gather_offset_s16\000vldrbq_gather_offset_s32\000vldrbq_gather_off"
    "set_s8\000vldrbq_gather_offset_u16\000vldrbq_gather_offset_u32\000vldrb"
    "q_gather_offset_u8\000vldrbq_gather_offset_z\000vldrbq_gather_offset_z_"
    "s16\000vldrbq_gather_offset_z_s32\000vldrbq_gather_offset_z_s8\000vldrb"
    "q_gather_offset_z_u16\000vldrbq_gather_offset_z_u32\000vldrbq_gather_of"
    "fset_z_u8\000vldrbq_s16\000vldrbq_s32\000vldrbq_s8\000vldrbq_u16\000vld"
    "rbq_u32\000vldrbq_u8\000vldrbq_z_s16\000vldrbq_z_s32\000vldrbq_z_s8\000"
    "vldrbq_z_u16\000vldrbq_z_u32\000vldrbq_z_u8\000vldrdq_gather_base_s64\000"
    "vldrdq_gather_base_u64\000vldrdq_gather_base_wb_s64\000vldrdq_gather_ba"
    "se_wb_u64\000vldrdq_gather_base_wb_z_s64\000vldrdq_gather_base_wb_z_u64"
    "\000vldrdq_gather_base_z_s64\000vldrdq_gather_base_z_u64\000vldrdq_gath"
    "er_offset\000vldrdq_gather_offset_s64\000vldrdq_gather_offset_u64\000vl"
    "drdq_gather_offset_z\000vldrdq_gather_offset_z_s64\000vldrdq_gather_off"
    "set_z_u64\000vldrdq_gather_shifted_offset\000vldrdq_gather_shifted_offs"
    "et_s64\000vldrdq_gather_shifted_offset_u64\000vldrdq_gather_shifted_off"
    "set_z\000vldrdq_gather_shifted_offset_z_s64\000vldrdq_gather_shifted_of"
    "fset_z_u64\000vldrhq_f16\000vldrhq_gather_offset\000vldrhq_gather_offse"
    "t_f16\000vldrhq_gather_offset_s16\000vldrhq_gather_offset_s32\000vldrhq"
    "_gather_offset_u16\000vldrhq_gather_offset_u32\000vldrhq_gather_offset_"
    "z\000vldrhq_gather_offset_z_f16\000vldrhq_gather_offset_z_s16\000vldrhq"
    "_gather_offset_z_s32\000vldrhq_gather_offset_z_u16\000vldrhq_gather_off"
    "set_z_u32\000vldrhq_gather_shifted_offset\000vldrhq_gather_shifted_offs"
    "et_f16\000vldrhq_gather_shifted_offset_s16\000vldrhq_gather_shifted_off"
    "set_s32\000vldrhq_gather_shifted_offset_u16\000vldrhq_gather_shifted_of"
    "fset_u32\000vldrhq_gather_shifted_offset_z\000vldrhq_gather_shifted_off"
    "set_z_f16\000vldrhq_gather_shifted_offset_z_s16\000vldrhq_gather_shifte"
    "d_offset_z_s32\000vldrhq_gather_shifted_offset_z_u16\000vldrhq_gather_s"
    "hifted_offset_z_u32\000vldrhq_s16\000vldrhq_s32\000vldrhq_u16\000vldrhq"
    "_u32\000vldrhq_z_f16\000vldrhq_z_s16\000vldrhq_z_s32\000vldrhq_z_u16\000"
    "vldrhq_z_u32\000vldrwq_f32\000vldrwq_gather_base_f32\000vldrwq_gather_b"
    "ase_s32\000vldrwq_gather_base_u32\000vldrwq_gather_base_wb_f32\000vldrw"
    "q_gather_base_wb_s32\000vldrwq_gather_base_wb_u32\000vldrwq_gather_base"
    "_wb_z_f32\000vldrwq_gather_base_wb_z_s32\000vldrwq_gather_base_wb_z_u32"
    "\000vldrwq_gather_base_z_f32\000vldrwq_gather_base_z_s32\000vldrwq_gath"
    "er_base_z_u32\000vldrwq_gather_offset\000vldrwq_gather_offset_f32\000vl"
    "drwq_gather_offset_s32\000vldrwq_gather_offset_u32\000vldrwq_gather_off"
    "set_z\000vldrwq_gather_offset_z_f32\000vldrwq_gather_offset_z_s32\000vl"
    "drwq_gather_offset_z_u32\000vldrwq_gather_shifted_offset\000vldrwq_gath"
    "er_shifted_offset_f32\000vldrwq_gather_shifted_offset_s32\000vldrwq_gat"
    "her_shifted_offset_u32\000vldrwq_gather_shifted_offset_z\000vldrwq_gath"
    "er_shifted_offset_z_f32\000vldrwq_gather_shifted_offset_z_s32\000vldrwq"
    "_gather_shifted_offset_z_u32\000vldrwq_s32\000vldrwq_u32\000vldrwq_z_f3"
    "2\000vldrwq_z_s32\000vldrwq_z_u32\000vmaxaq_m\000vmaxaq_m_s16\000vmaxaq"
    "_m_s32\000vmaxaq_m_s8\000vmaxaq\000vmaxaq_s16\000vmaxaq_s32\000vmaxaq_s"
    "8\000vmaxavq_p\000vmaxavq_p_s16\000vmaxavq_p_s32\000vmaxavq_p_s8\000vma"
    "xavq\000vmaxavq_s16\000vmaxavq_s32\000vmaxavq_s8\000vmaxnmaq\000vmaxnma"
    "q_f16\000vmaxnmaq_f32\000vmaxnmaq_m\000vmaxnmaq_m_f16\000vmaxnmaq_m_f32"
    "\000vmaxnmavq\000vmaxnmavq_f16\000vmaxnmavq_f32\000vmaxnmavq_p\000vmaxn"
    "mavq_p_f16\000vmaxnmavq_p_f32\000vmaxnmq\000vmaxnmq_f16\000vmaxnmq_f32\000"
    "vmaxnmq_m\000vmaxnmq_m_f16\000vmaxnmq_m_f32\000vmaxnmq_x\000vmaxnmq_x_f"
    "16\000vmaxnmq_x_f32\000vmaxnmvq\000vmaxnmvq_f16\000vmaxnmvq_f32\000vmax"
    "nmvq_p\000vmaxnmvq_p_f16\000vmaxnmvq_p_f32\000vmaxq_m\000vmaxq_m_s16\000"
    "vmaxq_m_s32\000vmaxq_m_s8\000vmaxq_m_u16\000vmaxq_m_u32\000vmaxq_m_u8\000"
    "vmaxq\000vmaxq_s16\000vmaxq_s32\000vmaxq_s8\000vmaxq_u16\000vmaxq_u32\000"
    "vmaxq_u8\000vmaxq_x\000vmaxq_x_s16\000vmaxq_x_s32\000vmaxq_x_s8\000vmax"
    "q_x_u16\000vmaxq_x_u32\000vmaxq_x_u8\000vmaxvq_p\000vmaxvq_p_s16\000vma"
    "xvq_p_s32\000vmaxvq_p_s8\000vmaxvq_p_u16\000vmaxvq_p_u32\000vmaxvq_p_u8"
    "\000vmaxvq\000vmaxvq_s16\000vmaxvq_s32\000vmaxvq_s8\000vmaxvq_u16\000vm"
    "axvq_u32\000vmaxvq_u8\000vminaq_m\000vminaq_m_s16\000vminaq_m_s32\000vm"
    "inaq_m_s8\000vminaq\000vminaq_s16\000vminaq_s32\000vminaq_s8\000vminavq"
    "_p\000vminavq_p_s16\000vminavq_p_s32\000vminavq_p_s8\000vminavq\000vmin"
    "avq_s16\000vminavq_s32\000vminavq_s8\000vminnmaq\000vminnmaq_f16\000vmi"
    "nnmaq_f32\000vminnmaq_m\000vminnmaq_m_f16\000vminnmaq_m_f32\000vminnmav"
    "q\000vminnmavq_f16\000vminnmavq_f32\000vminnmavq_p\000vminnmavq_p_f16\000"
    "vminnmavq_p_f32\000vminnmq\000vminnmq_f16\000vminnmq_f32\000vminnmq_m\000"
    "vminnmq_m_f16\000vminnmq_m_f32\000vminnmq_x\000vminnmq_x_f16\000vminnmq"
    "_x_f32\000vminnmvq\000vminnmvq_f16\000vminnmvq_f32\000vminnmvq_p\000vmi"
    "nnmvq_p_f16\000vminnmvq_p_f32\000vminq_m\000vminq_m_s16\000vminq_m_s32\000"
    "vminq_m_s8\000vminq_m_u16\000vminq_m_u32\000vminq_m_u8\000vminq\000vmin"
    "q_s16\000vminq_s32\000vminq_s8\000vminq_u16\000vminq_u32\000vminq_u8\000"
    "vminq_x\000vminq_x_s16\000vminq_x_s32\000vminq_x_s8\000vminq_x_u16\000v"
    "minq_x_u32\000vminq_x_u8\000vminvq_p\000vminvq_p_s16\000vminvq_p_s32\000"
    "vminvq_p_s8\000vminvq_p_u16\000vminvq_p_u32\000vminvq_p_u8\000vminvq\000"
    "vminvq_s16\000vminvq_s32\000vminvq_s8\000vminvq_u16\000vminvq_u32\000vm"
    "invq_u8\000vmladavaq_p\000vmladavaq_p_s16\000vmladavaq_p_s32\000vmladav"
    "aq_p_s8\000vmladavaq_p_u16\000vmladavaq_p_u32\000vmladavaq_p_u8\000vmla"
    "davaq\000vmladavaq_s16\000vmladavaq_s32\000vmladavaq_s8\000vmladavaq_u1"
    "6\000vmladavaq_u32\000vmladavaq_u8\000vmladavaxq_p\000vmladavaxq_p_s16\000"
    "vmladavaxq_p_s32\000vmladavaxq_p_s8\000vmladavaxq\000vmladavaxq_s16\000"
    "vmladavaxq_s32\000vmladavaxq_s8\000vmladavq_p\000vmladavq_p_s16\000vmla"
    "davq_p_s32\000vmladavq_p_s8\000vmladavq_p_u16\000vmladavq_p_u32\000vmla"
    "davq_p_u8\000vmladavq\000vmladavq_s16\000vmladavq_s32\000vmladavq_s8\000"
    "vmladavq_u16\000vmladavq_u32\000vmladavq_u8\000vmladavxq_p\000vmladavxq"
    "_p_s16\000vmladavxq_p_s32\000vmladavxq_p_s8\000vmladavxq\000vmladavxq_s"
    "16\000vmladavxq_s32\000vmladavxq_s8\000vmlaldavaq_p\000vmlaldavaq_p_s16"
    "\000vmlaldavaq_p_s32\000vmlaldavaq_p_u16\000vmlaldavaq_p_u32\000vmlalda"
    "vaq\000vmlaldavaq_s16\000vmlaldavaq_s32\000vmlaldavaq_u16\000vmlaldavaq"
    "_u32\000vmlaldavaxq_p\000vmlaldavaxq_p_s16\000vmlaldavaxq_p_s32\000vmla"
    "ldavaxq\000vmlaldavaxq_s16\000vmlaldavaxq_s32\000vmlaldavq_p\000vmlalda"
    "vq_p_s16\000vmlaldavq_p_s32\000vmlaldavq_p_u16\000vmlaldavq_p_u32\000vm"
    "laldavq\000vmlaldavq_s16\000vmlaldavq_s32\000vmlaldavq_u16\000vmlaldavq"
    "_u32\000vmlaldavxq_p\000vmlaldavxq_p_s16\000vmlaldavxq_p_s32\000vmlalda"
    "vxq\000vmlaldavxq_s16\000vmlaldavxq_s32\000vmlaq_m\000vmlaq_m_n_s16\000"
    "vmlaq_m_n_s32\000vmlaq_m_n_s8\000vmlaq_m_n_u16\000vmlaq_m_n_u32\000vmla"
    "q_m_n_u8\000vmlaq\000vmlaq_n_s16\000vmlaq_n_s32\000vmlaq_n_s8\000vmlaq_"
    "n_u16\000vmlaq_n_u32\000vmlaq_n_u8\000vmlasq_m\000vmlasq_m_n_s16\000vml"
    "asq_m_n_s32\000vmlasq_m_n_s8\000vmlasq_m_n_u16\000vmlasq_m_n_u32\000vml"
    "asq_m_n_u8\000vmlasq\000vmlasq_n_s16\000vmlasq_n_s32\000vmlasq_n_s8\000"
    "vmlasq_n_u16\000vmlasq_n_u32\000vmlasq_n_u8\000vmlsdavaq_p\000vmlsdavaq"
    "_p_s16\000vmlsdavaq_p_s32\000vmlsdavaq_p_s8\000vmlsdavaq\000vmlsdavaq_s"
    "16\000vmlsdavaq_s32\000vmlsdavaq_s8\000vmlsdavaxq_p\000vmlsdavaxq_p_s16"
    "\000vmlsdavaxq_p_s32\000vmlsdavaxq_p_s8\000vmlsdavaxq\000vmlsdavaxq_s16"
    "\000vmlsdavaxq_s32\000vmlsdavaxq_s8\000vmlsdavq_p\000vmlsdavq_p_s16\000"
    "vmlsdavq_p_s32\000vmlsdavq_p_s8\000vmlsdavq\000vmlsdavq_s16\000vmlsdavq"
    "_s32\000vmlsdavq_s8\000vmlsdavxq_p\000vmlsdavxq_p_s16\000vmlsdavxq_p_s3"
    "2\000vmlsdavxq_p_s8\000vmlsdavxq\000vmlsdavxq_s16\000vmlsdavxq_s32\000v"
    "mlsdavxq_s8\000vmlsldavaq_p\000vmlsldavaq_p_s16\000vmlsldavaq_p_s32\000"
    "vmlsldavaq\000vmlsldavaq_s16\000vmlsldavaq_s32\000vmlsldavaxq_p\000vmls"
    "ldavaxq_p_s16\000vmlsldavaxq_p_s32\000vmlsldavaxq\000vmlsldavaxq_s16\000"
    "vmlsldavaxq_s32\000vmlsldavq_p\000vmlsldavq_p_s16\000vmlsldavq_p_s32\000"
    "vmlsldavq\000vmlsldavq_s16\000vmlsldavq_s32\000vmlsldavxq_p\000vmlsldav"
    "xq_p_s16\000vmlsldavxq_p_s32\000vmlsldavxq\000vmlsldavxq_s16\000vmlslda"
    "vxq_s32\000vmovlbq_m\000vmovlbq_m_s16\000vmovlbq_m_s8\000vmovlbq_m_u16\000"
    "vmovlbq_m_u8\000vmovlbq\000vmovlbq_s16\000vmovlbq_s8\000vmovlbq_u16\000"
    "vmovlbq_u8\000vmovlbq_x\000vmovlbq_x_s16\000vmovlbq_x_s8\000vmovlbq_x_u"
    "16\000vmovlbq_x_u8\000vmovltq_m\000vmovltq_m_s16\000vmovltq_m_s8\000vmo"
    "vltq_m_u16\000vmovltq_m_u8\000vmovltq\000vmovltq_s16\000vmovltq_s8\000v"
    "movltq_u16\000vmovltq_u8\000vmovltq_x\000vmovltq_x_s16\000vmovltq_x_s8\000"
    "vmovltq_x_u16\000vmovltq_x_u8\000vmovnbq_m\000vmovnbq_m_s16\000vmovnbq_"
    "m_s32\000vmovnbq_m_u16\000vmovnbq_m_u32\000vmovnbq\000vmovnbq_s16\000vm"
    "ovnbq_s32\000vmovnbq_u16\000vmovnbq_u32\000vmovntq_m\000vmovntq_m_s16\000"
    "vmovntq_m_s32\000vmovntq_m_u16\000vmovntq_m_u32\000vmovntq\000vmovntq_s"
    "16\000vmovntq_s32\000vmovntq_u16\000vmovntq_u32\000vmulhq_m\000vmulhq_m"
    "_s16\000vmulhq_m_s32\000vmulhq_m_s8\000vmulhq_m_u16\000vmulhq_m_u32\000"
    "vmulhq_m_u8\000vmulhq\000vmulhq_s16\000vmulhq_s32\000vmulhq_s8\000vmulh"
    "q_u16\000vmulhq_u32\000vmulhq_u8\000vmulhq_x\000vmulhq_x_s16\000vmulhq_"
    "x_s32\000vmulhq_x_s8\000vmulhq_x_u16\000vmulhq_x_u32\000vmulhq_x_u8\000"
    "vmullbq_int_m\000vmullbq_int_m_s16\000vmullbq_int_m_s32\000vmullbq_int_"
    "m_s8\000vmullbq_int_m_u16\000vmullbq_int_m_u32\000vmullbq_int_m_u8\000v"
    "mullbq_int\000vmullbq_int_s16\000vmullbq_int_s32\000vmullbq_int_s8\000v"
    "mullbq_int_u16\000vmullbq_int_u32\000vmullbq_int_u8\000vmullbq_int_x\000"
    "vmullbq_int_x_s16\000vmullbq_int_x_s32\000vmullbq_int_x_s8\000vmullbq_i"
    "nt_x_u16\000vmullbq_int_x_u32\000vmullbq_int_x_u8\000vmullbq_poly_m\000"
    "vmullbq_poly_m_p16\000vmullbq_poly_m_p8\000vmullbq_poly\000vmullbq_poly"
    "_p16\000vmullbq_poly_p8\000vmullbq_poly_x\000vmullbq_poly_x_p16\000vmul"
    "lbq_poly_x_p8\000vmulltq_int_m\000vmulltq_int_m_s16\000vmulltq_int_m_s3"
    "2\000vmulltq_int_m_s8\000vmulltq_int_m_u16\000vmulltq_int_m_u32\000vmul"
    "ltq_int_m_u8\000vmulltq_int\000vmulltq_int_s16\000vmulltq_int_s32\000vm"
    "ulltq_int_s8\000vmulltq_int_u16\000vmulltq_int_u32\000vmulltq_int_u8\000"
    "vmulltq_int_x\000vmulltq_int_x_s16\000vmulltq_int_x_s32\000vmulltq_int_"
    "x_s8\000vmulltq_int_x_u16\000vmulltq_int_x_u32\000vmulltq_int_x_u8\000v"
    "mulltq_poly_m\000vmulltq_poly_m_p16\000vmulltq_poly_m_p8\000vmulltq_pol"
    "y\000vmulltq_poly_p16\000vmulltq_poly_p8\000vmulltq_poly_x\000vmulltq_p"
    "oly_x_p16\000vmulltq_poly_x_p8\000vmulq\000vmulq_f16\000vmulq_f32\000vm"
    "ulq_m\000vmulq_m_f16\000vmulq_m_f32\000vmulq_m_n_f16\000vmulq_m_n_f32\000"
    "vmulq_m_n_s16\000vmulq_m_n_s32\000vmulq_m_n_s8\000vmulq_m_n_u16\000vmul"
    "q_m_n_u32\000vmulq_m_n_u8\000vmulq_m_s16\000vmulq_m_s32\000vmulq_m_s8\000"
    "vmulq_m_u16\000vmulq_m_u32\000vmulq_m_u8\000vmulq_n_f16\000vmulq_n_f32\000"
    "vmulq_n_s16\000vmulq_n_s32\000vmulq_n_s8\000vmulq_n_u16\000vmulq_n_u32\000"
    "vmulq_n_u8\000vmulq_s16\000vmulq_s32\000vmulq_s8\000vmulq_u16\000vmulq_"
    "u32\000vmulq_u8\000vmulq_x\000vmulq_x_f16\000vmulq_x_f32\000vmulq_x_n_f"
    "16\000vmulq_x_n_f32\000vmulq_x_n_s16\000vmulq_x_n_s32\000vmulq_x_n_s8\000"
    "vmulq_x_n_u16\000vmulq_x_n_u32\000vmulq_x_n_u8\000vmulq_x_s16\000vmulq_"
    "x_s32\000vmulq_x_s8\000vmulq_x_u16\000vmulq_x_u32\000vmulq_x_u8\000vmvn"
    "q_m\000vmvnq_m_n_s16\000vmvnq_m_n_s32\000vmvnq_m_n_u16\000vmvnq_m_n_u32"
    "\000vmvnq_m_s16\000vmvnq_m_s32\000vmvnq_m_s8\000vmvnq_m_u16\000vmvnq_m_"
    "u32\000vmvnq_m_u8\000vmvnq_n_s16\000vmvnq_n_s32\000vmvnq_n_u16\000vmvnq"
    "_n_u32\000vmvnq\000vmvnq_s16\000vmvnq_s32\000vmvnq_s8\000vmvnq_u16\000v"
    "mvnq_u32\000vmvnq_u8\000vmvnq_x_n_s16\000vmvnq_x_n_s32\000vmvnq_x_n_u16"
    "\000vmvnq_x_n_u32\000vmvnq_x\000vmvnq_x_s16\000vmvnq_x_s32\000vmvnq_x_s"
    "8\000vmvnq_x_u16\000vmvnq_x_u32\000vmvnq_x_u8\000vnegq\000vnegq_f16\000"
    "vnegq_f32\000vnegq_m\000vnegq_m_f16\000vnegq_m_f32\000vnegq_m_s16\000vn"
    "egq_m_s32\000vnegq_m_s8\000vnegq_s16\000vnegq_s32\000vnegq_s8\000vnegq_"
    "x\000vnegq_x_f16\000vnegq_x_f32\000vnegq_x_s16\000vnegq_x_s32\000vnegq_"
    "x_s8\000vornq\000vornq_f16\000vornq_f32\000vornq_m\000vornq_m_f16\000vo"
    "rnq_m_f32\000vornq_m_s16\000vornq_m_s32\000vornq_m_s8\000vornq_m_u16\000"
    "vornq_m_u32\000vornq_m_u8\000vornq_s16\000vornq_s32\000vornq_s8\000vorn"
    "q_u16\000vornq_u32\000vornq_u8\000vornq_x\000vornq_x_f16\000vornq_x_f32"
    "\000vornq_x_s16\000vornq_x_s32\000vornq_x_s8\000vornq_x_u16\000vornq_x_"
    "u32\000vornq_x_u8\000vorrq\000vorrq_f16\000vorrq_f32\000vorrq_m\000vorr"
    "q_m_f16\000vorrq_m_f32\000vorrq_m_n\000vorrq_m_n_s16\000vorrq_m_n_s32\000"
    "vorrq_m_n_u16\000vorrq_m_n_u32\000vorrq_m_s16\000vorrq_m_s32\000vorrq_m"
    "_s8\000vorrq_m_u16\000vorrq_m_u32\000vorrq_m_u8\000vorrq_n_s16\000vorrq"
    "_n_s32\000vorrq_n_u16\000vorrq_n_u32\000vorrq_s16\000vorrq_s32\000vorrq"
    "_s8\000vorrq_u16\000vorrq_u32\000vorrq_u8\000vorrq_x\000vorrq_x_f16\000"
    "vorrq_x_f32\000vorrq_x_s16\000vorrq_x_s32\000vorrq_x_s8\000vorrq_x_u16\000"
    "vorrq_x_u32\000vorrq_x_u8\000vpnot\000vpselq\000vpselq_f16\000vpselq_f3"
    "2\000vpselq_s16\000vpselq_s32\000vpselq_s64\000vpselq_s8\000vpselq_u16\000"
    "vpselq_u32\000vpselq_u64\000vpselq_u8\000vqabsq_m\000vqabsq_m_s16\000vq"
    "absq_m_s32\000vqabsq_m_s8\000vqabsq\000vqabsq_s16\000vqabsq_s32\000vqab"
    "sq_s8\000vqaddq_m\000vqaddq_m_n_s16\000vqaddq_m_n_s32\000vqaddq_m_n_s8\000"
    "vqaddq_m_n_u16\000vqaddq_m_n_u32\000vqaddq_m_n_u8\000vqaddq_m_s16\000vq"
    "addq_m_s32\000vqaddq_m_s8\000vqaddq_m_u16\000vqaddq_m_u32\000vqaddq_m_u"
    "8\000vqaddq\000vqaddq_n_s16\000vqaddq_n_s32\000vqaddq_n_s8\000vqaddq_n_"
    "u16\000vqaddq_n_u32\000vqaddq_n_u8\000vqaddq_s16\000vqaddq_s32\000vqadd"
    "q_s8\000vqaddq_u16\000vqaddq_u32\000vqaddq_u8\000vqdmladhq_m\000vqdmlad"
    "hq_m_s16\000vqdmladhq_m_s32\000vqdmladhq_m_s8\000vqdmladhq\000vqdmladhq"
    "_s16\000vqdmladhq_s32\000vqdmladhq_s8\000vqdmladhxq_m\000vqdmladhxq_m_s"
    "16\000vqdmladhxq_m_s32\000vqdmladhxq_m_s8\000vqdmladhxq\000vqdmladhxq_s"
    "16\000vqdmladhxq_s32\000vqdmladhxq_s8\000vqdmlahq_m\000vqdmlahq_m_n_s16"
    "\000vqdmlahq_m_n_s32\000vqdmlahq_m_n_s8\000vqdmlahq\000vqdmlahq_n_s16\000"
    "vqdmlahq_n_s32\000vqdmlahq_n_s8\000vqdmlashq_m\000vqdmlashq_m_n_s16\000"
    "vqdmlashq_m_n_s32\000vqdmlashq_m_n_s8\000vqdmlashq\000vqdmlashq_n_s16\000"
    "vqdmlashq_n_s32\000vqdmlashq_n_s8\000vqdmlsdhq_m\000vqdmlsdhq_m_s16\000"
    "vqdmlsdhq_m_s32\000vqdmlsdhq_m_s8\000vqdmlsdhq\000vqdmlsdhq_s16\000vqdm"
    "lsdhq_s32\000vqdmlsdhq_s8\000vqdmlsdhxq_m\000vqdmlsdhxq_m_s16\000vqdmls"
    "dhxq_m_s32\000vqdmlsdhxq_m_s8\000vqdmlsdhxq\000vqdmlsdhxq_s16\000vqdmls"
    "dhxq_s32\000vqdmlsdhxq_s8\000vqdmulhq_m\000vqdmulhq_m_n_s16\000vqdmulhq"
    "_m_n_s32\000vqdmulhq_m_n_s8\000vqdmulhq_m_s16\000vqdmulhq_m_s32\000vqdm"
    "ulhq_m_s8\000vqdmulhq\000vqdmulhq_n_s16\000vqdmulhq_n_s32\000vqdmulhq_n"
    "_s8\000vqdmulhq_s16\000vqdmulhq_s32\000vqdmulhq_s8\000vqdmullbq_m\000vq"
    "dmullbq_m_n_s16\000vqdmullbq_m_n_s32\000vqdmullbq_m_s16\000vqdmullbq_m_"
    "s32\000vqdmullbq\000vqdmullbq_n_s16\000vqdmullbq_n_s32\000vqdmullbq_s16"
    "\000vqdmullbq_s32\000vqdmulltq_m\000vqdmulltq_m_n_s16\000vqdmulltq_m_n_"
    "s32\000vqdmulltq_m_s16\000vqdmulltq_m_s32\000vqdmulltq\000vqdmulltq_n_s"
    "16\000vqdmulltq_n_s32\000vqdmulltq_s16\000vqdmulltq_s32\000vqmovnbq_m\000"
    "vqmovnbq_m_s16\000vqmovnbq_m_s32\000vqmovnbq_m_u16\000vqmovnbq_m_u32\000"
    "vqmovnbq\000vqmovnbq_s16\000vqmovnbq_s32\000vqmovnbq_u16\000vqmovnbq_u3"
    "2\000vqmovntq_m\000vqmovntq_m_s16\000vqmovntq_m_s32\000vqmovntq_m_u16\000"
    "vqmovntq_m_u32\000vqmovntq\000vqmovntq_s16\000vqmovntq_s32\000vqmovntq_"
    "u16\000vqmovntq_u32\000vqmovunbq_m\000vqmovunbq_m_s16\000vqmovunbq_m_s3"
    "2\000vqmovunbq\000vqmovunbq_s16\000vqmovunbq_s32\000vqmovuntq_m\000vqmo"
    "vuntq_m_s16\000vqmovuntq_m_s32\000vqmovuntq\000vqmovuntq_s16\000vqmovun"
    "tq_s32\000vqnegq_m\000vqnegq_m_s16\000vqnegq_m_s32\000vqnegq_m_s8\000vq"
    "negq\000vqnegq_s16\000vqnegq_s32\000vqnegq_s8\000vqrdmladhq_m\000vqrdml"
    "adhq_m_s16\000vqrdmladhq_m_s32\000vqrdmladhq_m_s8\000vqrdmladhq\000vqrd"
    "mladhq_s16\000vqrdmladhq_s32\000vqrdmladhq_s8\000vqrdmladhxq_m\000vqrdm"
    "ladhxq_m_s16\000vqrdmladhxq_m_s32\000vqrdmladhxq_m_s8\000vqrdmladhxq\000"
    "vqrdmladhxq_s16\000vqrdmladhxq_s32\000vqrdmladhxq_s8\000vqrdmlahq_m\000"
    "vqrdmlahq_m_n_s16\000vqrdmlahq_m_n_s32\000vqrdmlahq_m_n_s8\000vqrdmlahq"
    "\000vqrdmlahq_n_s16\000vqrdmlahq_n_s32\000vqrdmlahq_n_s8\000vqrdmlashq_"
    "m\000vqrdmlashq_m_n_s16\000vqrdmlashq_m_n_s32\000vqrdmlashq_m_n_s8\000v"
    "qrdmlashq\000vqrdmlashq_n_s16\000vqrdmlashq_n_s32\000vqrdmlashq_n_s8\000"
    "vqrdmlsdhq_m\000vqrdmlsdhq_m_s16\000vqrdmlsdhq_m_s32\000vqrdmlsdhq_m_s8"
    "\000vqrdmlsdhq\000vqrdmlsdhq_s16\000vqrdmlsdhq_s32\000vqrdmlsdhq_s8\000"
    "vqrdmlsdhxq_m\000vqrdmlsdhxq_m_s16\000vqrdmlsdhxq_m_s32\000vqrdmlsdhxq_"
    "m_s8\000vqrdmlsdhxq\000vqrdmlsdhxq_s16\000vqrdmlsdhxq_s32\000vqrdmlsdhx"
    "q_s8\000vqrdmulhq_m\000vqrdmulhq_m_n_s16\000vqrdmulhq_m_n_s32\000vqrdmu"
    "lhq_m_n_s8\000vqrdmulhq_m_s16\000vqrdmulhq_m_s32\000vqrdmulhq_m_s8\000v"
    "qrdmulhq\000vqrdmulhq_n_s16\000vqrdmulhq_n_s32\000vqrdmulhq_n_s8\000vqr"
    "dmulhq_s16\000vqrdmulhq_s32\000vqrdmulhq_s8\000vqrshlq_m_n\000vqrshlq_m"
    "_n_s16\000vqrshlq_m_n_s32\000vqrshlq_m_n_s8\000vqrshlq_m_n_u16\000vqrsh"
    "lq_m_n_u32\000vqrshlq_m_n_u8\000vqrshlq_m\000vqrshlq_m_s16\000vqrshlq_m"
    "_s32\000vqrshlq_m_s8\000vqrshlq_m_u16\000vqrshlq_m_u32\000vqrshlq_m_u8\000"
    "vqrshlq\000vqrshlq_n_s16\000vqrshlq_n_s32\000vqrshlq_n_s8\000vqrshlq_n_"
    "u16\000vqrshlq_n_u32\000vqrshlq_n_u8\000vqrshlq_s16\000vqrshlq_s32\000v"
    "qrshlq_s8\000vqrshlq_u16\000vqrshlq_u32\000vqrshlq_u8\000vqrshrnbq_m\000"
    "vqrshrnbq_m_n_s16\000vqrshrnbq_m_n_s32\000vqrshrnbq_m_n_u16\000vqrshrnb"
    "q_m_n_u32\000vqrshrnbq\000vqrshrnbq_n_s16\000vqrshrnbq_n_s32\000vqrshrn"
    "bq_n_u16\000vqrshrnbq_n_u32\000vqrshrntq_m\000vqrshrntq_m_n_s16\000vqrs"
    "hrntq_m_n_s32\000vqrshrntq_m_n_u16\000vqrshrntq_m_n_u32\000vqrshrntq\000"
    "vqrshrntq_n_s16\000vqrshrntq_n_s32\000vqrshrntq_n_u16\000vqrshrntq_n_u3"
    "2\000vqrshrunbq_m\000vqrshrunbq_m_n_s16\000vqrshrunbq_m_n_s32\000vqrshr"
    "unbq\000vqrshrunbq_n_s16\000vqrshrunbq_n_s32\000vqrshruntq_m\000vqrshru"
    "ntq_m_n_s16\000vqrshruntq_m_n_s32\000vqrshruntq\000vqrshruntq_n_s16\000"
    "vqrshruntq_n_s32\000vqshlq_m_n\000vqshlq_m_n_s16\000vqshlq_m_n_s32\000v"
    "qshlq_m_n_s8\000vqshlq_m_n_u16\000vqshlq_m_n_u32\000vqshlq_m_n_u8\000vq"
    "shlq_m_r\000vqshlq_m_r_s16\000vqshlq_m_r_s32\000vqshlq_m_r_s8\000vqshlq"
    "_m_r_u16\000vqshlq_m_r_u32\000vqshlq_m_r_u8\000vqshlq_m\000vqshlq_m_s16"
    "\000vqshlq_m_s32\000vqshlq_m_s8\000vqshlq_m_u16\000vqshlq_m_u32\000vqsh"
    "lq_m_u8\000vqshlq_n\000vqshlq_n_s16\000vqshlq_n_s32\000vqshlq_n_s8\000v"
    "qshlq_n_u16\000vqshlq_n_u32\000vqshlq_n_u8\000vqshlq_r\000vqshlq_r_s16\000"
    "vqshlq_r_s32\000vqshlq_r_s8\000vqshlq_r_u16\000vqshlq_r_u32\000vqshlq_r"
    "_u8\000vqshlq\000vqshlq_s16\000vqshlq_s32\000vqshlq_s8\000vqshlq_u16\000"
    "vqshlq_u32\000vqshlq_u8\000vqshluq_m\000vqshluq_m_n_s16\000vqshluq_m_n_"
    "s32\000vqshluq_m_n_s8\000vqshluq\000vqshluq_n_s16\000vqshluq_n_s32\000v"
    "qshluq_n_s8\000vqshrnbq_m\000vqshrnbq_m_n_s16\000vqshrnbq_m_n_s32\000vq"
    "shrnbq_m_n_u16\000vqshrnbq_m_n_u32\000vqshrnbq\000vqshrnbq_n_s16\000vqs"
    "hrnbq_n_s32\000vqshrnbq_n_u16\000vqshrnbq_n_u32\000vqshrntq_m\000vqshrn"
    "tq_m_n_s16\000vqshrntq_m_n_s32\000vqshrntq_m_n_u16\000vqshrntq_m_n_u32\000"
    "vqshrntq\000vqshrntq_n_s16\000vqshrntq_n_s32\000vqshrntq_n_u16\000vqshr"
    "ntq_n_u32\000vqshrunbq_m\000vqshrunbq_m_n_s16\000vqshrunbq_m_n_s32\000v"
    "qshrunbq\000vqshrunbq_n_s16\000vqshrunbq_n_s32\000vqshruntq_m\000vqshru"
    "ntq_m_n_s16\000vqshruntq_m_n_s32\000vqshruntq\000vqshruntq_n_s16\000vqs"
    "hruntq_n_s32\000vqsubq_m\000vqsubq_m_n_s16\000vqsubq_m_n_s32\000vqsubq_"
    "m_n_s8\000vqsubq_m_n_u16\000vqsubq_m_n_u32\000vqsubq_m_n_u8\000vqsubq_m"
    "_s16\000vqsubq_m_s32\000vqsubq_m_s8\000vqsubq_m_u16\000vqsubq_m_u32\000"
    "vqsubq_m_u8\000vqsubq\000vqsubq_n_s16\000vqsubq_n_s32\000vqsubq_n_s8\000"
    "vqsubq_n_u16\000vqsubq_n_u32\000vqsubq_n_u8\000vqsubq_s16\000vqsubq_s32"
    "\000vqsubq_s8\000vqsubq_u16\000vqsubq_u32\000vqsubq_u8\000vreinterpretq"
    "_f16\000vreinterpretq_f16_f32\000vreinterpretq_f16_s16\000vreinterpretq"
    "_f16_s32\000vreinterpretq_f16_s64\000vreinterpretq_f16_s8\000vreinterpr"
    "etq_f16_u16\000vreinterpretq_f16_u32\000vreinterpretq_f16_u64\000vreint"
    "erpretq_f16_u8\000vreinterpretq_f32\000vreinterpretq_f32_f16\000vreinte"
    "rpretq_f32_s16\000vreinterpretq_f32_s32\000vreinterpretq_f32_s64\000vre"
    "interpretq_f32_s8\000vreinterpretq_f32_u16\000vreinterpretq_f32_u32\000"
    "vreinterpretq_f32_u64\000vreinterpretq_f32_u8\000vreinterpretq_s16\000v"
    "reinterpretq_s16_f16\000vreinterpretq_s16_f32\000vreinterpretq_s16_s32\000"
    "vreinterpretq_s16_s64\000vreinterpretq_s16_s8\000vreinterpretq_s16_u16\000"
    "vreinterpretq_s16_u32\000vreinterpretq_s16_u64\000vreinterpretq_s16_u8\000"
    "vreinterpretq_s32\000vreinterpretq_s32_f16\000vreinterpretq_s32_f32\000"
    "vreinterpretq_s32_s16\000vreinterpretq_s32_s64\000vreinterpretq_s32_s8\000"
    "vreinterpretq_s32_u16\000vreinterpretq_s32_u32\000vreinterpretq_s32_u64"
    "\000vreinterpretq_s32_u8\000vreinterpretq_s64\000vreinterpretq_s64_f16\000"
    "vreinterpretq_s64_f32\000vreinterpretq_s64_s16\000vreinterpretq_s64_s32"
    "\000vreinterpretq_s64_s8\000vreinterpretq_s64_u16\000vreinterpretq_s64_"
    "u32\000vreinterpretq_s64_u64\000vreinterpretq_s64_u8\000vreinterpretq_s"
    "8\000vreinterpretq_s8_f16\000vreinterpretq_s8_f32\000vreinterpretq_s8_s"
    "16\000vreinterpretq_s8_s32\000vreinterpretq_s8_s64\000vreinterpretq_s8_"
    "u16\000vreinterpretq_s8_u32\000vreinterpretq_s8_u64\000vreinterpretq_s8"
    "_u8\000vreinterpretq_u16\000vreinterpretq_u16_f16\000vreinterpretq_u16_"
    "f32\000vreinterpretq_u16_s16\000vreinterpretq_u16_s32\000vreinterpretq_"
    "u16_s64\000vreinterpretq_u16_s8\000vreinterpretq_u16_u32\000vreinterpre"
    "tq_u16_u64\000vreinterpretq_u16_u8\000vreinterpretq_u32\000vreinterpret"
    "q_u32_f16\000vreinterpretq_u32_f32\000vreinterpretq_u32_s16\000vreinter"
    "pretq_u32_s32\000vreinterpretq_u32_s64\000vreinterpretq_u32_s8\000vrein"
    "terpretq_u32_u16\000vreinterpretq_u32_u64\000vreinterpretq_u32_u8\000vr"
    "einterpretq_u64\000vreinterpretq_u64_f16\000vreinterpretq_u64_f32\000vr"
    "einterpretq_u64_s16\000vreinterpretq_u64_s32\000vreinterpretq_u64_s64\000"
    "vreinterpretq_u64_s8\000vreinterpretq_u64_u16\000vreinterpretq_u64_u32\000"
    "vreinterpretq_u64_u8\000vreinterpretq_u8\000vreinterpretq_u8_f16\000vre"
    "interpretq_u8_f32\000vreinterpretq_u8_s16\000vreinterpretq_u8_s32\000vr"
    "einterpretq_u8_s64\000vreinterpretq_u8_s8\000vreinterpretq_u8_u16\000vr"
    "einterpretq_u8_u32\000vreinterpretq_u8_u64\000vrev16q_m\000vrev16q_m_s8"
    "\000vrev16q_m_u8\000vrev16q\000vrev16q_s8\000vrev16q_u8\000vrev16q_x\000"
    "vrev16q_x_s8\000vrev16q_x_u8\000vrev32q\000vrev32q_f16\000vrev32q_m\000"
    "vrev32q_m_f16\000vrev32q_m_s16\000vrev32q_m_s8\000vrev32q_m_u16\000vrev"
    "32q_m_u8\000vrev32q_s16\000vrev32q_s8\000vrev32q_u16\000vrev32q_u8\000v"
    "rev32q_x\000vrev32q_x_f16\000vrev32q_x_s16\000vrev32q_x_s8\000vrev32q_x"
    "_u16\000vrev32q_x_u8\000vrev64q\000vrev64q_f16\000vrev64q_f32\000vrev64"
    "q_m\000vrev64q_m_f16\000vrev64q_m_f32\000vrev64q_m_s16\000vrev64q_m_s32"
    "\000vrev64q_m_s8\000vrev64q_m_u16\000vrev64q_m_u32\000vrev64q_m_u8\000v"
    "rev64q_s16\000vrev64q_s32\000vrev64q_s8\000vrev64q_u16\000vrev64q_u32\000"
    "vrev64q_u8\000vrev64q_x\000vrev64q_x_f16\000vrev64q_x_f32\000vrev64q_x_"
    "s16\000vrev64q_x_s32\000vrev64q_x_s8\000vrev64q_x_u16\000vrev64q_x_u32\000"
    "vrev64q_x_u8\000vrhaddq_m\000vrhaddq_m_s16\000vrhaddq_m_s32\000vrhaddq_"
    "m_s8\000vrhaddq_m_u16\000vrhaddq_m_u32\000vrhaddq_m_u8\000vrhaddq\000vr"
    "haddq_s16\000vrhaddq_s32\000vrhaddq_s8\000vrhaddq_u16\000vrhaddq_u32\000"
    "vrhaddq_u8\000vrhaddq_x\000vrhaddq_x_s16\000vrhaddq_x_s32\000vrhaddq_x_"
    "s8\000vrhaddq_x_u16\000vrhaddq_x_u32\000vrhaddq_x_u8\000vrmlaldavhaq_p\000"
    "vrmlaldavhaq_p_s32\000vrmlaldavhaq_p_u32\000vrmlaldavhaq\000vrmlaldavha"
    "q_s32\000vrmlaldavhaq_u32\000vrmlaldavhaxq_p\000vrmlaldavhaxq_p_s32\000"
    "vrmlaldavhaxq\000vrmlaldavhaxq_s32\000vrmlaldavhq_p\000vrmlaldavhq_p_s3"
    "2\000vrmlaldavhq_p_u32\000vrmlaldavhq\000vrmlaldavhq_s32\000vrmlaldavhq"
    "_u32\000vrmlaldavhxq_p\000vrmlaldavhxq_p_s32\000vrmlaldavhxq\000vrmlald"
    "avhxq_s32\000vrmlsldavhaq_p\000vrmlsldavhaq_p_s32\000vrmlsldavhaq\000vr"
    "mlsldavhaq_s32\000vrmlsldavhaxq_p\000vrmlsldavhaxq_p_s32\000vrmlsldavha"
    "xq\000vrmlsldavhaxq_s32\000vrmlsldavhq_p\000vrmlsldavhq_p_s32\000vrmlsl"
    "davhq\000vrmlsldavhq_s32\000vrmlsldavhxq_p\000vrmlsldavhxq_p_s32\000vrm"
    "lsldavhxq\000vrmlsldavhxq_s32\000vrmulhq_m\000vrmulhq_m_s16\000vrmulhq_"
    "m_s32\000vrmulhq_m_s8\000vrmulhq_m_u16\000vrmulhq_m_u32\000vrmulhq_m_u8"
    "\000vrmulhq\000vrmulhq_s16\000vrmulhq_s32\000vrmulhq_s8\000vrmulhq_u16\000"
    "vrmulhq_u32\000vrmulhq_u8\000vrmulhq_x\000vrmulhq_x_s16\000vrmulhq_x_s3"
    "2\000vrmulhq_x_s8\000vrmulhq_x_u16\000vrmulhq_x_u32\000vrmulhq_x_u8\000"
    "vrndaq\000vrndaq_f16\000vrndaq_f32\000vrndaq_m\000vrndaq_m_f16\000vrnda"
    "q_m_f32\000vrndaq_x\000vrndaq_x_f16\000vrndaq_x_f32\000vrndmq\000vrndmq"
    "_f16\000vrndmq_f32\000vrndmq_m\000vrndmq_m_f16\000vrndmq_m_f32\000vrndm"
    "q_x\000vrndmq_x_f16\000vrndmq_x_f32\000vrndnq\000vrndnq_f16\000vrndnq_f"
    "32\000vrndnq_m\000vrndnq_m_f16\000vrndnq_m_f32\000vrndnq_x\000vrndnq_x_"
    "f16\000vrndnq_x_f32\000vrndpq\000vrndpq_f16\000vrndpq_f32\000vrndpq_m\000"
    "vrndpq_m_f16\000vrndpq_m_f32\000vrndpq_x\000vrndpq_x_f16\000vrndpq_x_f3"
    "2\000vrndq\000vrndq_f16\000vrndq_f32\000vrndq_m\000vrndq_m_f16\000vrndq"
    "_m_f32\000vrndq_x\000vrndq_x_f16\000vrndq_x_f32\000vrndxq\000vrndxq_f16"
    "\000vrndxq_f32\000vrndxq_m\000vrndxq_m_f16\000vrndxq_m_f32\000vrndxq_x\000"
    "vrndxq_x_f16\000vrndxq_x_f32\000vrshlq_m_n\000vrshlq_m_n_s16\000vrshlq_"
    "m_n_s32\000vrshlq_m_n_s8\000vrshlq_m_n_u16\000vrshlq_m_n_u32\000vrshlq_"
    "m_n_u8\000vrshlq_m\000vrshlq_m_s16\000vrshlq_m_s32\000vrshlq_m_s8\000vr"
    "shlq_m_u16\000vrshlq_m_u32\000vrshlq_m_u8\000vrshlq\000vrshlq_n_s16\000"
    "vrshlq_n_s32\000vrshlq_n_s8\000vrshlq_n_u16\000vrshlq_n_u32\000vrshlq_n"
    "_u8\000vrshlq_s16\000vrshlq_s32\000vrshlq_s8\000vrshlq_u16\000vrshlq_u3"
    "2\000vrshlq_u8\000vrshlq_x\000vrshlq_x_s16\000vrshlq_x_s32\000vrshlq_x_"
    "s8\000vrshlq_x_u16\000vrshlq_x_u32\000vrshlq_x_u8\000vrshrnbq_m\000vrsh"
    "rnbq_m_n_s16\000vrshrnbq_m_n_s32\000vrshrnbq_m_n_u16\000vrshrnbq_m_n_u3"
    "2\000vrshrnbq\000vrshrnbq_n_s16\000vrshrnbq_n_s32\000vrshrnbq_n_u16\000"
    "vrshrnbq_n_u32\000vrshrntq_m\000vrshrntq_m_n_s16\000vrshrntq_m_n_s32\000"
    "vrshrntq_m_n_u16\000vrshrntq_m_n_u32\000vrshrntq\000vrshrntq_n_s16\000v"
    "rshrntq_n_s32\000vrshrntq_n_u16\000vrshrntq_n_u32\000vrshrq_m\000vrshrq"
    "_m_n_s16\000vrshrq_m_n_s32\000vrshrq_m_n_s8\000vrshrq_m_n_u16\000vrshrq"
    "_m_n_u32\000vrshrq_m_n_u8\000vrshrq\000vrshrq_n_s16\000vrshrq_n_s32\000"
    "vrshrq_n_s8\000vrshrq_n_u16\000vrshrq_n_u32\000vrshrq_n_u8\000vrshrq_x\000"
    "vrshrq_x_n_s16\000vrshrq_x_n_s32\000vrshrq_x_n_s8\000vrshrq_x_n_u16\000"
    "vrshrq_x_n_u32\000vrshrq_x_n_u8\000vsbciq_m\000vsbciq_m_s32\000vsbciq_m"
    "_u32\000vsbciq\000vsbciq_s32\000vsbciq_u32\000vsbcq_m\000vsbcq_m_s32\000"
    "vsbcq_m_u32\000vsbcq\000vsbcq_s32\000vsbcq_u32\000vsetq_lane\000vsetq_l"
    "ane_f16\000vsetq_lane_f32\000vsetq_lane_s16\000vsetq_lane_s32\000vsetq_"
    "lane_s64\000vsetq_lane_s8\000vsetq_lane_u16\000vsetq_lane_u32\000vsetq_"
    "lane_u64\000vsetq_lane_u8\000vshlcq_m\000vshlcq_m_s16\000vshlcq_m_s32\000"
    "vshlcq_m_s8\000vshlcq_m_u16\000vshlcq_m_u32\000vshlcq_m_u8\000vshlcq\000"
    "vshlcq_s16\000vshlcq_s32\000vshlcq_s8\000vshlcq_u16\000vshlcq_u32\000vs"
    "hlcq_u8\000vshllbq_m\000vshllbq_m_n_s16\000vshllbq_m_n_s8\000vshllbq_m_"
    "n_u16\000vshllbq_m_n_u8\000vshllbq\000vshllbq_n_s16\000vshllbq_n_s8\000"
    "vshllbq_n_u16\000vshllbq_n_u8\000vshllbq_x\000vshllbq_x_n_s16\000vshllb"
    "q_x_n_s8\000vshllbq_x_n_u16\000vshllbq_x_n_u8\000vshlltq_m\000vshlltq_m"
    "_n_s16\000vshlltq_m_n_s8\000vshlltq_m_n_u16\000vshlltq_m_n_u8\000vshllt"
    "q\000vshlltq_n_s16\000vshlltq_n_s8\000vshlltq_n_u16\000vshlltq_n_u8\000"
    "vshlltq_x\000vshlltq_x_n_s16\000vshlltq_x_n_s8\000vshlltq_x_n_u16\000vs"
    "hlltq_x_n_u8\000vshlq_m_n\000vshlq_m_n_s16\000vshlq_m_n_s32\000vshlq_m_"
    "n_s8\000vshlq_m_n_u16\000vshlq_m_n_u32\000vshlq_m_n_u8\000vshlq_m_r\000"
    "vshlq_m_r_s16\000vshlq_m_r_s32\000vshlq_m_r_s8\000vshlq_m_r_u16\000vshl"
    "q_m_r_u32\000vshlq_m_r_u8\000vshlq_m\000vshlq_m_s16\000vshlq_m_s32\000v"
    "shlq_m_s8\000vshlq_m_u16\000vshlq_m_u32\000vshlq_m_u8\000vshlq_n\000vsh"
    "lq_n_s16\000vshlq_n_s32\000vshlq_n_s8\000vshlq_n_u16\000vshlq_n_u32\000"
    "vshlq_n_u8\000vshlq_r\000vshlq_r_s16\000vshlq_r_s32\000vshlq_r_s8\000vs"
    "hlq_r_u16\000vshlq_r_u32\000vshlq_r_u8\000vshlq\000vshlq_s16\000vshlq_s"
    "32\000vshlq_s8\000vshlq_u16\000vshlq_u32\000vshlq_u8\000vshlq_x_n\000vs"
    "hlq_x_n_s16\000vshlq_x_n_s32\000vshlq_x_n_s8\000vshlq_x_n_u16\000vshlq_"
    "x_n_u32\000vshlq_x_n_u8\000vshlq_x\000vshlq_x_s16\000vshlq_x_s32\000vsh"
    "lq_x_s8\000vshlq_x_u16\000vshlq_x_u32\000vshlq_x_u8\000vshrnbq_m\000vsh"
    "rnbq_m_n_s16\000vshrnbq_m_n_s32\000vshrnbq_m_n_u16\000vshrnbq_m_n_u32\000"
    "vshrnbq\000vshrnbq_n_s16\000vshrnbq_n_s32\000vshrnbq_n_u16\000vshrnbq_n"
    "_u32\000vshrntq_m\000vshrntq_m_n_s16\000vshrntq_m_n_s32\000vshrntq_m_n_"
    "u16\000vshrntq_m_n_u32\000vshrntq\000vshrntq_n_s16\000vshrntq_n_s32\000"
    "vshrntq_n_u16\000vshrntq_n_u32\000vshrq_m\000vshrq_m_n_s16\000vshrq_m_n"
    "_s32\000vshrq_m_n_s8\000vshrq_m_n_u16\000vshrq_m_n_u32\000vshrq_m_n_u8\000"
    "vshrq\000vshrq_n_s16\000vshrq_n_s32\000vshrq_n_s8\000vshrq_n_u16\000vsh"
    "rq_n_u32\000vshrq_n_u8\000vshrq_x\000vshrq_x_n_s16\000vshrq_x_n_s32\000"
    "vshrq_x_n_s8\000vshrq_x_n_u16\000vshrq_x_n_u32\000vshrq_x_n_u8\000vsliq"
    "_m\000vsliq_m_n_s16\000vsliq_m_n_s32\000vsliq_m_n_s8\000vsliq_m_n_u16\000"
    "vsliq_m_n_u32\000vsliq_m_n_u8\000vsliq\000vsliq_n_s16\000vsliq_n_s32\000"
    "vsliq_n_s8\000vsliq_n_u16\000vsliq_n_u32\000vsliq_n_u8\000vsriq_m\000vs"
    "riq_m_n_s16\000vsriq_m_n_s32\000vsriq_m_n_s8\000vsriq_m_n_u16\000vsriq_"
    "m_n_u32\000vsriq_m_n_u8\000vsriq\000vsriq_n_s16\000vsriq_n_s32\000vsriq"
    "_n_s8\000vsriq_n_u16\000vsriq_n_u32\000vsriq_n_u8\000vst1q\000vst1q_f16"
    "\000vst1q_f32\000vst1q_p\000vst1q_p_f16\000vst1q_p_f32\000vst1q_p_s16\000"
    "vst1q_p_s32\000vst1q_p_s8\000vst1q_p_u16\000vst1q_p_u32\000vst1q_p_u8\000"
    "vst1q_s16\000vst1q_s32\000vst1q_s8\000vst1q_u16\000vst1q_u32\000vst1q_u"
    "8\000vst2q\000vst2q_f16\000vst2q_f32\000vst2q_s16\000vst2q_s32\000vst2q"
    "_s8\000vst2q_u16\000vst2q_u32\000vst2q_u8\000vst4q\000vst4q_f16\000vst4"
    "q_f32\000vst4q_s16\000vst4q_s32\000vst4q_s8\000vst4q_u16\000vst4q_u32\000"
    "vst4q_u8\000vstrbq_p\000vstrbq_p_s16\000vstrbq_p_s32\000vstrbq_p_s8\000"
    "vstrbq_p_u16\000vstrbq_p_u32\000vstrbq_p_u8\000vstrbq\000vstrbq_s16\000"
    "vstrbq_s32\000vstrbq_s8\000vstrbq_scatter_offset_p\000vstrbq_scatter_of"
    "fset_p_s16\000vstrbq_scatter_offset_p_s32\000vstrbq_scatter_offset_p_s8"
    "\000vstrbq_scatter_offset_p_u16\000vstrbq_scatter_offset_p_u32\000vstrb"
    "q_scatter_offset_p_u8\000vstrbq_scatter_offset\000vstrbq_scatter_offset"
    "_s16\000vstrbq_scatter_offset_s32\000vstrbq_scatter_offset_s8\000vstrbq"
    "_scatter_offset_u16\000vstrbq_scatter_offset_u32\000vstrbq_scatter_offs"
    "et_u8\000vstrbq_u16\000vstrbq_u32\000vstrbq_u8\000vstrdq_scatter_base_p"
    "\000vstrdq_scatter_base_p_s64\000vstrdq_scatter_base_p_u64\000vstrdq_sc"
    "atter_base\000vstrdq_scatter_base_s64\000vstrdq_scatter_base_u64\000vst"
    "rdq_scatter_base_wb_p\000vstrdq_scatter_base_wb_p_s64\000vstrdq_scatter"
    "_base_wb_p_u64\000vstrdq_scatter_base_wb\000vstrdq_scatter_base_wb_s64\000"
    "vstrdq_scatter_base_wb_u64\000vstrdq_scatter_offset_p\000vstrdq_scatter"
    "_offset_p_s64\000vstrdq_scatter_offset_p_u64\000vstrdq_scatter_offset\000"
    "vstrdq_scatter_offset_s64\000vstrdq_scatter_offset_u64\000vstrdq_scatte"
    "r_shifted_offset_p\000vstrdq_scatter_shifted_offset_p_s64\000vstrdq_sca"
    "tter_shifted_offset_p_u64\000vstrdq_scatter_shifted_offset\000vstrdq_sc"
    "atter_shifted_offset_s64\000vstrdq_scatter_shifted_offset_u64\000vstrhq"
    "\000vstrhq_f16\000vstrhq_p\000vstrhq_p_f16\000vstrhq_p_s16\000vstrhq_p_"
    "s32\000vstrhq_p_u16\000vstrhq_p_u32\000vstrhq_s16\000vstrhq_s32\000vstr"
    "hq_scatter_offset\000vstrhq_scatter_offset_f16\000vstrhq_scatter_offset"
    "_p\000vstrhq_scatter_offset_p_f16\000vstrhq_scatter_offset_p_s16\000vst"
    "rhq_scatter_offset_p_s32\000vstrhq_scatter_offset_p_u16\000vstrhq_scatt"
    "er_offset_p_u32\000vstrhq_scatter_offset_s16\000vstrhq_scatter_offset_s"
    "32\000vstrhq_scatter_offset_u16\000vstrhq_scatter_offset_u32\000vstrhq_"
    "scatter_shifted_offset\000vstrhq_scatter_shifted_offset_f16\000vstrhq_s"
    "catter_shifted_offset_p\000vstrhq_scatter_shifted_offset_p_f16\000vstrh"
    "q_scatter_shifted_offset_p_s16\000vstrhq_scatter_shifted_offset_p_s32\000"
    "vstrhq_scatter_shifted_offset_p_u16\000vstrhq_scatter_shifted_offset_p_"
    "u32\000vstrhq_scatter_shifted_offset_s16\000vstrhq_scatter_shifted_offs"
    "et_s32\000vstrhq_scatter_shifted_offset_u16\000vstrhq_scatter_shifted_o"
    "ffset_u32\000vstrhq_u16\000vstrhq_u32\000vstrwq\000vstrwq_f32\000vstrwq"
    "_p\000vstrwq_p_f32\000vstrwq_p_s32\000vstrwq_p_u32\000vstrwq_s32\000vst"
    "rwq_scatter_base\000vstrwq_scatter_base_f32\000vstrwq_scatter_base_p\000"
    "vstrwq_scatter_base_p_f32\000vstrwq_scatter_base_p_s32\000vstrwq_scatte"
    "r_base_p_u32\000vstrwq_scatter_base_s32\000vstrwq_scatter_base_u32\000v"
    "strwq_scatter_base_wb\000vstrwq_scatter_base_wb_f32\000vstrwq_scatter_b"
    "ase_wb_p\000vstrwq_scatter_base_wb_p_f32\000vstrwq_scatter_base_wb_p_s3"
    "2\000vstrwq_scatter_base_wb_p_u32\000vstrwq_scatter_base_wb_s32\000vstr"
    "wq_scatter_base_wb_u32\000vstrwq_scatter_offset\000vstrwq_scatter_offse"
    "t_f32\000vstrwq_scatter_offset_p\000vstrwq_scatter_offset_p_f32\000vstr"
    "wq_scatter_offset_p_s32\000vstrwq_scatter_offset_p_u32\000vstrwq_scatte"
    "r_offset_s32\000vstrwq_scatter_offset_u32\000vstrwq_scatter_shifted_off"
    "set\000vstrwq_scatter_shifted_offset_f32\000vstrwq_scatter_shifted_offs"
    "et_p\000vstrwq_scatter_shifted_offset_p_f32\000vstrwq_scatter_shifted_o"
    "ffset_p_s32\000vstrwq_scatter_shifted_offset_p_u32\000vstrwq_scatter_sh"
    "ifted_offset_s32\000vstrwq_scatter_shifted_offset_u32\000vstrwq_u32\000"
    "vsubq\000vsubq_f16\000vsubq_f32\000vsubq_m\000vsubq_m_f16\000vsubq_m_f3"
    "2\000vsubq_m_n_f16\000vsubq_m_n_f32\000vsubq_m_n_s16\000vsubq_m_n_s32\000"
    "vsubq_m_n_s8\000vsubq_m_n_u16\000vsubq_m_n_u32\000vsubq_m_n_u8\000vsubq"
    "_m_s16\000vsubq_m_s32\000vsubq_m_s8\000vsubq_m_u16\000vsubq_m_u32\000vs"
    "ubq_m_u8\000vsubq_n_f16\000vsubq_n_f32\000vsubq_n_s16\000vsubq_n_s32\000"
    "vsubq_n_s8\000vsubq_n_u16\000vsubq_n_u32\000vsubq_n_u8\000vsubq_s16\000"
    "vsubq_s32\000vsubq_s8\000vsubq_u16\000vsubq_u32\000vsubq_u8\000vsubq_x\000"
    "vsubq_x_f16\000vsubq_x_f32\000vsubq_x_n_f16\000vsubq_x_n_f32\000vsubq_x"
    "_n_s16\000vsubq_x_n_s32\000vsubq_x_n_s8\000vsubq_x_n_u16\000vsubq_x_n_u"
    "32\000vsubq_x_n_u8\000vsubq_x_s16\000vsubq_x_s32\000vsubq_x_s8\000vsubq"
    "_x_u16\000vsubq_x_u32\000vsubq_x_u8\000vuninitializedq_f16\000vuninitia"
    "lizedq_f32\000vuninitializedq\000vuninitializedq_polymorphic_f16\000vun"
    "initializedq_polymorphic_f32\000vuninitializedq_polymorphic_s16\000vuni"
    "nitializedq_polymorphic_s32\000vuninitializedq_polymorphic_s64\000vunin"
    "itializedq_polymorphic_s8\000vuninitializedq_polymorphic_u16\000vuninit"
    "ializedq_polymorphic_u32\000vuninitializedq_polymorphic_u64\000vuniniti"
    "alizedq_polymorphic_u8\000vuninitializedq_s16\000vuninitializedq_s32\000"
    "vuninitializedq_s64\000vuninitializedq_s8\000vuninitializedq_u16\000vun"
    "initializedq_u32\000vuninitializedq_u64\000vuninitializedq_u8\000"};

