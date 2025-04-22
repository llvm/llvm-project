/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* List of builtins that Clang recognizes                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


#ifdef GET_BUILTIN_ENUMERATORS
    BI_BitScanForward64,
    BI_BitScanReverse64,
    BI_InterlockedCompareExchange128,
    BI__faststorefence,
    BI__mulh,
    BI__shiftleft128,
    BI__shiftright128,
    BI__umulh,
    BI_mul128,
    BI_umul128,
    BI__builtin_ia32_aadd64,
    BI__builtin_ia32_aand64,
    BI__builtin_ia32_addcarryx_u64,
    BI__builtin_ia32_aor64,
    BI__builtin_ia32_axor64,
    BI__builtin_ia32_bextr_u64,
    BI__builtin_ia32_bextri_u64,
    BI__builtin_ia32_bzhi_di,
    BI__builtin_ia32_clui,
    BI__builtin_ia32_cmpccxadd32,
    BI__builtin_ia32_cmpccxadd64,
    BI__builtin_ia32_crc32di,
    BI__builtin_ia32_cvtsd2si64,
    BI__builtin_ia32_cvtsi2sd64,
    BI__builtin_ia32_cvtsi2ss64,
    BI__builtin_ia32_cvtss2si64,
    BI__builtin_ia32_cvttsd2si64,
    BI__builtin_ia32_cvttss2si64,
    BI__builtin_ia32_cvtusi2sd64,
    BI__builtin_ia32_cvtusi2ss64,
    BI__builtin_ia32_directstore_u64,
    BI__builtin_ia32_fxrstor64,
    BI__builtin_ia32_fxsave64,
    BI__builtin_ia32_incsspq,
    BI__builtin_ia32_lwpins64,
    BI__builtin_ia32_lwpval64,
    BI__builtin_ia32_lzcnt_u64,
    BI__builtin_ia32_movnti64,
    BI__builtin_ia32_movrsdi,
    BI__builtin_ia32_movrshi,
    BI__builtin_ia32_movrsqi,
    BI__builtin_ia32_movrssi,
    BI__builtin_ia32_pdep_di,
    BI__builtin_ia32_pext_di,
    BI__builtin_ia32_prefetchi,
    BI__builtin_ia32_ptwrite64,
    BI__builtin_ia32_rdfsbase32,
    BI__builtin_ia32_rdfsbase64,
    BI__builtin_ia32_rdgsbase32,
    BI__builtin_ia32_rdgsbase64,
    BI__builtin_ia32_rdrand64_step,
    BI__builtin_ia32_rdseed64_step,
    BI__builtin_ia32_rdsspq,
    BI__builtin_ia32_readeflags_u64,
    BI__builtin_ia32_senduipi,
    BI__builtin_ia32_stui,
    BI__builtin_ia32_subborrow_u64,
    BI__builtin_ia32_t2rpntlvwz0,
    BI__builtin_ia32_t2rpntlvwz0_internal,
    BI__builtin_ia32_t2rpntlvwz0rs,
    BI__builtin_ia32_t2rpntlvwz0rs_internal,
    BI__builtin_ia32_t2rpntlvwz0rst1,
    BI__builtin_ia32_t2rpntlvwz0rst1_internal,
    BI__builtin_ia32_t2rpntlvwz0t1,
    BI__builtin_ia32_t2rpntlvwz0t1_internal,
    BI__builtin_ia32_t2rpntlvwz1,
    BI__builtin_ia32_t2rpntlvwz1_internal,
    BI__builtin_ia32_t2rpntlvwz1rs,
    BI__builtin_ia32_t2rpntlvwz1rs_internal,
    BI__builtin_ia32_t2rpntlvwz1rst1,
    BI__builtin_ia32_t2rpntlvwz1rst1_internal,
    BI__builtin_ia32_t2rpntlvwz1t1,
    BI__builtin_ia32_t2rpntlvwz1t1_internal,
    BI__builtin_ia32_tcmmimfp16ps,
    BI__builtin_ia32_tcmmimfp16ps_internal,
    BI__builtin_ia32_tcmmrlfp16ps,
    BI__builtin_ia32_tcmmrlfp16ps_internal,
    BI__builtin_ia32_tconjtcmmimfp16ps,
    BI__builtin_ia32_tconjtcmmimfp16ps_internal,
    BI__builtin_ia32_tconjtfp16,
    BI__builtin_ia32_tconjtfp16_internal,
    BI__builtin_ia32_tcvtrowd2ps,
    BI__builtin_ia32_tcvtrowd2ps_internal,
    BI__builtin_ia32_tcvtrowps2bf16h,
    BI__builtin_ia32_tcvtrowps2bf16h_internal,
    BI__builtin_ia32_tcvtrowps2bf16l,
    BI__builtin_ia32_tcvtrowps2bf16l_internal,
    BI__builtin_ia32_tcvtrowps2phh,
    BI__builtin_ia32_tcvtrowps2phh_internal,
    BI__builtin_ia32_tcvtrowps2phl,
    BI__builtin_ia32_tcvtrowps2phl_internal,
    BI__builtin_ia32_tdpbf8ps,
    BI__builtin_ia32_tdpbf8ps_internal,
    BI__builtin_ia32_tdpbf16ps,
    BI__builtin_ia32_tdpbf16ps_internal,
    BI__builtin_ia32_tdpbhf8ps,
    BI__builtin_ia32_tdpbhf8ps_internal,
    BI__builtin_ia32_tdpbssd,
    BI__builtin_ia32_tdpbssd_internal,
    BI__builtin_ia32_tdpbsud,
    BI__builtin_ia32_tdpbsud_internal,
    BI__builtin_ia32_tdpbusd,
    BI__builtin_ia32_tdpbusd_internal,
    BI__builtin_ia32_tdpbuud,
    BI__builtin_ia32_tdpbuud_internal,
    BI__builtin_ia32_tdpfp16ps,
    BI__builtin_ia32_tdpfp16ps_internal,
    BI__builtin_ia32_tdphbf8ps,
    BI__builtin_ia32_tdphbf8ps_internal,
    BI__builtin_ia32_tdphf8ps,
    BI__builtin_ia32_tdphf8ps_internal,
    BI__builtin_ia32_testui,
    BI__builtin_ia32_tile_loadconfig,
    BI__builtin_ia32_tile_loadconfig_internal,
    BI__builtin_ia32_tile_storeconfig,
    BI__builtin_ia32_tileloadd64,
    BI__builtin_ia32_tileloadd64_internal,
    BI__builtin_ia32_tileloaddrs64,
    BI__builtin_ia32_tileloaddrs64_internal,
    BI__builtin_ia32_tileloaddrst164,
    BI__builtin_ia32_tileloaddrst164_internal,
    BI__builtin_ia32_tileloaddt164,
    BI__builtin_ia32_tileloaddt164_internal,
    BI__builtin_ia32_tilemovrow,
    BI__builtin_ia32_tilemovrow_internal,
    BI__builtin_ia32_tilerelease,
    BI__builtin_ia32_tilestored64,
    BI__builtin_ia32_tilestored64_internal,
    BI__builtin_ia32_tilezero,
    BI__builtin_ia32_tilezero_internal,
    BI__builtin_ia32_tmmultf32ps,
    BI__builtin_ia32_tmmultf32ps_internal,
    BI__builtin_ia32_ttcmmimfp16ps,
    BI__builtin_ia32_ttcmmimfp16ps_internal,
    BI__builtin_ia32_ttcmmrlfp16ps,
    BI__builtin_ia32_ttcmmrlfp16ps_internal,
    BI__builtin_ia32_ttdpbf16ps,
    BI__builtin_ia32_ttdpbf16ps_internal,
    BI__builtin_ia32_ttdpfp16ps,
    BI__builtin_ia32_ttdpfp16ps_internal,
    BI__builtin_ia32_ttmmultf32ps,
    BI__builtin_ia32_ttmmultf32ps_internal,
    BI__builtin_ia32_ttransposed,
    BI__builtin_ia32_ttransposed_internal,
    BI__builtin_ia32_tzcnt_u64,
    BI__builtin_ia32_urdmsr,
    BI__builtin_ia32_uwrmsr,
    BI__builtin_ia32_vcvtsd2si64,
    BI__builtin_ia32_vcvtsd2usi64,
    BI__builtin_ia32_vcvtsh2si64,
    BI__builtin_ia32_vcvtsh2usi64,
    BI__builtin_ia32_vcvtsi642sh,
    BI__builtin_ia32_vcvtss2si64,
    BI__builtin_ia32_vcvtss2usi64,
    BI__builtin_ia32_vcvttsd2si64,
    BI__builtin_ia32_vcvttsd2sis64,
    BI__builtin_ia32_vcvttsd2usi64,
    BI__builtin_ia32_vcvttsd2usis64,
    BI__builtin_ia32_vcvttsh2si64,
    BI__builtin_ia32_vcvttsh2usi64,
    BI__builtin_ia32_vcvttss2si64,
    BI__builtin_ia32_vcvttss2sis64,
    BI__builtin_ia32_vcvttss2usi64,
    BI__builtin_ia32_vcvttss2usis64,
    BI__builtin_ia32_vcvtusi642sh,
    BI__builtin_ia32_vec_ext_v4di,
    BI__builtin_ia32_vec_set_v2di,
    BI__builtin_ia32_vec_set_v4di,
    BI__builtin_ia32_vmovrsb128,
    BI__builtin_ia32_vmovrsb256,
    BI__builtin_ia32_vmovrsb512,
    BI__builtin_ia32_vmovrsd128,
    BI__builtin_ia32_vmovrsd256,
    BI__builtin_ia32_vmovrsd512,
    BI__builtin_ia32_vmovrsq128,
    BI__builtin_ia32_vmovrsq256,
    BI__builtin_ia32_vmovrsq512,
    BI__builtin_ia32_vmovrsw128,
    BI__builtin_ia32_vmovrsw256,
    BI__builtin_ia32_vmovrsw512,
    BI__builtin_ia32_wrfsbase32,
    BI__builtin_ia32_wrfsbase64,
    BI__builtin_ia32_wrgsbase32,
    BI__builtin_ia32_wrgsbase64,
    BI__builtin_ia32_writeeflags_u64,
    BI__builtin_ia32_wrssq,
    BI__builtin_ia32_wrussq,
    BI__builtin_ia32_xrstor64,
    BI__builtin_ia32_xrstors64,
    BI__builtin_ia32_xsave64,
    BI__builtin_ia32_xsavec64,
    BI__builtin_ia32_xsaveopt64,
    BI__builtin_ia32_xsaves64,

#endif // GET_BUILTIN_ENUMERATORS

#ifdef GET_BUILTIN_STR_TABLE

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif
static constexpr char BuiltinStringsStorage[] = 
  "\0"
  "_BitScanForward64\0"
  "UcUNi*ULLi\0"
  "nh\0"
  "_BitScanReverse64\0"
  "_InterlockedCompareExchange128\0"
  "UcLLiD*LLiLLiLLi*\0"
  "__faststorefence\0"
  "v\0"
  "__mulh\0"
  "LLiLLiLLi\0"
  "nch\0"
  "__shiftleft128\0"
  "ULLiULLiULLiUc\0"
  "__shiftright128\0"
  "__umulh\0"
  "ULLiULLiULLi\0"
  "_mul128\0"
  "LLiLLiLLiLLi*\0"
  "_umul128\0"
  "ULLiULLiULLiULLi*\0"
  "aadd64\0"
  "vv*SOi\0"
  "n\0"
  "raoint\0"
  "aand64\0"
  "addcarryx_u64\0"
  "UcUcUOiUOiUOi*\0"
  "nE\0"
  "aor64\0"
  "axor64\0"
  "bextr_u64\0"
  "UOiUOiUOi\0"
  "ncE\0"
  "bmi\0"
  "bextri_u64\0"
  "UOiUOiIUOi\0"
  "tbm\0"
  "bzhi_di\0"
  "bmi2\0"
  "clui\0"
  "uintr\0"
  "cmpccxadd32\0"
  "Siv*SiSiIi\0"
  "cmpccxadd\0"
  "cmpccxadd64\0"
  "SOiSOi*SOiSOiIi\0"
  "crc32di\0"
  "nc\0"
  "crc32\0"
  "cvtsd2si64\0"
  "OiV2d\0"
  "ncV:128:\0"
  "sse2\0"
  "cvtsi2sd64\0"
  "V2dV2dOiIi\0"
  "avx512f\0"
  "cvtsi2ss64\0"
  "V4fV4fOiIi\0"
  "cvtss2si64\0"
  "OiV4f\0"
  "sse\0"
  "cvttsd2si64\0"
  "cvttss2si64\0"
  "cvtusi2sd64\0"
  "V2dV2dUOiIi\0"
  "cvtusi2ss64\0"
  "V4fV4fUOiIi\0"
  "directstore_u64\0"
  "vULi*ULi\0"
  "movdiri\0"
  "fxrstor64\0"
  "vv*\0"
  "fxsr\0"
  "fxsave64\0"
  "incsspq\0"
  "vUOi\0"
  "shstk\0"
  "lwpins64\0"
  "UcUOiUiIUi\0"
  "lwp\0"
  "lwpval64\0"
  "vUOiUiIUi\0"
  "lzcnt_u64\0"
  "UOiUOi\0"
  "movnti64\0"
  "vOi*Oi\0"
  "movrsdi\0"
  "SOivC*\0"
  "movrs\0"
  "movrshi\0"
  "SsvC*\0"
  "movrsqi\0"
  "ScvC*\0"
  "movrssi\0"
  "SivC*\0"
  "pdep_di\0"
  "pext_di\0"
  "prefetchi\0"
  "vvC*Ui\0"
  "ptwrite64\0"
  "ptwrite\0"
  "rdfsbase32\0"
  "Ui\0"
  "fsgsbase\0"
  "rdfsbase64\0"
  "UOi\0"
  "rdgsbase32\0"
  "rdgsbase64\0"
  "rdrand64_step\0"
  "UiUOi*\0"
  "rdrnd\0"
  "rdseed64_step\0"
  "rdseed\0"
  "rdsspq\0"
  "readeflags_u64\0"
  "senduipi\0"
  "vUWi\0"
  "stui\0"
  "subborrow_u64\0"
  "t2rpntlvwz0\0"
  "vIUcvC*z\0"
  "amx-transpose\0"
  "t2rpntlvwz0_internal\0"
  "vUsUsUsV256i*V256i*vC*z\0"
  "t2rpntlvwz0rs\0"
  "amx-movrs,amx-transpose\0"
  "t2rpntlvwz0rs_internal\0"
  "t2rpntlvwz0rst1\0"
  "t2rpntlvwz0rst1_internal\0"
  "t2rpntlvwz0t1\0"
  "t2rpntlvwz0t1_internal\0"
  "t2rpntlvwz1\0"
  "t2rpntlvwz1_internal\0"
  "t2rpntlvwz1rs\0"
  "t2rpntlvwz1rs_internal\0"
  "t2rpntlvwz1rst1\0"
  "t2rpntlvwz1rst1_internal\0"
  "t2rpntlvwz1t1\0"
  "t2rpntlvwz1t1_internal\0"
  "tcmmimfp16ps\0"
  "vIUcIUcIUc\0"
  "amx-complex\0"
  "tcmmimfp16ps_internal\0"
  "V256iUsUsUsV256iV256iV256i\0"
  "tcmmrlfp16ps\0"
  "tcmmrlfp16ps_internal\0"
  "tconjtcmmimfp16ps\0"
  "amx-complex,amx-transpose\0"
  "tconjtcmmimfp16ps_internal\0"
  "tconjtfp16\0"
  "vIUcIUc\0"
  "tconjtfp16_internal\0"
  "V256iUsUsV256i\0"
  "tcvtrowd2ps\0"
  "V16fIUcUi\0"
  "amx-avx512,avx10.2-512\0"
  "tcvtrowd2ps_internal\0"
  "V16fUsUsV256iUi\0"
  "tcvtrowps2bf16h\0"
  "V32yIUcUi\0"
  "tcvtrowps2bf16h_internal\0"
  "V32yUsUsV256iUi\0"
  "tcvtrowps2bf16l\0"
  "tcvtrowps2bf16l_internal\0"
  "tcvtrowps2phh\0"
  "V32xIUcUi\0"
  "tcvtrowps2phh_internal\0"
  "V32xUsUsV256iUi\0"
  "tcvtrowps2phl\0"
  "tcvtrowps2phl_internal\0"
  "tdpbf8ps\0"
  "vIUcUIcUIc\0"
  "amx-fp8\0"
  "tdpbf8ps_internal\0"
  "tdpbf16ps\0"
  "amx-bf16\0"
  "tdpbf16ps_internal\0"
  "tdpbhf8ps\0"
  "tdpbhf8ps_internal\0"
  "tdpbssd\0"
  "amx-int8\0"
  "tdpbssd_internal\0"
  "tdpbsud\0"
  "tdpbsud_internal\0"
  "tdpbusd\0"
  "tdpbusd_internal\0"
  "tdpbuud\0"
  "tdpbuud_internal\0"
  "tdpfp16ps\0"
  "amx-fp16\0"
  "tdpfp16ps_internal\0"
  "tdphbf8ps\0"
  "tdphbf8ps_internal\0"
  "tdphf8ps\0"
  "tdphf8ps_internal\0"
  "testui\0"
  "Uc\0"
  "tile_loadconfig\0"
  "vvC*\0"
  "amx-tile\0"
  "tile_loadconfig_internal\0"
  "tile_storeconfig\0"
  "tileloadd64\0"
  "tileloadd64_internal\0"
  "V256iUsUsvC*z\0"
  "tileloaddrs64\0"
  "amx-movrs\0"
  "tileloaddrs64_internal\0"
  "tileloaddrst164\0"
  "tileloaddrst164_internal\0"
  "tileloaddt164\0"
  "tileloaddt164_internal\0"
  "tilemovrow\0"
  "V16iIUcUi\0"
  "tilemovrow_internal\0"
  "V16iUsUsV256iUi\0"
  "tilerelease\0"
  "tilestored64\0"
  "vIUcv*z\0"
  "tilestored64_internal\0"
  "vUsUsv*zV256i\0"
  "tilezero\0"
  "vUc\0"
  "tilezero_internal\0"
  "V256iUsUs\0"
  "tmmultf32ps\0"
  "amx-tf32\0"
  "tmmultf32ps_internal\0"
  "ttcmmimfp16ps\0"
  "ttcmmimfp16ps_internal\0"
  "ttcmmrlfp16ps\0"
  "ttcmmrlfp16ps_internal\0"
  "ttdpbf16ps\0"
  "amx-bf16,amx-transpose\0"
  "ttdpbf16ps_internal\0"
  "ttdpfp16ps\0"
  "amx-fp16,amx-transpose\0"
  "ttdpfp16ps_internal\0"
  "ttmmultf32ps\0"
  "amx-tf32,amx-transpose\0"
  "ttmmultf32ps_internal\0"
  "ttransposed\0"
  "ttransposed_internal\0"
  "tzcnt_u64\0"
  "urdmsr\0"
  "usermsr\0"
  "uwrmsr\0"
  "vUOiUOi\0"
  "vcvtsd2si64\0"
  "OiV2dIi\0"
  "vcvtsd2usi64\0"
  "UOiV2dIi\0"
  "vcvtsh2si64\0"
  "OiV8xIi\0"
  "avx512fp16\0"
  "vcvtsh2usi64\0"
  "UOiV8xIi\0"
  "vcvtsi642sh\0"
  "V8xV8xOiIi\0"
  "vcvtss2si64\0"
  "OiV4fIi\0"
  "vcvtss2usi64\0"
  "UOiV4fIi\0"
  "vcvttsd2si64\0"
  "vcvttsd2sis64\0"
  "avx10.2-256\0"
  "vcvttsd2usi64\0"
  "vcvttsd2usis64\0"
  "vcvttsh2si64\0"
  "vcvttsh2usi64\0"
  "vcvttss2si64\0"
  "vcvttss2sis64\0"
  "vcvttss2usi64\0"
  "vcvttss2usis64\0"
  "vcvtusi642sh\0"
  "V8xV8xUOiIi\0"
  "vec_ext_v4di\0"
  "OiV4OiIi\0"
  "ncV:256:\0"
  "avx\0"
  "vec_set_v2di\0"
  "V2OiV2OiOiIi\0"
  "sse4.1\0"
  "vec_set_v4di\0"
  "V4OiV4OiOiIi\0"
  "vmovrsb128\0"
  "V16cV16cC*\0"
  "nV:128:\0"
  "movrs,avx10.2-256\0"
  "vmovrsb256\0"
  "V32cV32cC*\0"
  "nV:256:\0"
  "vmovrsb512\0"
  "V64cV64cC*\0"
  "nV:512:\0"
  "movrs,avx10.2-512\0"
  "vmovrsd128\0"
  "V4iV4iC*\0"
  "vmovrsd256\0"
  "V8iV8iC*\0"
  "vmovrsd512\0"
  "V16iV16iC*\0"
  "vmovrsq128\0"
  "V2OiV2OiC*\0"
  "vmovrsq256\0"
  "V4OiV4OiC*\0"
  "vmovrsq512\0"
  "V8OiV8OiC*\0"
  "vmovrsw128\0"
  "V8sV8sC*\0"
  "vmovrsw256\0"
  "V16sV16sC*\0"
  "vmovrsw512\0"
  "V32sV32sC*\0"
  "wrfsbase32\0"
  "vUi\0"
  "wrfsbase64\0"
  "wrgsbase32\0"
  "wrgsbase64\0"
  "writeeflags_u64\0"
  "wrssq\0"
  "vUOiv*\0"
  "wrussq\0"
  "xrstor64\0"
  "vv*UOi\0"
  "xsave\0"
  "xrstors64\0"
  "xsaves\0"
  "xsave64\0"
  "xsavec64\0"
  "xsavec\0"
  "xsaveopt64\0"
  "xsaveopt\0"
  "xsaves64\0"
  ;
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

static constexpr llvm::StringTable BuiltinStrings =
    BuiltinStringsStorage;

#endif // GET_BUILTIN_STR_TABLE

#ifdef GET_BUILTIN_INFOS
    Builtin::Info{Builtin::Info::StrOffsets{1 /* _BitScanForward64 */, 19 /* UcUNi*ULLi */, 30 /* nh */, 0}, HeaderDesc::INTRIN_H, ALL_MS_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{33 /* _BitScanReverse64 */, 19 /* UcUNi*ULLi */, 30 /* nh */, 0}, HeaderDesc::INTRIN_H, ALL_MS_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{51 /* _InterlockedCompareExchange128 */, 82 /* UcLLiD*LLiLLiLLi* */, 30 /* nh */, 0}, HeaderDesc::INTRIN_H, ALL_MS_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{100 /* __faststorefence */, 117 /* v */, 30 /* nh */, 0}, HeaderDesc::INTRIN_H, ALL_MS_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{119 /* __mulh */, 126 /* LLiLLiLLi */, 136 /* nch */, 0}, HeaderDesc::INTRIN_H, ALL_MS_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{140 /* __shiftleft128 */, 155 /* ULLiULLiULLiUc */, 136 /* nch */, 0}, HeaderDesc::INTRIN_H, ALL_MS_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{170 /* __shiftright128 */, 155 /* ULLiULLiULLiUc */, 136 /* nch */, 0}, HeaderDesc::INTRIN_H, ALL_MS_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{186 /* __umulh */, 194 /* ULLiULLiULLi */, 136 /* nch */, 0}, HeaderDesc::INTRIN_H, ALL_MS_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{207 /* _mul128 */, 215 /* LLiLLiLLiLLi* */, 136 /* nch */, 0}, HeaderDesc::INTRIN_H, ALL_MS_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{229 /* _umul128 */, 238 /* ULLiULLiULLiULLi* */, 136 /* nch */, 0}, HeaderDesc::INTRIN_H, ALL_MS_LANGUAGES},

#endif // GET_BUILTIN_INFOS

#ifdef GET_BUILTIN_PREFIXED_INFOS
    Builtin::Info{Builtin::Info::StrOffsets{256 /* aadd64 */, 263 /* vv*SOi */, 270 /* n */, 272 /* raoint */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{279 /* aand64 */, 263 /* vv*SOi */, 270 /* n */, 272 /* raoint */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{286 /* addcarryx_u64 */, 300 /* UcUcUOiUOiUOi* */, 315 /* nE */, 0 /*  */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{318 /* aor64 */, 263 /* vv*SOi */, 270 /* n */, 272 /* raoint */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{324 /* axor64 */, 263 /* vv*SOi */, 270 /* n */, 272 /* raoint */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{331 /* bextr_u64 */, 341 /* UOiUOiUOi */, 351 /* ncE */, 355 /* bmi */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{359 /* bextri_u64 */, 370 /* UOiUOiIUOi */, 351 /* ncE */, 381 /* tbm */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{385 /* bzhi_di */, 341 /* UOiUOiUOi */, 351 /* ncE */, 393 /* bmi2 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{398 /* clui */, 117 /* v */, 270 /* n */, 403 /* uintr */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{409 /* cmpccxadd32 */, 421 /* Siv*SiSiIi */, 270 /* n */, 432 /* cmpccxadd */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{442 /* cmpccxadd64 */, 454 /* SOiSOi*SOiSOiIi */, 270 /* n */, 432 /* cmpccxadd */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{470 /* crc32di */, 341 /* UOiUOiUOi */, 478 /* nc */, 481 /* crc32 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{487 /* cvtsd2si64 */, 498 /* OiV2d */, 504 /* ncV:128: */, 513 /* sse2 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{518 /* cvtsi2sd64 */, 529 /* V2dV2dOiIi */, 504 /* ncV:128: */, 540 /* avx512f */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{548 /* cvtsi2ss64 */, 559 /* V4fV4fOiIi */, 504 /* ncV:128: */, 540 /* avx512f */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{570 /* cvtss2si64 */, 581 /* OiV4f */, 504 /* ncV:128: */, 587 /* sse */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{591 /* cvttsd2si64 */, 498 /* OiV2d */, 504 /* ncV:128: */, 513 /* sse2 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{603 /* cvttss2si64 */, 581 /* OiV4f */, 504 /* ncV:128: */, 587 /* sse */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{615 /* cvtusi2sd64 */, 627 /* V2dV2dUOiIi */, 504 /* ncV:128: */, 540 /* avx512f */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{639 /* cvtusi2ss64 */, 651 /* V4fV4fUOiIi */, 504 /* ncV:128: */, 540 /* avx512f */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{663 /* directstore_u64 */, 679 /* vULi*ULi */, 270 /* n */, 688 /* movdiri */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{696 /* fxrstor64 */, 706 /* vv* */, 270 /* n */, 710 /* fxsr */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{715 /* fxsave64 */, 706 /* vv* */, 270 /* n */, 710 /* fxsr */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{724 /* incsspq */, 732 /* vUOi */, 270 /* n */, 737 /* shstk */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{743 /* lwpins64 */, 752 /* UcUOiUiIUi */, 270 /* n */, 763 /* lwp */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{767 /* lwpval64 */, 776 /* vUOiUiIUi */, 270 /* n */, 763 /* lwp */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{786 /* lzcnt_u64 */, 796 /* UOiUOi */, 351 /* ncE */, 0 /*  */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{803 /* movnti64 */, 812 /* vOi*Oi */, 270 /* n */, 513 /* sse2 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{819 /* movrsdi */, 827 /* SOivC* */, 270 /* n */, 834 /* movrs */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{840 /* movrshi */, 848 /* SsvC* */, 270 /* n */, 834 /* movrs */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{854 /* movrsqi */, 862 /* ScvC* */, 270 /* n */, 834 /* movrs */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{868 /* movrssi */, 876 /* SivC* */, 270 /* n */, 834 /* movrs */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{882 /* pdep_di */, 341 /* UOiUOiUOi */, 351 /* ncE */, 393 /* bmi2 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{890 /* pext_di */, 341 /* UOiUOiUOi */, 351 /* ncE */, 393 /* bmi2 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{898 /* prefetchi */, 908 /* vvC*Ui */, 478 /* nc */, 898 /* prefetchi */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{915 /* ptwrite64 */, 732 /* vUOi */, 270 /* n */, 925 /* ptwrite */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{933 /* rdfsbase32 */, 944 /* Ui */, 270 /* n */, 947 /* fsgsbase */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{956 /* rdfsbase64 */, 967 /* UOi */, 270 /* n */, 947 /* fsgsbase */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{971 /* rdgsbase32 */, 944 /* Ui */, 270 /* n */, 947 /* fsgsbase */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{982 /* rdgsbase64 */, 967 /* UOi */, 270 /* n */, 947 /* fsgsbase */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{993 /* rdrand64_step */, 1007 /* UiUOi* */, 270 /* n */, 1014 /* rdrnd */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1020 /* rdseed64_step */, 1007 /* UiUOi* */, 270 /* n */, 1034 /* rdseed */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1041 /* rdsspq */, 796 /* UOiUOi */, 270 /* n */, 737 /* shstk */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1048 /* readeflags_u64 */, 967 /* UOi */, 270 /* n */, 0 /*  */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1063 /* senduipi */, 1072 /* vUWi */, 270 /* n */, 403 /* uintr */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1077 /* stui */, 117 /* v */, 270 /* n */, 403 /* uintr */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1082 /* subborrow_u64 */, 300 /* UcUcUOiUOiUOi* */, 315 /* nE */, 0 /*  */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1096 /* t2rpntlvwz0 */, 1108 /* vIUcvC*z */, 270 /* n */, 1117 /* amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1131 /* t2rpntlvwz0_internal */, 1152 /* vUsUsUsV256i*V256i*vC*z */, 270 /* n */, 1117 /* amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1176 /* t2rpntlvwz0rs */, 1108 /* vIUcvC*z */, 270 /* n */, 1190 /* amx-movrs,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1214 /* t2rpntlvwz0rs_internal */, 1152 /* vUsUsUsV256i*V256i*vC*z */, 270 /* n */, 1190 /* amx-movrs,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1237 /* t2rpntlvwz0rst1 */, 1108 /* vIUcvC*z */, 270 /* n */, 1190 /* amx-movrs,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1253 /* t2rpntlvwz0rst1_internal */, 1152 /* vUsUsUsV256i*V256i*vC*z */, 270 /* n */, 1190 /* amx-movrs,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1278 /* t2rpntlvwz0t1 */, 1108 /* vIUcvC*z */, 270 /* n */, 1117 /* amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1292 /* t2rpntlvwz0t1_internal */, 1152 /* vUsUsUsV256i*V256i*vC*z */, 270 /* n */, 1117 /* amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1315 /* t2rpntlvwz1 */, 1108 /* vIUcvC*z */, 270 /* n */, 1117 /* amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1327 /* t2rpntlvwz1_internal */, 1152 /* vUsUsUsV256i*V256i*vC*z */, 270 /* n */, 1117 /* amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1348 /* t2rpntlvwz1rs */, 1108 /* vIUcvC*z */, 270 /* n */, 1190 /* amx-movrs,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1362 /* t2rpntlvwz1rs_internal */, 1152 /* vUsUsUsV256i*V256i*vC*z */, 270 /* n */, 1190 /* amx-movrs,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1385 /* t2rpntlvwz1rst1 */, 1108 /* vIUcvC*z */, 270 /* n */, 1190 /* amx-movrs,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1401 /* t2rpntlvwz1rst1_internal */, 1152 /* vUsUsUsV256i*V256i*vC*z */, 270 /* n */, 1190 /* amx-movrs,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1426 /* t2rpntlvwz1t1 */, 1108 /* vIUcvC*z */, 270 /* n */, 1117 /* amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1440 /* t2rpntlvwz1t1_internal */, 1152 /* vUsUsUsV256i*V256i*vC*z */, 270 /* n */, 1117 /* amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1463 /* tcmmimfp16ps */, 1476 /* vIUcIUcIUc */, 270 /* n */, 1487 /* amx-complex */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1499 /* tcmmimfp16ps_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 1487 /* amx-complex */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1548 /* tcmmrlfp16ps */, 1476 /* vIUcIUcIUc */, 270 /* n */, 1487 /* amx-complex */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1561 /* tcmmrlfp16ps_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 1487 /* amx-complex */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1583 /* tconjtcmmimfp16ps */, 1476 /* vIUcIUcIUc */, 270 /* n */, 1601 /* amx-complex,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1627 /* tconjtcmmimfp16ps_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 1601 /* amx-complex,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1654 /* tconjtfp16 */, 1665 /* vIUcIUc */, 270 /* n */, 1601 /* amx-complex,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1673 /* tconjtfp16_internal */, 1693 /* V256iUsUsV256i */, 270 /* n */, 1601 /* amx-complex,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1708 /* tcvtrowd2ps */, 1720 /* V16fIUcUi */, 270 /* n */, 1730 /* amx-avx512,avx10.2-512 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1753 /* tcvtrowd2ps_internal */, 1774 /* V16fUsUsV256iUi */, 270 /* n */, 1730 /* amx-avx512,avx10.2-512 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1790 /* tcvtrowps2bf16h */, 1806 /* V32yIUcUi */, 270 /* n */, 1730 /* amx-avx512,avx10.2-512 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1816 /* tcvtrowps2bf16h_internal */, 1841 /* V32yUsUsV256iUi */, 270 /* n */, 1730 /* amx-avx512,avx10.2-512 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1857 /* tcvtrowps2bf16l */, 1806 /* V32yIUcUi */, 270 /* n */, 1730 /* amx-avx512,avx10.2-512 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1873 /* tcvtrowps2bf16l_internal */, 1841 /* V32yUsUsV256iUi */, 270 /* n */, 1730 /* amx-avx512,avx10.2-512 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1898 /* tcvtrowps2phh */, 1912 /* V32xIUcUi */, 270 /* n */, 1730 /* amx-avx512,avx10.2-512 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1922 /* tcvtrowps2phh_internal */, 1945 /* V32xUsUsV256iUi */, 270 /* n */, 1730 /* amx-avx512,avx10.2-512 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1961 /* tcvtrowps2phl */, 1912 /* V32xIUcUi */, 270 /* n */, 1730 /* amx-avx512,avx10.2-512 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1975 /* tcvtrowps2phl_internal */, 1945 /* V32xUsUsV256iUi */, 270 /* n */, 1730 /* amx-avx512,avx10.2-512 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{1998 /* tdpbf8ps */, 2007 /* vIUcUIcUIc */, 270 /* n */, 2018 /* amx-fp8 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2026 /* tdpbf8ps_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 2018 /* amx-fp8 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2044 /* tdpbf16ps */, 1476 /* vIUcIUcIUc */, 270 /* n */, 2054 /* amx-bf16 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2063 /* tdpbf16ps_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 2054 /* amx-bf16 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2082 /* tdpbhf8ps */, 2007 /* vIUcUIcUIc */, 270 /* n */, 2018 /* amx-fp8 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2092 /* tdpbhf8ps_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 2018 /* amx-fp8 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2111 /* tdpbssd */, 1476 /* vIUcIUcIUc */, 270 /* n */, 2119 /* amx-int8 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2128 /* tdpbssd_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 2119 /* amx-int8 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2145 /* tdpbsud */, 1476 /* vIUcIUcIUc */, 270 /* n */, 2119 /* amx-int8 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2153 /* tdpbsud_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 2119 /* amx-int8 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2170 /* tdpbusd */, 1476 /* vIUcIUcIUc */, 270 /* n */, 2119 /* amx-int8 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2178 /* tdpbusd_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 2119 /* amx-int8 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2195 /* tdpbuud */, 1476 /* vIUcIUcIUc */, 270 /* n */, 2119 /* amx-int8 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2203 /* tdpbuud_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 2119 /* amx-int8 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2220 /* tdpfp16ps */, 1476 /* vIUcIUcIUc */, 270 /* n */, 2230 /* amx-fp16 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2239 /* tdpfp16ps_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 2230 /* amx-fp16 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2258 /* tdphbf8ps */, 2007 /* vIUcUIcUIc */, 270 /* n */, 2018 /* amx-fp8 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2268 /* tdphbf8ps_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 2018 /* amx-fp8 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2287 /* tdphf8ps */, 2007 /* vIUcUIcUIc */, 270 /* n */, 2018 /* amx-fp8 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2296 /* tdphf8ps_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 2018 /* amx-fp8 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2314 /* testui */, 2321 /* Uc */, 270 /* n */, 403 /* uintr */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2324 /* tile_loadconfig */, 2340 /* vvC* */, 270 /* n */, 2345 /* amx-tile */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2354 /* tile_loadconfig_internal */, 2340 /* vvC* */, 270 /* n */, 2345 /* amx-tile */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2379 /* tile_storeconfig */, 2340 /* vvC* */, 270 /* n */, 2345 /* amx-tile */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2396 /* tileloadd64 */, 1108 /* vIUcvC*z */, 270 /* n */, 2345 /* amx-tile */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2408 /* tileloadd64_internal */, 2429 /* V256iUsUsvC*z */, 270 /* n */, 2345 /* amx-tile */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2443 /* tileloaddrs64 */, 1108 /* vIUcvC*z */, 270 /* n */, 2457 /* amx-movrs */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2467 /* tileloaddrs64_internal */, 2429 /* V256iUsUsvC*z */, 270 /* n */, 2457 /* amx-movrs */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2490 /* tileloaddrst164 */, 1108 /* vIUcvC*z */, 270 /* n */, 2457 /* amx-movrs */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2506 /* tileloaddrst164_internal */, 2429 /* V256iUsUsvC*z */, 270 /* n */, 2457 /* amx-movrs */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2531 /* tileloaddt164 */, 1108 /* vIUcvC*z */, 270 /* n */, 2345 /* amx-tile */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2545 /* tileloaddt164_internal */, 2429 /* V256iUsUsvC*z */, 270 /* n */, 2345 /* amx-tile */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2568 /* tilemovrow */, 2579 /* V16iIUcUi */, 270 /* n */, 1730 /* amx-avx512,avx10.2-512 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2589 /* tilemovrow_internal */, 2609 /* V16iUsUsV256iUi */, 270 /* n */, 1730 /* amx-avx512,avx10.2-512 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2625 /* tilerelease */, 117 /* v */, 270 /* n */, 2345 /* amx-tile */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2637 /* tilestored64 */, 2650 /* vIUcv*z */, 270 /* n */, 2345 /* amx-tile */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2658 /* tilestored64_internal */, 2680 /* vUsUsv*zV256i */, 270 /* n */, 2345 /* amx-tile */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2694 /* tilezero */, 2703 /* vUc */, 270 /* n */, 2345 /* amx-tile */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2707 /* tilezero_internal */, 2725 /* V256iUsUs */, 270 /* n */, 2345 /* amx-tile */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2735 /* tmmultf32ps */, 1476 /* vIUcIUcIUc */, 270 /* n */, 2747 /* amx-tf32 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2756 /* tmmultf32ps_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 2747 /* amx-tf32 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2777 /* ttcmmimfp16ps */, 1476 /* vIUcIUcIUc */, 270 /* n */, 1601 /* amx-complex,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2791 /* ttcmmimfp16ps_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 1601 /* amx-complex,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2814 /* ttcmmrlfp16ps */, 1476 /* vIUcIUcIUc */, 270 /* n */, 1601 /* amx-complex,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2828 /* ttcmmrlfp16ps_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 1601 /* amx-complex,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2851 /* ttdpbf16ps */, 1476 /* vIUcIUcIUc */, 270 /* n */, 2862 /* amx-bf16,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2885 /* ttdpbf16ps_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 2862 /* amx-bf16,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2905 /* ttdpfp16ps */, 1476 /* vIUcIUcIUc */, 270 /* n */, 2916 /* amx-fp16,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2939 /* ttdpfp16ps_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 2916 /* amx-fp16,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2959 /* ttmmultf32ps */, 1476 /* vIUcIUcIUc */, 270 /* n */, 2972 /* amx-tf32,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{2995 /* ttmmultf32ps_internal */, 1521 /* V256iUsUsUsV256iV256iV256i */, 270 /* n */, 2972 /* amx-tf32,amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3017 /* ttransposed */, 1665 /* vIUcIUc */, 270 /* n */, 1117 /* amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3029 /* ttransposed_internal */, 1693 /* V256iUsUsV256i */, 270 /* n */, 1117 /* amx-transpose */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3050 /* tzcnt_u64 */, 796 /* UOiUOi */, 351 /* ncE */, 0 /*  */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3060 /* urdmsr */, 796 /* UOiUOi */, 270 /* n */, 3067 /* usermsr */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3075 /* uwrmsr */, 3082 /* vUOiUOi */, 270 /* n */, 3067 /* usermsr */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3090 /* vcvtsd2si64 */, 3102 /* OiV2dIi */, 504 /* ncV:128: */, 540 /* avx512f */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3110 /* vcvtsd2usi64 */, 3123 /* UOiV2dIi */, 504 /* ncV:128: */, 540 /* avx512f */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3132 /* vcvtsh2si64 */, 3144 /* OiV8xIi */, 504 /* ncV:128: */, 3152 /* avx512fp16 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3163 /* vcvtsh2usi64 */, 3176 /* UOiV8xIi */, 504 /* ncV:128: */, 3152 /* avx512fp16 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3185 /* vcvtsi642sh */, 3197 /* V8xV8xOiIi */, 504 /* ncV:128: */, 3152 /* avx512fp16 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3208 /* vcvtss2si64 */, 3220 /* OiV4fIi */, 504 /* ncV:128: */, 540 /* avx512f */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3228 /* vcvtss2usi64 */, 3241 /* UOiV4fIi */, 504 /* ncV:128: */, 540 /* avx512f */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3250 /* vcvttsd2si64 */, 3102 /* OiV2dIi */, 504 /* ncV:128: */, 540 /* avx512f */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3263 /* vcvttsd2sis64 */, 3102 /* OiV2dIi */, 504 /* ncV:128: */, 3277 /* avx10.2-256 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3289 /* vcvttsd2usi64 */, 3123 /* UOiV2dIi */, 504 /* ncV:128: */, 540 /* avx512f */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3303 /* vcvttsd2usis64 */, 3123 /* UOiV2dIi */, 504 /* ncV:128: */, 3277 /* avx10.2-256 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3318 /* vcvttsh2si64 */, 3144 /* OiV8xIi */, 504 /* ncV:128: */, 3152 /* avx512fp16 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3331 /* vcvttsh2usi64 */, 3176 /* UOiV8xIi */, 504 /* ncV:128: */, 3152 /* avx512fp16 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3345 /* vcvttss2si64 */, 3220 /* OiV4fIi */, 504 /* ncV:128: */, 540 /* avx512f */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3358 /* vcvttss2sis64 */, 3220 /* OiV4fIi */, 504 /* ncV:128: */, 3277 /* avx10.2-256 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3372 /* vcvttss2usi64 */, 3241 /* UOiV4fIi */, 504 /* ncV:128: */, 540 /* avx512f */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3386 /* vcvttss2usis64 */, 3241 /* UOiV4fIi */, 504 /* ncV:128: */, 3277 /* avx10.2-256 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3401 /* vcvtusi642sh */, 3414 /* V8xV8xUOiIi */, 504 /* ncV:128: */, 3152 /* avx512fp16 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3426 /* vec_ext_v4di */, 3439 /* OiV4OiIi */, 3448 /* ncV:256: */, 3457 /* avx */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3461 /* vec_set_v2di */, 3474 /* V2OiV2OiOiIi */, 504 /* ncV:128: */, 3487 /* sse4.1 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3494 /* vec_set_v4di */, 3507 /* V4OiV4OiOiIi */, 3448 /* ncV:256: */, 3457 /* avx */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3520 /* vmovrsb128 */, 3531 /* V16cV16cC* */, 3542 /* nV:128: */, 3550 /* movrs,avx10.2-256 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3568 /* vmovrsb256 */, 3579 /* V32cV32cC* */, 3590 /* nV:256: */, 3550 /* movrs,avx10.2-256 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3598 /* vmovrsb512 */, 3609 /* V64cV64cC* */, 3620 /* nV:512: */, 3628 /* movrs,avx10.2-512 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3646 /* vmovrsd128 */, 3657 /* V4iV4iC* */, 3542 /* nV:128: */, 3550 /* movrs,avx10.2-256 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3666 /* vmovrsd256 */, 3677 /* V8iV8iC* */, 3590 /* nV:256: */, 3550 /* movrs,avx10.2-256 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3686 /* vmovrsd512 */, 3697 /* V16iV16iC* */, 3620 /* nV:512: */, 3628 /* movrs,avx10.2-512 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3708 /* vmovrsq128 */, 3719 /* V2OiV2OiC* */, 3542 /* nV:128: */, 3550 /* movrs,avx10.2-256 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3730 /* vmovrsq256 */, 3741 /* V4OiV4OiC* */, 3590 /* nV:256: */, 3550 /* movrs,avx10.2-256 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3752 /* vmovrsq512 */, 3763 /* V8OiV8OiC* */, 3620 /* nV:512: */, 3628 /* movrs,avx10.2-512 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3774 /* vmovrsw128 */, 3785 /* V8sV8sC* */, 3542 /* nV:128: */, 3550 /* movrs,avx10.2-256 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3794 /* vmovrsw256 */, 3805 /* V16sV16sC* */, 3590 /* nV:256: */, 3550 /* movrs,avx10.2-256 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3816 /* vmovrsw512 */, 3827 /* V32sV32sC* */, 3620 /* nV:512: */, 3628 /* movrs,avx10.2-512 */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3838 /* wrfsbase32 */, 3849 /* vUi */, 270 /* n */, 947 /* fsgsbase */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3853 /* wrfsbase64 */, 732 /* vUOi */, 270 /* n */, 947 /* fsgsbase */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3864 /* wrgsbase32 */, 3849 /* vUi */, 270 /* n */, 947 /* fsgsbase */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3875 /* wrgsbase64 */, 732 /* vUOi */, 270 /* n */, 947 /* fsgsbase */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3886 /* writeeflags_u64 */, 732 /* vUOi */, 270 /* n */, 0 /*  */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3902 /* wrssq */, 3908 /* vUOiv* */, 270 /* n */, 737 /* shstk */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3915 /* wrussq */, 3908 /* vUOiv* */, 270 /* n */, 737 /* shstk */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3922 /* xrstor64 */, 3931 /* vv*UOi */, 270 /* n */, 3938 /* xsave */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3944 /* xrstors64 */, 3931 /* vv*UOi */, 270 /* n */, 3954 /* xsaves */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3961 /* xsave64 */, 3931 /* vv*UOi */, 270 /* n */, 3938 /* xsave */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3969 /* xsavec64 */, 3931 /* vv*UOi */, 270 /* n */, 3978 /* xsavec */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{3985 /* xsaveopt64 */, 3931 /* vv*UOi */, 270 /* n */, 3996 /* xsaveopt */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{4005 /* xsaves64 */, 3931 /* vv*UOi */, 270 /* n */, 3954 /* xsaves */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},

#endif // GET_BUILTIN_PREFIXED_INFOS

#ifdef ATOMIC_BUILTIN

#endif // ATOMIC_BUILTIN
#undef ATOMIC_BUILTIN
