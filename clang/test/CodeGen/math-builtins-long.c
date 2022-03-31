// RUN: %clang_cc1 -triple x86_64-unknown-unknown -w -S -o - -emit-llvm %s \
// RUN:   -fmath-errno | FileCheck %s -check-prefix=F80
// RUN: %clang_cc1 -triple ppc64le-unknown-unknown -w -S -o - -emit-llvm %s \
// RUN:   -fmath-errno | FileCheck %s -check-prefix=PPC
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -mlong-double-128 -w -S \
// RUN:   -o - -emit-llvm %s -fmath-errno | FileCheck %s -check-prefix=X86F128
// RUN: %clang_cc1 -triple ppc64le-unknown-unknown -mabi=ieeelongdouble -w -S \
// RUN:   -o - -emit-llvm %s -fmath-errno | FileCheck %s -check-prefix=PPCF128

void bar(long double);

void foo(long double f, long double *l, int *i, const char *c) {
  // F80: call x86_fp80 @fmodl(x86_fp80 noundef %{{.+}}, x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @fmodl(ppc_fp128 noundef %{{.+}}, ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @fmodl(fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  // PPCF128: call fp128 @fmodf128(fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  __builtin_fmodl(f,f);

  // F80: call x86_fp80 @atan2l(x86_fp80 noundef %{{.+}}, x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @atan2l(ppc_fp128 noundef %{{.+}}, ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @atan2l(fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  // PPCF128: call fp128 @atan2f128(fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  __builtin_atan2l(f,f);

  // F80: call x86_fp80 @llvm.copysign.f80(x86_fp80 %{{.+}}, x86_fp80 %{{.+}})
  // PPC: call ppc_fp128 @llvm.copysign.ppcf128(ppc_fp128 %{{.+}}, ppc_fp128 %{{.+}})
  // X86F128: call fp128 @llvm.copysign.f128(fp128 %{{.+}}, fp128 %{{.+}})
  // PPCF128: call fp128 @llvm.copysign.f128(fp128 %{{.+}}, fp128 %{{.+}})
  __builtin_copysignl(f,f);

  // F80: call x86_fp80 @llvm.fabs.f80(x86_fp80 %{{.+}})
  // PPC: call ppc_fp128 @llvm.fabs.ppcf128(ppc_fp128 %{{.+}})
  // X86F128: call fp128 @llvm.fabs.f128(fp128 %{{.+}})
  // PPCF128: call fp128 @llvm.fabs.f128(fp128 %{{.+}})
  __builtin_fabsl(f);

  // F80: call x86_fp80 @frexpl(x86_fp80 noundef %{{.+}}, i32* noundef %{{.+}})
  // PPC: call ppc_fp128 @frexpl(ppc_fp128 noundef %{{.+}}, i32* noundef %{{.+}})
  // X86F128: call fp128 @frexpl(fp128 noundef %{{.+}}, i32* noundef %{{.+}})
  // PPCF128: call fp128 @frexpf128(fp128 noundef %{{.+}}, i32* noundef %{{.+}})
  __builtin_frexpl(f,i);

  // F80: store x86_fp80 0xK7FFF8000000000000000, x86_fp80*
  // PPC: store ppc_fp128 0xM7FF00000000000000000000000000000, ppc_fp128*
  // X86F128: store fp128 0xL00000000000000007FFF000000000000, fp128*
  // PPCF128: store fp128 0xL00000000000000007FFF000000000000, fp128*
  *l = __builtin_huge_vall();

  // F80: store x86_fp80 0xK7FFF8000000000000000, x86_fp80*
  // PPC: store ppc_fp128 0xM7FF00000000000000000000000000000, ppc_fp128*
  // X86F128: store fp128 0xL00000000000000007FFF000000000000, fp128*
  // PPCF128: store fp128 0xL00000000000000007FFF000000000000, fp128*
  *l = __builtin_infl();

  // F80: call x86_fp80 @ldexpl(x86_fp80 noundef %{{.+}}, i32 noundef %{{.+}})
  // PPC: call ppc_fp128 @ldexpl(ppc_fp128 noundef %{{.+}}, {{(signext)?.+}})
  // X86F128: call fp128 @ldexpl(fp128 noundef %{{.+}}, {{(signext)?.+}})
  // PPCF128: call fp128 @ldexpf128(fp128 noundef %{{.+}}, {{(signext)?.+}})
  __builtin_ldexpl(f,f);

  // F80: call x86_fp80 @modfl(x86_fp80 noundef %{{.+}}, x86_fp80* noundef %{{.+}})
  // PPC: call ppc_fp128 @modfl(ppc_fp128 noundef %{{.+}}, ppc_fp128* noundef %{{.+}})
  // X86F128: call fp128 @modfl(fp128 noundef %{{.+}}, fp128* noundef %{{.+}})
  // PPCF128: call fp128 @modff128(fp128 noundef %{{.+}}, fp128* noundef %{{.+}})
  __builtin_modfl(f,l);

  // F80: call x86_fp80 @nanl(i8* noundef %{{.+}})
  // PPC: call ppc_fp128 @nanl(i8* noundef %{{.+}})
  // X86F128: call fp128 @nanl(i8* noundef %{{.+}})
  // PPCF128: call fp128 @nanf128(i8* noundef %{{.+}})
  __builtin_nanl(c);

  // F80: call x86_fp80 @nansl(i8* noundef %{{.+}})
  // PPC: call ppc_fp128 @nansl(i8* noundef %{{.+}})
  // X86F128: call fp128 @nansl(i8* noundef %{{.+}})
  // PPCF128: call fp128 @nansf128(i8* noundef %{{.+}})
  __builtin_nansl(c);

  // F80: call x86_fp80 @powl(x86_fp80 noundef %{{.+}}, x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @powl(ppc_fp128 noundef %{{.+}}, ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @powl(fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  // PPCF128: call fp128 @powf128(fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  __builtin_powl(f,f);

  // F80: call x86_fp80 @acosl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @acosl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @acosl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @acosf128(fp128 noundef %{{.+}})
  __builtin_acosl(f);

  // F80: call x86_fp80 @acoshl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @acoshl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @acoshl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @acoshf128(fp128 noundef %{{.+}})
  __builtin_acoshl(f);

  // F80: call x86_fp80 @asinl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @asinl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @asinl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @asinf128(fp128 noundef %{{.+}})
  __builtin_asinl(f);

  // F80: call x86_fp80 @asinhl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @asinhl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @asinhl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @asinhf128(fp128 noundef %{{.+}})
  __builtin_asinhl(f);

  // F80: call x86_fp80 @atanl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @atanl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @atanl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @atanf128(fp128 noundef %{{.+}})
  __builtin_atanl(f);

  // F80: call x86_fp80 @atanhl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @atanhl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @atanhl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @atanhf128(fp128 noundef %{{.+}})
  __builtin_atanhl(f);

  // F80: call x86_fp80 @cbrtl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @cbrtl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @cbrtl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @cbrtf128(fp128 noundef %{{.+}})
  __builtin_cbrtl(f);

  // F80: call x86_fp80 @llvm.ceil.f80(x86_fp80 %{{.+}})
  // PPC: call ppc_fp128 @llvm.ceil.ppcf128(ppc_fp128 %{{.+}})
  // X86F128: call fp128 @llvm.ceil.f128(fp128 %{{.+}})
  // PPCF128: call fp128 @llvm.ceil.f128(fp128 %{{.+}})
  __builtin_ceill(f);

  // F80: call x86_fp80 @cosl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @cosl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @cosl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @cosf128(fp128 noundef %{{.+}})
  __builtin_cosl(f);

  // F80: call x86_fp80 @coshl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @coshl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @coshl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @coshf128(fp128 noundef %{{.+}})
  __builtin_coshl(f);

  // F80: call x86_fp80 @llvm.floor.f80(x86_fp80 %{{.+}})
  // PPC: call ppc_fp128 @llvm.floor.ppcf128(ppc_fp128 %{{.+}})
  // X86F128: call fp128 @llvm.floor.f128(fp128 %{{.+}})
  // PPCF128: call fp128 @llvm.floor.f128(fp128 %{{.+}})
  __builtin_floorl(f);

  // F80: call x86_fp80 @llvm.maxnum.f80(x86_fp80 %{{.+}}, x86_fp80 %{{.+}})
  // PPC: call ppc_fp128 @llvm.maxnum.ppcf128(ppc_fp128 %{{.+}}, ppc_fp128 %{{.+}})
  // X86F128: call fp128 @llvm.maxnum.f128(fp128 %{{.+}}, fp128 %{{.+}})
  // PPCF128: call fp128 @llvm.maxnum.f128(fp128 %{{.+}}, fp128 %{{.+}})
  __builtin_fmaxl(f,f);

  // F80: call x86_fp80 @llvm.minnum.f80(x86_fp80 %{{.+}}, x86_fp80 %{{.+}})
  // PPC: call ppc_fp128 @llvm.minnum.ppcf128(ppc_fp128 %{{.+}}, ppc_fp128 %{{.+}})
  // X86F128: call fp128 @llvm.minnum.f128(fp128 %{{.+}}, fp128 %{{.+}})
  // PPCF128: call fp128 @llvm.minnum.f128(fp128 %{{.+}}, fp128 %{{.+}})
  __builtin_fminl(f,f);

  // F80: call x86_fp80 @llvm.nearbyint.f80(x86_fp80 %{{.+}})
  // PPC: call ppc_fp128 @llvm.nearbyint.ppcf128(ppc_fp128 %{{.+}})
  // X86F128: call fp128 @llvm.nearbyint.f128(fp128 %{{.+}})
  // PPCF128: call fp128 @llvm.nearbyint.f128(fp128 %{{.+}})
  __builtin_nearbyintl(f);

  // F80: call x86_fp80 @llvm.trunc.f80(x86_fp80 %{{.+}})
  // PPC: call ppc_fp128 @llvm.trunc.ppcf128(ppc_fp128  %{{.+}})
  // X86F128: call fp128 @llvm.trunc.f128(fp128 %{{.+}})
  // PPCF128: call fp128 @llvm.trunc.f128(fp128 %{{.+}})
  __builtin_truncl(f);

  // F80: call x86_fp80 @llvm.rint.f80(x86_fp80 %{{.+}})
  // PPC: call ppc_fp128 @llvm.rint.ppcf128(ppc_fp128 %{{.+}})
  // X86F128: call fp128 @llvm.rint.f128(fp128 %{{.+}})
  // PPCF128: call fp128 @llvm.rint.f128(fp128 %{{.+}})
  __builtin_rintl(f);

  // F80: call x86_fp80 @llvm.round.f80(x86_fp80 %{{.+}})
  // PPC: call ppc_fp128 @llvm.round.ppcf128(ppc_fp128 %{{.+}})
  // X86F128: call fp128 @llvm.round.f128(fp128 %{{.+}})
  // PPCF128: call fp128 @llvm.round.f128(fp128 %{{.+}})
  __builtin_roundl(f);

  // F80: call x86_fp80 @erfl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @erfl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @erfl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @erff128(fp128 noundef %{{.+}})
  __builtin_erfl(f);

  // F80: call x86_fp80 @erfcl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @erfcl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @erfcl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @erfcf128(fp128 noundef %{{.+}})
  __builtin_erfcl(f);

  // F80: call x86_fp80 @expl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @expl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @expl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @expf128(fp128 noundef %{{.+}})
  __builtin_expl(f);

  // F80: call x86_fp80 @exp2l(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @exp2l(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @exp2l(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @exp2f128(fp128 noundef %{{.+}})
  __builtin_exp2l(f);

  // F80: call x86_fp80 @expm1l(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @expm1l(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @expm1l(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @expm1f128(fp128 noundef %{{.+}})
  __builtin_expm1l(f);

  // F80: call x86_fp80 @fdiml(x86_fp80 noundef %{{.+}}, x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @fdiml(ppc_fp128 noundef %{{.+}}, ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @fdiml(fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  // PPCF128: call fp128 @fdimf128(fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  __builtin_fdiml(f,f);

  // F80: call x86_fp80 @fmal(x86_fp80 noundef %{{.+}}, x86_fp80 noundef %{{.+}}, x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @fmal(ppc_fp128 noundef %{{.+}}, ppc_fp128 noundef %{{.+}}, ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @fmal(fp128 noundef %{{.+}}, fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  // PPCF128: call fp128 @fmaf128(fp128 noundef %{{.+}}, fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  __builtin_fmal(f,f,f);

  // F80: call x86_fp80 @hypotl(x86_fp80 noundef %{{.+}}, x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @hypotl(ppc_fp128 noundef %{{.+}}, ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @hypotl(fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  // PPCF128: call fp128 @hypotf128(fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  __builtin_hypotl(f,f);

  // F80: call i32 @ilogbl(x86_fp80 noundef %{{.+}})
  // PPC: call {{(i32)|(signext i32)}} @ilogbl(ppc_fp128 noundef %{{.+}})
  // X86F128: call {{(i32)|(signext i32)}} @ilogbl(fp128 noundef %{{.+}})
  // PPCF128: call {{(i32)|(signext i32)}} @ilogbf128(fp128 noundef %{{.+}})
  __builtin_ilogbl(f);

  // F80: call x86_fp80 @lgammal(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @lgammal(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @lgammal(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @lgammaf128(fp128 noundef %{{.+}})
  __builtin_lgammal(f);

  // F80: call i64 @llrintl(x86_fp80 noundef %{{.+}})
  // PPC: call i64 @llrintl(ppc_fp128 noundef %{{.+}})
  // X86F128: call i64 @llrintl(fp128 noundef %{{.+}})
  // PPCF128: call i64 @llrintf128(fp128 noundef %{{.+}})
  __builtin_llrintl(f);

  // F80: call i64 @llroundl(x86_fp80 noundef %{{.+}})
  // PPC: call i64 @llroundl(ppc_fp128 noundef %{{.+}})
  // X86F128: call i64 @llroundl(fp128 noundef %{{.+}})
  // PPCF128: call i64 @llroundf128(fp128 noundef %{{.+}})
  __builtin_llroundl(f);

  // F80: call x86_fp80 @logl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @logl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @logl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @logf128(fp128 noundef %{{.+}})
  __builtin_logl(f);

  // F80: call x86_fp80 @log10l(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @log10l(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @log10l(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @log10f128(fp128 noundef %{{.+}})
  __builtin_log10l(f);

  // F80: call x86_fp80 @log1pl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @log1pl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @log1pl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @log1pf128(fp128 noundef %{{.+}})
  __builtin_log1pl(f);

  // F80: call x86_fp80 @log2l(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @log2l(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @log2l(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @log2f128(fp128 noundef %{{.+}})
  __builtin_log2l(f);

  // F80: call x86_fp80 @logbl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @logbl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @logbl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @logbf128(fp128 noundef %{{.+}})
  __builtin_logbl(f);

  // F80: call i64 @lrintl(x86_fp80 noundef %{{.+}})
  // PPC: call i64 @lrintl(ppc_fp128 noundef %{{.+}})
  // X86F128: call i64 @lrintl(fp128 noundef %{{.+}})
  // PPCF128: call i64 @lrintf128(fp128 noundef %{{.+}})
  __builtin_lrintl(f);

  // F80: call i64 @lroundl(x86_fp80 noundef %{{.+}})
  // PPC: call i64 @lroundl(ppc_fp128 noundef %{{.+}})
  // X86F128: call i64 @lroundl(fp128 noundef %{{.+}})
  // PPCF128: call i64 @lroundf128(fp128 noundef %{{.+}})
  __builtin_lroundl(f);

  // F80: call x86_fp80 @nextafterl(x86_fp80 noundef %{{.+}}, x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @nextafterl(ppc_fp128 noundef %{{.+}}, ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @nextafterl(fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  // PPCF128: call fp128 @nextafterf128(fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  __builtin_nextafterl(f,f);

  // F80: call x86_fp80 @nexttowardl(x86_fp80 noundef %{{.+}}, x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @nexttowardl(ppc_fp128 noundef %{{.+}}, ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @nexttowardl(fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  // PPCF128: call fp128 @__nexttowardieee128(fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  __builtin_nexttowardl(f,f);

  // F80: call x86_fp80 @remainderl(x86_fp80 noundef %{{.+}}, x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @remainderl(ppc_fp128 noundef %{{.+}}, ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @remainderl(fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  // PPCF128: call fp128 @remainderf128(fp128 noundef %{{.+}}, fp128 noundef %{{.+}})
  __builtin_remainderl(f,f);

  // F80: call x86_fp80 @remquol(x86_fp80 noundef %{{.+}}, x86_fp80 noundef %{{.+}}, i32* noundef %{{.+}})
  // PPC: call ppc_fp128 @remquol(ppc_fp128 noundef %{{.+}}, ppc_fp128 noundef %{{.+}}, i32* noundef %{{.+}})
  // X86F128: call fp128 @remquol(fp128 noundef %{{.+}}, fp128 noundef %{{.+}}, i32* noundef %{{.+}})
  // PPCF128: call fp128 @remquof128(fp128 noundef %{{.+}}, fp128 noundef %{{.+}}, i32* noundef %{{.+}})
  __builtin_remquol(f,f,i);

  // F80: call x86_fp80 @scalblnl(x86_fp80 noundef %{{.+}}, i64 noundef %{{.+}})
  // PPC: call ppc_fp128 @scalblnl(ppc_fp128 noundef %{{.+}}, i64 noundef %{{.+}})
  // X86F128: call fp128 @scalblnl(fp128 noundef %{{.+}}, i64 noundef %{{.+}})
  // PPCF128: call fp128 @scalblnf128(fp128 noundef %{{.+}}, i64 noundef %{{.+}})
  __builtin_scalblnl(f,f);

  // F80: call x86_fp80 @scalbnl(x86_fp80 noundef %{{.+}}, i32 noundef %{{.+}})
  // PPC: call ppc_fp128 @scalbnl(ppc_fp128 noundef %{{.+}}, {{(signext)?.+}})
  // X86F128: call fp128 @scalbnl(fp128 noundef %{{.+}}, {{(signext)?.+}})
  // PPCF128: call fp128 @scalbnf128(fp128 noundef %{{.+}}, {{(signext)?.+}})
  __builtin_scalbnl(f,f);

  // F80: call x86_fp80 @sinl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @sinl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @sinl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @sinf128(fp128 noundef %{{.+}})
  __builtin_sinl(f);

  // F80: call x86_fp80 @sinhl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @sinhl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @sinhl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @sinhf128(fp128 noundef %{{.+}})
  __builtin_sinhl(f);

  // F80: call x86_fp80 @sqrtl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @sqrtl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @sqrtl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @sqrtf128(fp128 noundef %{{.+}})
  __builtin_sqrtl(f);

  // F80: call x86_fp80 @tanl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @tanl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @tanl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @tanf128(fp128 noundef %{{.+}})
  __builtin_tanl(f);

  // F80: call x86_fp80 @tanhl(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @tanhl(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @tanhl(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @tanhf128(fp128 noundef %{{.+}})
  __builtin_tanhl(f);

  // F80: call x86_fp80 @tgammal(x86_fp80 noundef %{{.+}})
  // PPC: call ppc_fp128 @tgammal(ppc_fp128 noundef %{{.+}})
  // X86F128: call fp128 @tgammal(fp128 noundef %{{.+}})
  // PPCF128: call fp128 @tgammaf128(fp128 noundef %{{.+}})
  __builtin_tgammal(f);
}
