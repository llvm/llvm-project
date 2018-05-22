// RUN: %clang_cc1 -DUSE_64 -triple x86_64-unknown-unknown -target-feature +fxsr -target-feature +avx -target-feature +xsaveopt -target-feature +xsaves -target-feature +xsavec -target-feature +mwaitx -target-feature +clzero -target-feature +shstk -target-feature +wbnoinvd -target-feature +cldemote -emit-llvm -o %t %s
// RUN: %clang_cc1 -DUSE_ALL -triple x86_64-unknown-unknown -target-feature +fxsr -target-feature +avx -target-feature +xsaveopt -target-feature +xsaves -target-feature +xsavec -target-feature +mwaitx -target-feature +shstk -target-feature +clzero -target-feature +wbnoinvd -target-feature +cldemote -fsyntax-only -o %t %s

#ifdef USE_ALL
#define USE_3DNOW
#define USE_64
#define USE_SSE4
#endif

// 64-bit
typedef char V8c __attribute__((vector_size(8 * sizeof(char))));
typedef signed short V4s __attribute__((vector_size(8)));
typedef signed int V2i __attribute__((vector_size(8)));
typedef signed long long V1LLi __attribute__((vector_size(8)));

typedef float V2f __attribute__((vector_size(8)));

// 128-bit
typedef char V16c __attribute__((vector_size(16)));
typedef signed short V8s __attribute__((vector_size(16)));
typedef signed int V4i __attribute__((vector_size(16)));
typedef signed long long V2LLi __attribute__((vector_size(16)));

typedef float V4f __attribute__((vector_size(16)));
typedef double V2d __attribute__((vector_size(16)));

// 256-bit
typedef char V32c __attribute__((vector_size(32)));
typedef signed int V8i __attribute__((vector_size(32)));
typedef signed long long V4LLi __attribute__((vector_size(32)));

typedef double V4d __attribute__((vector_size(32)));
typedef float  V8f __attribute__((vector_size(32)));

void f0() {
  signed char         tmp_c;
//  unsigned char       tmp_Uc;
  signed short        tmp_s;
#ifdef USE_ALL
  unsigned short      tmp_Us;
#endif
  signed int          tmp_i;
  unsigned int        tmp_Ui;
  signed long long    tmp_LLi;
  unsigned long long  tmp_ULLi;
  float               tmp_f;
  double              tmp_d;

  void*          tmp_vp;
  const void*    tmp_vCp;
  char*          tmp_cp; 
  const char*    tmp_cCp; 
  int*           tmp_ip;
  float*         tmp_fp;
  const float*   tmp_fCp;
  double*        tmp_dp;
  const double*  tmp_dCp;
  long long*     tmp_LLip;

#define imm_i 32
#define imm_i_0_2 0
#define imm_i_0_4 3
#define imm_i_0_8 7
#define imm_i_0_16 15
  // Check this.
#define imm_i_0_256 0

  V2i*   tmp_V2ip;
  V1LLi* tmp_V1LLip;
  V2LLi* tmp_V2LLip;

  // 64-bit
  V8c    tmp_V8c;
  V4s    tmp_V4s;
  V2i    tmp_V2i;
  V1LLi  tmp_V1LLi;
#ifdef USE_3DNOW
  V2f    tmp_V2f;
#endif

  // 128-bit
  V16c   tmp_V16c;
  V8s    tmp_V8s;
  V4i    tmp_V4i;
  V2LLi  tmp_V2LLi;
  V4f    tmp_V4f;
  V2d    tmp_V2d;
  V2d*   tmp_V2dp;
  V4f*   tmp_V4fp;
  const V2d* tmp_V2dCp;
  const V4f* tmp_V4fCp;

  // 256-bit
  V32c   tmp_V32c;
  V4d    tmp_V4d;
  V8f    tmp_V8f;
  V4LLi  tmp_V4LLi;
  V8i    tmp_V8i;
  V4LLi* tmp_V4LLip;
  V4d*   tmp_V4dp;
  V8f*   tmp_V8fp;
  const V4d* tmp_V4dCp;
  const V8f* tmp_V8fCp;

  tmp_V2LLi = __builtin_ia32_undef128();
  tmp_V4LLi = __builtin_ia32_undef256();

  tmp_i = __builtin_ia32_comieq(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_comilt(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_comile(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_comigt(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_comige(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_comineq(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_ucomieq(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_ucomilt(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_ucomile(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_ucomigt(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_ucomige(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_ucomineq(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_comisdeq(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_comisdlt(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_comisdle(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_comisdgt(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_comisdge(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_comisdneq(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_ucomisdeq(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_ucomisdlt(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_ucomisdle(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_ucomisdgt(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_ucomisdge(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_ucomisdneq(tmp_V2d, tmp_V2d);
  tmp_V4f = __builtin_ia32_cmpps(tmp_V4f, tmp_V4f, 0);
  tmp_V4f = __builtin_ia32_cmpps(tmp_V4f, tmp_V4f, 1);
  tmp_V4f = __builtin_ia32_cmpps(tmp_V4f, tmp_V4f, 2);
  tmp_V4f = __builtin_ia32_cmpps(tmp_V4f, tmp_V4f, 3);
  tmp_V4f = __builtin_ia32_cmpps(tmp_V4f, tmp_V4f, 4);
  tmp_V4f = __builtin_ia32_cmpps(tmp_V4f, tmp_V4f, 5);
  tmp_V4f = __builtin_ia32_cmpps(tmp_V4f, tmp_V4f, 6);
  tmp_V4f = __builtin_ia32_cmpps(tmp_V4f, tmp_V4f, 7);
  tmp_V4f = __builtin_ia32_cmpss(tmp_V4f, tmp_V4f, 0);
  tmp_V4f = __builtin_ia32_cmpss(tmp_V4f, tmp_V4f, 1);
  tmp_V4f = __builtin_ia32_cmpss(tmp_V4f, tmp_V4f, 2);
  tmp_V4f = __builtin_ia32_cmpss(tmp_V4f, tmp_V4f, 3);
  tmp_V4f = __builtin_ia32_cmpss(tmp_V4f, tmp_V4f, 4);
  tmp_V4f = __builtin_ia32_cmpss(tmp_V4f, tmp_V4f, 5);
  tmp_V4f = __builtin_ia32_cmpss(tmp_V4f, tmp_V4f, 6);
  tmp_V4f = __builtin_ia32_cmpss(tmp_V4f, tmp_V4f, 7);
  tmp_V4f = __builtin_ia32_minps(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_maxps(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_minss(tmp_V4f, tmp_V4f);
  tmp_V4f = __builtin_ia32_maxss(tmp_V4f, tmp_V4f);

  tmp_V8c = __builtin_ia32_paddsb(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_paddsw(tmp_V4s, tmp_V4s);
  tmp_V8c = __builtin_ia32_psubsb(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_psubsw(tmp_V4s, tmp_V4s);
  tmp_V8c = __builtin_ia32_paddusb(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_paddusw(tmp_V4s, tmp_V4s);
  tmp_V8c = __builtin_ia32_psubusb(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_psubusw(tmp_V4s, tmp_V4s);
  tmp_V4s = __builtin_ia32_pmulhw(tmp_V4s, tmp_V4s);
  tmp_V4s = __builtin_ia32_pmulhuw(tmp_V4s, tmp_V4s);
  tmp_V8c = __builtin_ia32_pcmpeqb(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_pcmpeqw(tmp_V4s, tmp_V4s);
  tmp_V2i = __builtin_ia32_pcmpeqd(tmp_V2i, tmp_V2i);
  tmp_V8c = __builtin_ia32_pcmpgtb(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_pcmpgtw(tmp_V4s, tmp_V4s);
  tmp_V2i = __builtin_ia32_pcmpgtd(tmp_V2i, tmp_V2i);
  tmp_V8c = __builtin_ia32_pmaxub(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_pmaxsw(tmp_V4s, tmp_V4s);
  tmp_V8c = __builtin_ia32_pminub(tmp_V8c, tmp_V8c);
  tmp_V4s = __builtin_ia32_pminsw(tmp_V4s, tmp_V4s);
  tmp_V2d = __builtin_ia32_cmppd(tmp_V2d, tmp_V2d, 0);
  tmp_V2d = __builtin_ia32_cmppd(tmp_V2d, tmp_V2d, 1);
  tmp_V2d = __builtin_ia32_cmppd(tmp_V2d, tmp_V2d, 2);
  tmp_V2d = __builtin_ia32_cmppd(tmp_V2d, tmp_V2d, 3);
  tmp_V2d = __builtin_ia32_cmppd(tmp_V2d, tmp_V2d, 4);
  tmp_V2d = __builtin_ia32_cmppd(tmp_V2d, tmp_V2d, 5);
  tmp_V2d = __builtin_ia32_cmppd(tmp_V2d, tmp_V2d, 6);
  tmp_V2d = __builtin_ia32_cmppd(tmp_V2d, tmp_V2d, 7);
  tmp_V2d = __builtin_ia32_cmpsd(tmp_V2d, tmp_V2d, 0);
  tmp_V2d = __builtin_ia32_cmpsd(tmp_V2d, tmp_V2d, 1);
  tmp_V2d = __builtin_ia32_cmpsd(tmp_V2d, tmp_V2d, 2);
  tmp_V2d = __builtin_ia32_cmpsd(tmp_V2d, tmp_V2d, 3);
  tmp_V2d = __builtin_ia32_cmpsd(tmp_V2d, tmp_V2d, 4);
  tmp_V2d = __builtin_ia32_cmpsd(tmp_V2d, tmp_V2d, 5);
  tmp_V2d = __builtin_ia32_cmpsd(tmp_V2d, tmp_V2d, 6);
  tmp_V2d = __builtin_ia32_cmpsd(tmp_V2d, tmp_V2d, 7);
  tmp_V2d = __builtin_ia32_minpd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_maxpd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_minsd(tmp_V2d, tmp_V2d);
  tmp_V2d = __builtin_ia32_maxsd(tmp_V2d, tmp_V2d);
  tmp_V16c = __builtin_ia32_paddsb128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_paddsw128(tmp_V8s, tmp_V8s);
  tmp_V16c = __builtin_ia32_psubsb128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_psubsw128(tmp_V8s, tmp_V8s);
  tmp_V16c = __builtin_ia32_paddusb128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_paddusw128(tmp_V8s, tmp_V8s);
  tmp_V16c = __builtin_ia32_psubusb128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_psubusw128(tmp_V8s, tmp_V8s);
  tmp_V8s = __builtin_ia32_pmulhw128(tmp_V8s, tmp_V8s);
  tmp_V16c = __builtin_ia32_pmaxub128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_pmaxsw128(tmp_V8s, tmp_V8s);
  tmp_V16c = __builtin_ia32_pminub128(tmp_V16c, tmp_V16c);
  tmp_V8s = __builtin_ia32_pminsw128(tmp_V8s, tmp_V8s);
  tmp_V8s = __builtin_ia32_packsswb128(tmp_V8s, tmp_V8s);
  tmp_V4i = __builtin_ia32_packssdw128(tmp_V4i, tmp_V4i);
  tmp_V8s = __builtin_ia32_packuswb128(tmp_V8s, tmp_V8s);
  tmp_V8s = __builtin_ia32_pmulhuw128(tmp_V8s, tmp_V8s);
  tmp_V4f = __builtin_ia32_addsubps(tmp_V4f, tmp_V4f);
  tmp_V2d = __builtin_ia32_addsubpd(tmp_V2d, tmp_V2d);
  tmp_V4f = __builtin_ia32_haddps(tmp_V4f, tmp_V4f);
  tmp_V2d = __builtin_ia32_haddpd(tmp_V2d, tmp_V2d);
  tmp_V4f = __builtin_ia32_hsubps(tmp_V4f, tmp_V4f);
  tmp_V2d = __builtin_ia32_hsubpd(tmp_V2d, tmp_V2d);
  tmp_V8s = __builtin_ia32_phaddw128(tmp_V8s, tmp_V8s);
  tmp_V4s = __builtin_ia32_phaddw(tmp_V4s, tmp_V4s);
  tmp_V4i = __builtin_ia32_phaddd128(tmp_V4i, tmp_V4i);
  tmp_V2i = __builtin_ia32_phaddd(tmp_V2i, tmp_V2i);
  tmp_V8s = __builtin_ia32_phaddsw128(tmp_V8s, tmp_V8s);
  tmp_V4s = __builtin_ia32_phaddsw(tmp_V4s, tmp_V4s);
  tmp_V8s = __builtin_ia32_phsubw128(tmp_V8s, tmp_V8s);
  tmp_V4s = __builtin_ia32_phsubw(tmp_V4s, tmp_V4s);
  tmp_V4i = __builtin_ia32_phsubd128(tmp_V4i, tmp_V4i);
  tmp_V2i = __builtin_ia32_phsubd(tmp_V2i, tmp_V2i);
  tmp_V8s = __builtin_ia32_phsubsw128(tmp_V8s, tmp_V8s);
  tmp_V4s = __builtin_ia32_phsubsw(tmp_V4s, tmp_V4s);
  tmp_V16c = __builtin_ia32_pmaddubsw128(tmp_V16c, tmp_V16c);
  tmp_V8c = __builtin_ia32_pmaddubsw(tmp_V8c, tmp_V8c);
  tmp_V8s = __builtin_ia32_pmulhrsw128(tmp_V8s, tmp_V8s);
  tmp_V4s = __builtin_ia32_pmulhrsw(tmp_V4s, tmp_V4s);
  tmp_V16c = __builtin_ia32_pshufb128(tmp_V16c, tmp_V16c);
  tmp_V8c = __builtin_ia32_pshufb(tmp_V8c, tmp_V8c);
  tmp_V16c = __builtin_ia32_psignb128(tmp_V16c, tmp_V16c);
  tmp_V8c = __builtin_ia32_psignb(tmp_V8c, tmp_V8c);
  tmp_V8s = __builtin_ia32_psignw128(tmp_V8s, tmp_V8s);
  tmp_V4s = __builtin_ia32_psignw(tmp_V4s, tmp_V4s);
  tmp_V4i = __builtin_ia32_psignd128(tmp_V4i, tmp_V4i);
  tmp_V2i = __builtin_ia32_psignd(tmp_V2i, tmp_V2i);
  tmp_V16c = __builtin_ia32_pabsb128(tmp_V16c);
  tmp_V8c = __builtin_ia32_pabsb(tmp_V8c);
  tmp_V8s = __builtin_ia32_pabsw128(tmp_V8s);
  tmp_V4s = __builtin_ia32_pabsw(tmp_V4s);
  tmp_V4i = __builtin_ia32_pabsd128(tmp_V4i);
  tmp_V2i = __builtin_ia32_pabsd(tmp_V2i);
  tmp_V4s = __builtin_ia32_psllw(tmp_V4s, tmp_V1LLi);
  tmp_V2i = __builtin_ia32_pslld(tmp_V2i, tmp_V1LLi);
  tmp_V1LLi = __builtin_ia32_psllq(tmp_V1LLi, tmp_V1LLi);
  tmp_V4s = __builtin_ia32_psrlw(tmp_V4s, tmp_V1LLi);
  tmp_V2i = __builtin_ia32_psrld(tmp_V2i, tmp_V1LLi);
  tmp_V1LLi = __builtin_ia32_psrlq(tmp_V1LLi, tmp_V1LLi);
  tmp_V4s = __builtin_ia32_psraw(tmp_V4s, tmp_V1LLi);
  tmp_V2i = __builtin_ia32_psrad(tmp_V2i, tmp_V1LLi);
  tmp_V2i = __builtin_ia32_pmaddwd(tmp_V4s, tmp_V4s);
  tmp_V8c = __builtin_ia32_packsswb(tmp_V4s, tmp_V4s);
  tmp_V4s = __builtin_ia32_packssdw(tmp_V2i, tmp_V2i);
  tmp_V8c = __builtin_ia32_packuswb(tmp_V4s, tmp_V4s);
  tmp_i = __builtin_ia32_vec_ext_v2si(tmp_V2i, 0);

  __builtin_ia32_incsspd(tmp_Ui);
  __builtin_ia32_incsspq(tmp_ULLi);
  tmp_Ui = __builtin_ia32_rdsspd(tmp_Ui);
  tmp_ULLi = __builtin_ia32_rdsspq(tmp_ULLi);
  __builtin_ia32_saveprevssp();
  __builtin_ia32_rstorssp(tmp_vp);
  __builtin_ia32_wrssd(tmp_Ui, tmp_vp);
  __builtin_ia32_wrssq(tmp_ULLi, tmp_vp);
  __builtin_ia32_wrussd(tmp_Ui, tmp_vp);
  __builtin_ia32_wrussq(tmp_ULLi, tmp_vp);
  __builtin_ia32_setssbsy();
  __builtin_ia32_clrssbsy(tmp_vp);

  (void) __builtin_ia32_ldmxcsr(tmp_Ui);
  (void) _mm_setcsr(tmp_Ui);
  tmp_Ui = __builtin_ia32_stmxcsr();
  tmp_Ui = _mm_getcsr();
  (void)__builtin_ia32_fxsave(tmp_vp);
  (void)__builtin_ia32_fxsave64(tmp_vp);
  (void)__builtin_ia32_fxrstor(tmp_vp);
  (void)__builtin_ia32_fxrstor64(tmp_vp);

  (void)__builtin_ia32_xsave(tmp_vp, tmp_ULLi);
  (void)__builtin_ia32_xsave64(tmp_vp, tmp_ULLi);
  (void)__builtin_ia32_xrstor(tmp_vp, tmp_ULLi);
  (void)__builtin_ia32_xrstor64(tmp_vp, tmp_ULLi);
  (void)__builtin_ia32_xsaveopt(tmp_vp, tmp_ULLi);
  (void)__builtin_ia32_xsaveopt64(tmp_vp, tmp_ULLi);
  (void)__builtin_ia32_xrstors(tmp_vp, tmp_ULLi);
  (void)__builtin_ia32_xrstors64(tmp_vp, tmp_ULLi);
  (void)__builtin_ia32_xsavec(tmp_vp, tmp_ULLi);
  (void)__builtin_ia32_xsavec64(tmp_vp, tmp_ULLi);
  (void)__builtin_ia32_xsaves(tmp_vp, tmp_ULLi);
  (void)__builtin_ia32_xsaves64(tmp_vp, tmp_ULLi);

  (void) __builtin_ia32_monitorx(tmp_vp, tmp_Ui, tmp_Ui);
  (void) __builtin_ia32_mwaitx(tmp_Ui, tmp_Ui, tmp_Ui);
  (void) __builtin_ia32_clzero(tmp_vp);
  (void) __builtin_ia32_cldemote(tmp_vp);

  tmp_V4f = __builtin_ia32_cvtpi2ps(tmp_V4f, tmp_V2i);
  tmp_V2i = __builtin_ia32_cvtps2pi(tmp_V4f);
  tmp_i = __builtin_ia32_cvtss2si(tmp_V4f);
  tmp_i = __builtin_ia32_cvttss2si(tmp_V4f);

  tmp_i = __builtin_ia32_rdtsc();
  tmp_i = __rdtsc();
  tmp_i = __builtin_ia32_rdtscp(&tmp_Ui);
  tmp_LLi = __builtin_ia32_rdpmc(tmp_i);
  __builtin_ia32_wbnoinvd();
#ifdef USE_64
  tmp_LLi = __builtin_ia32_cvtss2si64(tmp_V4f);
  tmp_LLi = __builtin_ia32_cvttss2si64(tmp_V4f);
#endif
  tmp_V2i = __builtin_ia32_cvttps2pi(tmp_V4f);
  (void) __builtin_ia32_maskmovq(tmp_V8c, tmp_V8c, tmp_cp);
  (void) __builtin_ia32_storehps(tmp_V2ip, tmp_V4f);
  (void) __builtin_ia32_storelps(tmp_V2ip, tmp_V4f);
  tmp_i = __builtin_ia32_movmskps(tmp_V4f);
  tmp_i = __builtin_ia32_pmovmskb(tmp_V8c);
  (void) __builtin_ia32_movntq(tmp_V1LLip, tmp_V1LLi);
  (void) __builtin_ia32_sfence();
  (void) _mm_sfence();

  tmp_V4s = __builtin_ia32_psadbw(tmp_V8c, tmp_V8c);
  tmp_V4f = __builtin_ia32_rcpps(tmp_V4f);
  tmp_V4f = __builtin_ia32_rcpss(tmp_V4f);
  tmp_V4f = __builtin_ia32_rsqrtps(tmp_V4f);
  tmp_V4f = __builtin_ia32_rsqrtss(tmp_V4f);
  tmp_V4f = __builtin_ia32_sqrtps(tmp_V4f);
  tmp_V4f = __builtin_ia32_sqrtss(tmp_V4f);
  (void) __builtin_ia32_maskmovdqu(tmp_V16c, tmp_V16c, tmp_cp);
  tmp_i = __builtin_ia32_movmskpd(tmp_V2d);
  tmp_i = __builtin_ia32_pmovmskb128(tmp_V16c);
  (void) __builtin_ia32_movnti(tmp_ip, tmp_i);
#ifdef USE_64
  (void) __builtin_ia32_movnti64(tmp_LLip, tmp_LLi);
#endif
  tmp_V2LLi = __builtin_ia32_psadbw128(tmp_V16c, tmp_V16c);
  tmp_V2d = __builtin_ia32_sqrtpd(tmp_V2d);
  tmp_V2d = __builtin_ia32_sqrtsd(tmp_V2d);
  tmp_V2LLi = __builtin_ia32_cvtpd2dq(tmp_V2d);
  tmp_V2i = __builtin_ia32_cvtpd2pi(tmp_V2d);
  tmp_V4f = __builtin_ia32_cvtpd2ps(tmp_V2d);
  tmp_V4i = __builtin_ia32_cvttpd2dq(tmp_V2d);
  tmp_V2i = __builtin_ia32_cvttpd2pi(tmp_V2d);
  tmp_V2d = __builtin_ia32_cvtpi2pd(tmp_V2i);
  tmp_i = __builtin_ia32_cvtsd2si(tmp_V2d);
  tmp_i = __builtin_ia32_cvttsd2si(tmp_V2d);
  tmp_V4f = __builtin_ia32_cvtsd2ss(tmp_V4f, tmp_V2d);
#ifdef USE_64
  tmp_LLi = __builtin_ia32_cvtsd2si64(tmp_V2d);
  tmp_LLi = __builtin_ia32_cvttsd2si64(tmp_V2d);
#endif
  tmp_V4i = __builtin_ia32_cvtps2dq(tmp_V4f);
  tmp_V4i = __builtin_ia32_cvttps2dq(tmp_V4f);
  (void) __builtin_ia32_clflush(tmp_vCp);
  (void) _mm_clflush(tmp_vCp);
  (void) __builtin_ia32_lfence();
  (void) _mm_lfence();
  (void) __builtin_ia32_mfence();
  (void) _mm_mfence();
  (void) __builtin_ia32_pause();
  (void) _mm_pause();
  tmp_V4s = __builtin_ia32_psllwi(tmp_V4s, tmp_i);
  tmp_V2i = __builtin_ia32_pslldi(tmp_V2i, tmp_i);
  tmp_V1LLi = __builtin_ia32_psllqi(tmp_V1LLi, tmp_i);
  tmp_V4s = __builtin_ia32_psrawi(tmp_V4s, tmp_i);
  tmp_V2i = __builtin_ia32_psradi(tmp_V2i, tmp_i);
  tmp_V4s = __builtin_ia32_psrlwi(tmp_V4s, tmp_i);
  tmp_V2i = __builtin_ia32_psrldi(tmp_V2i, tmp_i);
  tmp_V1LLi = __builtin_ia32_psrlqi(tmp_V1LLi, tmp_i);
  tmp_V1LLi = __builtin_ia32_pmuludq(tmp_V2i, tmp_V2i);
  tmp_V2LLi = __builtin_ia32_pmuludq128(tmp_V4i, tmp_V4i);
  tmp_V8s = __builtin_ia32_psraw128(tmp_V8s, tmp_V8s);
  tmp_V4i = __builtin_ia32_psrad128(tmp_V4i, tmp_V4i);
  tmp_V8s = __builtin_ia32_psrlw128(tmp_V8s, tmp_V8s);
  tmp_V4i = __builtin_ia32_psrld128(tmp_V4i, tmp_V4i);
  tmp_V2LLi = __builtin_ia32_psrlq128(tmp_V2LLi, tmp_V2LLi);
  tmp_V8s = __builtin_ia32_psllw128(tmp_V8s, tmp_V8s);
  tmp_V4i = __builtin_ia32_pslld128(tmp_V4i, tmp_V4i);
  tmp_V2LLi = __builtin_ia32_psllq128(tmp_V2LLi, tmp_V2LLi);
  tmp_V8s = __builtin_ia32_psllwi128(tmp_V8s, tmp_i);
  tmp_V4i = __builtin_ia32_pslldi128(tmp_V4i, tmp_i);
  tmp_V2LLi = __builtin_ia32_psllqi128(tmp_V2LLi, tmp_i);
  tmp_V8s = __builtin_ia32_psrlwi128(tmp_V8s, tmp_i);
  tmp_V4i = __builtin_ia32_psrldi128(tmp_V4i, tmp_i);
  tmp_V2LLi = __builtin_ia32_psrlqi128(tmp_V2LLi, tmp_i);
  tmp_V8s = __builtin_ia32_psrawi128(tmp_V8s, tmp_i);
  tmp_V4i = __builtin_ia32_psradi128(tmp_V4i, tmp_i);
  tmp_V8s = __builtin_ia32_pmaddwd128(tmp_V8s, tmp_V8s);
  (void) __builtin_ia32_monitor(tmp_vp, tmp_Ui, tmp_Ui);
  (void) __builtin_ia32_mwait(tmp_Ui, tmp_Ui);
  tmp_V16c = __builtin_ia32_lddqu(tmp_cCp);
  tmp_V2LLi = __builtin_ia32_palignr128(tmp_V2LLi, tmp_V2LLi, imm_i);
  tmp_V1LLi = __builtin_ia32_palignr(tmp_V1LLi, tmp_V1LLi, imm_i);
#ifdef USE_SSE4
  tmp_V16c = __builtin_ia32_pblendvb128(tmp_V16c, tmp_V16c, tmp_V16c);
  tmp_V2d = __builtin_ia32_blendvpd(tmp_V2d, tmp_V2d, tmp_V2d);
  tmp_V4f = __builtin_ia32_blendvps(tmp_V4f, tmp_V4f, tmp_V4f);
  tmp_V8s = __builtin_ia32_packusdw128(tmp_V4i, tmp_V4i);
  tmp_V16c = __builtin_ia32_pmaxsb128(tmp_V16c, tmp_V16c);
  tmp_V4i = __builtin_ia32_pmaxsd128(tmp_V4i, tmp_V4i);
  tmp_V4i = __builtin_ia32_pmaxud128(tmp_V4i, tmp_V4i);
  tmp_V8s = __builtin_ia32_pmaxuw128(tmp_V8s, tmp_V8s);
  tmp_V16c = __builtin_ia32_pminsb128(tmp_V16c, tmp_V16c);
  tmp_V4i = __builtin_ia32_pminsd128(tmp_V4i, tmp_V4i);
  tmp_V4i = __builtin_ia32_pminud128(tmp_V4i, tmp_V4i);
  tmp_V8s = __builtin_ia32_pminuw128(tmp_V8s, tmp_V8s);
  tmp_V2LLi = __builtin_ia32_pmuldq128(tmp_V4i, tmp_V4i);
  tmp_V4f = __builtin_ia32_roundps(tmp_V4f, imm_i_0_16);
  tmp_V4f = __builtin_ia32_roundss(tmp_V4f, tmp_V4f, imm_i_0_16);
  tmp_V2d = __builtin_ia32_roundsd(tmp_V2d, tmp_V2d, imm_i_0_16);
  tmp_V2d = __builtin_ia32_roundpd(tmp_V2d, imm_i_0_16);
  tmp_V4f = __builtin_ia32_insertps128(tmp_V4f, tmp_V4f, imm_i_0_256);
#endif

  tmp_V4d = __builtin_ia32_addsubpd256(tmp_V4d, tmp_V4d);
  tmp_V8f = __builtin_ia32_addsubps256(tmp_V8f, tmp_V8f);
  tmp_V4d = __builtin_ia32_haddpd256(tmp_V4d, tmp_V4d);
  tmp_V8f = __builtin_ia32_hsubps256(tmp_V8f, tmp_V8f);
  tmp_V4d = __builtin_ia32_hsubpd256(tmp_V4d, tmp_V4d);
  tmp_V8f = __builtin_ia32_haddps256(tmp_V8f, tmp_V8f);
  tmp_V4d = __builtin_ia32_maxpd256(tmp_V4d, tmp_V4d);
  tmp_V8f = __builtin_ia32_maxps256(tmp_V8f, tmp_V8f);
  tmp_V4d = __builtin_ia32_minpd256(tmp_V4d, tmp_V4d);
  tmp_V8f = __builtin_ia32_minps256(tmp_V8f, tmp_V8f);
  tmp_V2d = __builtin_ia32_vpermilvarpd(tmp_V2d, tmp_V2LLi);
  tmp_V4f = __builtin_ia32_vpermilvarps(tmp_V4f, tmp_V4i);
  tmp_V4d = __builtin_ia32_vpermilvarpd256(tmp_V4d, tmp_V4LLi);
  tmp_V8f = __builtin_ia32_vpermilvarps256(tmp_V8f, tmp_V8i);
  tmp_V4d = __builtin_ia32_blendvpd256(tmp_V4d, tmp_V4d, tmp_V4d);
  tmp_V8f = __builtin_ia32_blendvps256(tmp_V8f, tmp_V8f, tmp_V8f);
  tmp_V8f = __builtin_ia32_dpps256(tmp_V8f, tmp_V8f, 0x7);
  tmp_V4d = __builtin_ia32_cmppd256(tmp_V4d, tmp_V4d, 0);
  tmp_V8f = __builtin_ia32_cmpps256(tmp_V8f, tmp_V8f, 0);
  tmp_V4f = __builtin_ia32_cvtpd2ps256(tmp_V4d);
  tmp_V8i = __builtin_ia32_cvtps2dq256(tmp_V8f);
  tmp_V4i = __builtin_ia32_cvttpd2dq256(tmp_V4d);
  tmp_V4i = __builtin_ia32_cvtpd2dq256(tmp_V4d);
  tmp_V8i = __builtin_ia32_cvttps2dq256(tmp_V8f);
  tmp_V4d = __builtin_ia32_vperm2f128_pd256(tmp_V4d, tmp_V4d, 0x7);
  tmp_V8f = __builtin_ia32_vperm2f128_ps256(tmp_V8f, tmp_V8f, 0x7);
  tmp_V8i = __builtin_ia32_vperm2f128_si256(tmp_V8i, tmp_V8i, 0x7);
  tmp_V4d = __builtin_ia32_sqrtpd256(tmp_V4d);
  tmp_V8f = __builtin_ia32_sqrtps256(tmp_V8f);
  tmp_V8f = __builtin_ia32_rsqrtps256(tmp_V8f);
  tmp_V8f = __builtin_ia32_rcpps256(tmp_V8f);
  tmp_V4d = __builtin_ia32_roundpd256(tmp_V4d, 0x1);
  tmp_V8f = __builtin_ia32_roundps256(tmp_V8f, 0x1);
  tmp_i = __builtin_ia32_vtestzpd(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_vtestcpd(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_vtestnzcpd(tmp_V2d, tmp_V2d);
  tmp_i = __builtin_ia32_vtestzps(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_vtestcps(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_vtestnzcps(tmp_V4f, tmp_V4f);
  tmp_i = __builtin_ia32_vtestzpd256(tmp_V4d, tmp_V4d);
  tmp_i = __builtin_ia32_vtestcpd256(tmp_V4d, tmp_V4d);
  tmp_i = __builtin_ia32_vtestnzcpd256(tmp_V4d, tmp_V4d);
  tmp_i = __builtin_ia32_vtestzps256(tmp_V8f, tmp_V8f);
  tmp_i = __builtin_ia32_vtestcps256(tmp_V8f, tmp_V8f);
  tmp_i = __builtin_ia32_vtestnzcps256(tmp_V8f, tmp_V8f);
  tmp_i = __builtin_ia32_ptestz256(tmp_V4LLi, tmp_V4LLi);
  tmp_i = __builtin_ia32_ptestc256(tmp_V4LLi, tmp_V4LLi);
  tmp_i = __builtin_ia32_ptestnzc256(tmp_V4LLi, tmp_V4LLi);
  tmp_i = __builtin_ia32_movmskpd256(tmp_V4d);
  tmp_i = __builtin_ia32_movmskps256(tmp_V8f);
  __builtin_ia32_vzeroall();
  __builtin_ia32_vzeroupper();
  tmp_V4d = __builtin_ia32_vbroadcastf128_pd256(tmp_V2dCp);
  tmp_V8f = __builtin_ia32_vbroadcastf128_ps256(tmp_V4fCp);
  tmp_V32c = __builtin_ia32_lddqu256(tmp_cCp);
  tmp_V2d = __builtin_ia32_maskloadpd(tmp_V2dCp, tmp_V2LLi);
  tmp_V4f = __builtin_ia32_maskloadps(tmp_V4fCp, tmp_V4i);
  tmp_V4d = __builtin_ia32_maskloadpd256(tmp_V4dCp, tmp_V4LLi);
  tmp_V8f = __builtin_ia32_maskloadps256(tmp_V8fCp, tmp_V8i);
  __builtin_ia32_maskstorepd(tmp_V2dp, tmp_V2LLi, tmp_V2d);
  __builtin_ia32_maskstoreps(tmp_V4fp, tmp_V4i, tmp_V4f);
  __builtin_ia32_maskstorepd256(tmp_V4dp, tmp_V4LLi, tmp_V4d);
  __builtin_ia32_maskstoreps256(tmp_V8fp, tmp_V8i, tmp_V8f);

#ifdef USE_3DNOW
  tmp_V8c = __builtin_ia32_pavgusb(tmp_V8c, tmp_V8c);
  tmp_V2i = __builtin_ia32_pf2id(tmp_V2f);
  tmp_V2f = __builtin_ia32_pfacc(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfadd(tmp_V2f, tmp_V2f);
  tmp_V2i = __builtin_ia32_pfcmpeq(tmp_V2f, tmp_V2f);
  tmp_V2i = __builtin_ia32_pfcmpge(tmp_V2f, tmp_V2f);
  tmp_V2i = __builtin_ia32_pfcmpgt(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfmax(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfmin(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfmul(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfrcp(tmp_V2f);
  tmp_V2f = __builtin_ia32_pfrcpit1(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfrcpit2(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfrsqrt(tmp_V2f);
  tmp_V2f = __builtin_ia32_pfrsqit1(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfsub(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfsubr(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pi2fd(tmp_V2i);
  tmp_V4s = __builtin_ia32_pmulhrw(tmp_V4s, tmp_V4s);
  tmp_V2i = __builtin_ia32_pf2iw(tmp_V2f);
  tmp_V2f = __builtin_ia32_pfnacc(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pfpnacc(tmp_V2f, tmp_V2f);
  tmp_V2f = __builtin_ia32_pi2fw(tmp_V2i);
  tmp_V2f = __builtin_ia32_pswapdsf(tmp_V2f);
  tmp_V2i = __builtin_ia32_pswapdsi(tmp_V2i);

  tmp_V4i = __builtin_ia32_sha1rnds4(tmp_V4i, tmp_V4i, imm_i_0_4);
  tmp_V4i = __builtin_ia32_sha1nexte(tmp_V4i, tmp_V4i);
  tmp_V4i = __builtin_ia32_sha1msg1(tmp_V4i, tmp_V4i);
  tmp_V4i = __builtin_ia32_sha1msg2(tmp_V4i, tmp_V4i);
  tmp_V4i = __builtin_ia32_sha256rnds2(tmp_V4i, tmp_V4i, tmp_V4i);
  tmp_V4i = __builtin_ia32_sha256msg1(tmp_V4i, tmp_V4i);
  tmp_V4i = __builtin_ia32_sha256msg2(tmp_V4i, tmp_V4i);
#endif
}
