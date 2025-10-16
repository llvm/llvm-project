// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -o -
#define __MM_MALLOC_H

#include <x86intrin.h>

// No warnings.
extern __m256i a;
int __attribute__((target("avx"))) bar(void) {
  return _mm256_extract_epi32(a, 3);
}

int baz(void) {
  return bar();
}

int __attribute__((target("avx"))) qq_avx(void) {
  return _mm256_extract_epi32(a, 3);
}

int qq_noavx(void) {
  return 0;
}

extern __m256i a;
int qq(void) {
  if (__builtin_cpu_supports("avx"))
    return qq_avx();
  else
    return qq_noavx();
}

// Test that fma and fma4 are both separately and combined valid for an fma intrinsic.
__m128 __attribute__((target("fma"))) fma_1(__m128 a, __m128 b, __m128 c) {
  return __builtin_ia32_vfmaddsubps(a, b, c);
}

__m128 __attribute__((target("fma4"))) fma_2(__m128 a, __m128 b, __m128 c) {
  return __builtin_ia32_vfmaddsubps(a, b, c);
}

__m128 __attribute__((target("fma,fma4"))) fma_3(__m128 a, __m128 b, __m128 c) {
  return __builtin_ia32_vfmaddsubps(a, b, c);
}

void verifyfeaturestrings(void) {
  (void)__builtin_cpu_supports("cmov");
  (void)__builtin_cpu_supports("mmx");
  (void)__builtin_cpu_supports("popcnt");
  (void)__builtin_cpu_supports("sse");
  (void)__builtin_cpu_supports("sse2");
  (void)__builtin_cpu_supports("sse3");
  (void)__builtin_cpu_supports("ssse3");
  (void)__builtin_cpu_supports("sse4.1");
  (void)__builtin_cpu_supports("sse4.2");
  (void)__builtin_cpu_supports("avx");
  (void)__builtin_cpu_supports("avx2");
  (void)__builtin_cpu_supports("sse4a");
  (void)__builtin_cpu_supports("fma4");
  (void)__builtin_cpu_supports("xop");
  (void)__builtin_cpu_supports("fma");
  (void)__builtin_cpu_supports("avx512f");
  (void)__builtin_cpu_supports("bmi");
  (void)__builtin_cpu_supports("bmi2");
  (void)__builtin_cpu_supports("aes");
  (void)__builtin_cpu_supports("pclmul");
  (void)__builtin_cpu_supports("avx512vl");
  (void)__builtin_cpu_supports("avx512bw");
  (void)__builtin_cpu_supports("avx512dq");
  (void)__builtin_cpu_supports("avx512cd");
  (void)__builtin_cpu_supports("avx512vbmi");
  (void)__builtin_cpu_supports("avx512ifma");
  (void)__builtin_cpu_supports("avx5124vnniw");
  (void)__builtin_cpu_supports("avx5124fmaps");
  (void)__builtin_cpu_supports("avx512vpopcntdq");
  (void)__builtin_cpu_supports("avx512vbmi2");
  (void)__builtin_cpu_supports("gfni");
  (void)__builtin_cpu_supports("vpclmulqdq");
  (void)__builtin_cpu_supports("avx512vnni");
  (void)__builtin_cpu_supports("avx512bitalg");
  (void)__builtin_cpu_supports("avx512bf16");
  (void)__builtin_cpu_supports("avx512vp2intersect");
  (void)__builtin_cpu_supports("f16c");
  (void)__builtin_cpu_supports("avx512fp16");
  (void)__builtin_cpu_supports("3dnow");
  (void)__builtin_cpu_supports("adx");
  (void)__builtin_cpu_supports("cldemote");
  (void)__builtin_cpu_supports("clflushopt");
  (void)__builtin_cpu_supports("clwb");
  (void)__builtin_cpu_supports("clzero");
  (void)__builtin_cpu_supports("cx16");
  (void)__builtin_cpu_supports("enqcmd");
  (void)__builtin_cpu_supports("fsgsbase");
  (void)__builtin_cpu_supports("lwp");
  (void)__builtin_cpu_supports("lzcnt");
  (void)__builtin_cpu_supports("movbe");
  (void)__builtin_cpu_supports("movdir64b");
  (void)__builtin_cpu_supports("movdiri");
  (void)__builtin_cpu_supports("mwaitx");
  (void)__builtin_cpu_supports("pconfig");
  (void)__builtin_cpu_supports("pku");
  (void)__builtin_cpu_supports("prefetchwt1");
  (void)__builtin_cpu_supports("prfchw");
  (void)__builtin_cpu_supports("ptwrite");
  (void)__builtin_cpu_supports("rdpid");
  (void)__builtin_cpu_supports("rdrnd");
  (void)__builtin_cpu_supports("rdseed");
  (void)__builtin_cpu_supports("rtm");
  (void)__builtin_cpu_supports("serialize");
  (void)__builtin_cpu_supports("sgx");
  (void)__builtin_cpu_supports("sha");
  (void)__builtin_cpu_supports("shstk");
  (void)__builtin_cpu_supports("tbm");
  (void)__builtin_cpu_supports("tsxldtrk");
  (void)__builtin_cpu_supports("vaes");
  (void)__builtin_cpu_supports("waitpkg");
  (void)__builtin_cpu_supports("wbnoinvd");
  (void)__builtin_cpu_supports("xsave");
  (void)__builtin_cpu_supports("xsavec");
  (void)__builtin_cpu_supports("xsaveopt");
  (void)__builtin_cpu_supports("xsaves");
  (void)__builtin_cpu_supports("amx-tile");
  (void)__builtin_cpu_supports("amx-int8");
  (void)__builtin_cpu_supports("amx-bf16");
  (void)__builtin_cpu_supports("uintr");
  (void)__builtin_cpu_supports("hreset");
  (void)__builtin_cpu_supports("kl");
  (void)__builtin_cpu_supports("widekl");
  (void)__builtin_cpu_supports("avxvnni");
  (void)__builtin_cpu_supports("avxifma");
  (void)__builtin_cpu_supports("avxvnniint8");
  (void)__builtin_cpu_supports("avxneconvert");
  (void)__builtin_cpu_supports("cmpccxadd");
  (void)__builtin_cpu_supports("amx-fp16");
  (void)__builtin_cpu_supports("prefetchi");
  (void)__builtin_cpu_supports("raoint");
  (void)__builtin_cpu_supports("amx-complex");
  (void)__builtin_cpu_supports("avxvnniint16");
  (void)__builtin_cpu_supports("sm3");
  (void)__builtin_cpu_supports("sha512");
  (void)__builtin_cpu_supports("sm4");
  (void)__builtin_cpu_supports("apxf");
  (void)__builtin_cpu_supports("usermsr");
  (void)__builtin_cpu_supports("avx10.1");
  (void)__builtin_cpu_supports("avx10.2");
  (void)__builtin_cpu_supports("movrs");
}

void verifycpustrings(void) {
  (void)__builtin_cpu_is("alderlake");
  (void)__builtin_cpu_is("amd");
  (void)__builtin_cpu_is("amdfam10h");
  (void)__builtin_cpu_is("amdfam15h");
  (void)__builtin_cpu_is("amdfam17h");
  (void)__builtin_cpu_is("atom");
  (void)__builtin_cpu_is("barcelona");
  (void)__builtin_cpu_is("bdver1");
  (void)__builtin_cpu_is("bdver2");
  (void)__builtin_cpu_is("bdver3");
  (void)__builtin_cpu_is("bdver4");
  (void)__builtin_cpu_is("bonnell");
  (void)__builtin_cpu_is("broadwell");
  (void)__builtin_cpu_is("btver1");
  (void)__builtin_cpu_is("btver2");
  (void)__builtin_cpu_is("cannonlake");
  (void)__builtin_cpu_is("cascadelake");
  (void)__builtin_cpu_is("cooperlake");
  (void)__builtin_cpu_is("core2");
  (void)__builtin_cpu_is("corei7");
  (void)__builtin_cpu_is("goldmont");
  (void)__builtin_cpu_is("goldmont-plus");
  (void)__builtin_cpu_is("grandridge");
  (void)__builtin_cpu_is("graniterapids");
  (void)__builtin_cpu_is("emeraldrapids");
  (void)__builtin_cpu_is("graniterapids-d");
  (void)__builtin_cpu_is("arrowlake");
  (void)__builtin_cpu_is("arrowlake-s");
  (void)__builtin_cpu_is("lunarlake");
  (void)__builtin_cpu_is("clearwaterforest");
  (void)__builtin_cpu_is("pantherlake");
  (void)__builtin_cpu_is("wildcatlake");
  (void)__builtin_cpu_is("haswell");
  (void)__builtin_cpu_is("icelake-client");
  (void)__builtin_cpu_is("icelake-server");
  (void)__builtin_cpu_is("intel");
  (void)__builtin_cpu_is("istanbul");
  (void)__builtin_cpu_is("ivybridge");
  (void)__builtin_cpu_is("knl");
  (void)__builtin_cpu_is("knm");
  (void)__builtin_cpu_is("meteorlake");
  (void)__builtin_cpu_is("nehalem");
  (void)__builtin_cpu_is("raptorlake");
  (void)__builtin_cpu_is("rocketlake");
  (void)__builtin_cpu_is("sandybridge");
  (void)__builtin_cpu_is("shanghai");
  (void)__builtin_cpu_is("sierraforest");
  (void)__builtin_cpu_is("silvermont");
  (void)__builtin_cpu_is("skylake");
  (void)__builtin_cpu_is("skylake-avx512");
  (void)__builtin_cpu_is("slm");
  (void)__builtin_cpu_is("tigerlake");
  (void)__builtin_cpu_is("sapphirerapids");
  (void)__builtin_cpu_is("tremont");
  (void)__builtin_cpu_is("gracemont");
  (void)__builtin_cpu_is("westmere");
  (void)__builtin_cpu_is("znver1");
  (void)__builtin_cpu_is("znver2");
  (void)__builtin_cpu_is("znver3");
  (void)__builtin_cpu_is("znver4");
  (void)__builtin_cpu_is("znver5");
  (void)__builtin_cpu_is("diamondrapids");
}
