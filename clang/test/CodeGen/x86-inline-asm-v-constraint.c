// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -target-cpu x86-64 -o - |opt -instnamer -S |FileCheck %s --check-prefix SSE
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -target-cpu skylake -D AVX -o -|opt -instnamer -S  | FileCheck %s --check-prefixes AVX,SSE
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -target-cpu skylake-avx512 -D AVX512 -D AVX -o -|opt -instnamer -S  | FileCheck %s --check-prefixes AVX512,AVX,SSE
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -target-cpu knl -D AVX -D AVX512 -o - |opt -instnamer -S  | FileCheck %s --check-prefixes AVX512,AVX,SSE

typedef float __m128 __attribute__ ((vector_size (16)));
typedef float __m256 __attribute__ ((vector_size (32)));
typedef float __m512 __attribute__ ((vector_size (64)));

// SSE: call <4 x float> asm "vmovhlps $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(i64 %tmp, <4 x float> %tmp1)
__m128 testXMM(__m128 _xmm0, long _l) {
  __asm__("vmovhlps %1, %2, %0" :"=v"(_xmm0) : "v"(_l), "v"(_xmm0));
  return _xmm0;
}

// AVX: call <8 x float> asm "vmovsldup $1, $0", "=v,v,~{dirflag},~{fpsr},~{flags}"(<8 x float> %tmp)
__m256 testYMM(__m256 _ymm0) {
#ifdef AVX
  __asm__("vmovsldup %1, %0" :"=v"(_ymm0) : "v"(_ymm0));
#endif
  return _ymm0;
}

// AVX512: call <16 x float> asm "vpternlogd $$0, $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(<16 x float> %tmp, <16 x float> %tmp1)
__m512 testZMM(__m512 _zmm0, __m512 _zmm1) {
#ifdef AVX512
  __asm__("vpternlogd $0, %1, %2, %0" :"=v"(_zmm0) : "v"(_zmm1), "v"(_zmm0));
#endif
  return _zmm0;
}
