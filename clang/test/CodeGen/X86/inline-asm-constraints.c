// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -target-cpu x86-64 -o - |FileCheck %s --check-prefix SSE
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -target-cpu skylake -D AVX -o - | FileCheck %s --check-prefixes AVX,SSE
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -target-cpu skylake-avx512 -D AVX512 -D AVX -o - | FileCheck %s --check-prefixes AVX512,AVX,SSE
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -target-cpu knl -D AVX -D AVX512 -o - | FileCheck %s --check-prefixes AVX512,AVX,SSE

typedef float __m128 __attribute__ ((vector_size (16)));
typedef float __m256 __attribute__ ((vector_size (32)));
typedef float __m512 __attribute__ ((vector_size (64)));

// SSE: call <4 x float> asm "vmovhlps $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(i64 %0, <4 x float> %1)
__m128 testXMM(__m128 _xmm0, long _l) {
  __asm__("vmovhlps %1, %2, %0" :"=v"(_xmm0) : "v"(_l), "v"(_xmm0));
  return _xmm0;
}

// AVX: call <8 x float> asm "vmovsldup $1, $0", "=v,v,~{dirflag},~{fpsr},~{flags}"(<8 x float> %0)
__m256 testYMM(__m256 _ymm0) {
#ifdef AVX
  __asm__("vmovsldup %1, %0" :"=v"(_ymm0) : "v"(_ymm0));
#endif
  return _ymm0;
}

// AVX512: call <16 x float> asm "vpternlogd $$0, $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(<16 x float> %0, <16 x float> %1)
__m512 testZMM(__m512 _zmm0, __m512 _zmm1) {
#ifdef AVX512
  __asm__("vpternlogd $0, %1, %2, %0" :"=v"(_zmm0) : "v"(_zmm1), "v"(_zmm0));
#endif
  return _zmm0;
}

// SSE: call <4 x float> asm "pcmpeqd $0, $0", "=^Yz,~{dirflag},~{fpsr},~{flags}"()
__m128 testXMM0(void) {
  __m128 xmm0;
  __asm__("pcmpeqd %0, %0" :"=Yz"(xmm0));
  return xmm0;
}

// AVX: call <8 x float> asm "vpcmpeqd $0, $0, $0", "=^Yz,~{dirflag},~{fpsr},~{flags}"()
__m256 testYMM0(void) {
  __m256 ymm0;
#ifdef AVX
  __asm__("vpcmpeqd %0, %0, %0" :"=Yz"(ymm0));
#endif
  return ymm0;
}

// AVX512: call <16 x float> asm "vpternlogd $$255, $0, $0, $0", "=^Yz,~{dirflag},~{fpsr},~{flags}"()
__m512 testZMM0(void) {
  __m512 zmm0;
#ifdef AVX512
  __asm__("vpternlogd $255, %0, %0, %0" :"=Yz"(zmm0));
#endif
  return zmm0;
}

extern int var, arr[4];
struct Pair { int a, b; } pair;

// CHECK-LABEL: test_Ws(
// CHECK:         call void asm sideeffect "// ${0:p} ${1:p} ${2:p}", "^Ws,^Ws,^Ws,~{dirflag},~{fpsr},~{flags}"(ptr @var, ptr getelementptr inbounds ([4 x i32], ptr @arr, i64 0, i64 3), ptr @test_Ws)
// CHECK:         call void asm sideeffect "// $0", "^Ws,~{dirflag},~{fpsr},~{flags}"(ptr getelementptr inbounds (%struct.Pair, ptr @pair, i32 0, i32 1))
void test_Ws(void) {
  asm("// %p0 %p1 %p2" :: "Ws"(&var), "Ws"(&arr[3]), "Ws"(test_Ws));
  asm("// %0" :: "Ws"(&pair.b));
}
