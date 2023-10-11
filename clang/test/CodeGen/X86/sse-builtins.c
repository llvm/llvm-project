// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>

// NOTE: This should match the tests in llvm/test/CodeGen/X86/sse-intrinsics-fast-isel.ll

__m128 test_mm_add_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_add_ps
  // CHECK: fadd <4 x float>
  return _mm_add_ps(A, B);
}

__m128 test_mm_add_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_add_ss
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: fadd float
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 0
  return _mm_add_ss(A, B);
}

__m128 test_mm_and_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_and_ps
  // CHECK: and <4 x i32>
  return _mm_and_ps(A, B);
}

__m128 test_mm_andnot_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_andnot_ps
  // CHECK: xor <4 x i32> %{{.*}}, <i32 -1, i32 -1, i32 -1, i32 -1>
  // CHECK: and <4 x i32>
  return _mm_andnot_ps(A, B);
}

__m128 test_mm_cmpeq_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpeq_ps
  // CHECK:         [[CMP:%.*]] = fcmp oeq <4 x float>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpeq_ps(__a, __b);
}

__m128 test_mm_cmpeq_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpeq_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 0)
  return _mm_cmpeq_ss(__a, __b);
}

__m128 test_mm_cmpge_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpge_ps
  // CHECK:         [[CMP:%.*]] = fcmp ole <4 x float>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpge_ps(__a, __b);
}

__m128 test_mm_cmpge_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpge_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 4, i32 1, i32 2, i32 3>
  return _mm_cmpge_ss(__a, __b);
}

__m128 test_mm_cmpgt_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpgt_ps
  // CHECK:         [[CMP:%.*]] = fcmp olt <4 x float>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpgt_ps(__a, __b);
}

__m128 test_mm_cmpgt_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpgt_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 1)
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 4, i32 1, i32 2, i32 3>
  return _mm_cmpgt_ss(__a, __b);
}

__m128 test_mm_cmple_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmple_ps
  // CHECK:         [[CMP:%.*]] = fcmp ole <4 x float>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmple_ps(__a, __b);
}

__m128 test_mm_cmple_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmple_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm_cmple_ss(__a, __b);
}

__m128 test_mm_cmplt_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmplt_ps
  // CHECK:         [[CMP:%.*]] = fcmp olt <4 x float>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmplt_ps(__a, __b);
}

__m128 test_mm_cmplt_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmplt_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 1)
  return _mm_cmplt_ss(__a, __b);
}

__m128 test_mm_cmpneq_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpneq_ps
  // CHECK:         [[CMP:%.*]] = fcmp une <4 x float>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpneq_ps(__a, __b);
}

__m128 test_mm_cmpneq_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpneq_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 4)
  return _mm_cmpneq_ss(__a, __b);
}

__m128 test_mm_cmpnge_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpnge_ps
  // CHECK:         [[CMP:%.*]] = fcmp ugt <4 x float>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpnge_ps(__a, __b);
}

__m128 test_mm_cmpnge_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpnge_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 6)
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 4, i32 1, i32 2, i32 3>
  return _mm_cmpnge_ss(__a, __b);
}

__m128 test_mm_cmpngt_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpngt_ps
  // CHECK:         [[CMP:%.*]] = fcmp uge <4 x float>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpngt_ps(__a, __b);
}

__m128 test_mm_cmpngt_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpngt_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 5)
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 4, i32 1, i32 2, i32 3>
  return _mm_cmpngt_ss(__a, __b);
}

__m128 test_mm_cmpnle_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpnle_ps
  // CHECK:         [[CMP:%.*]] = fcmp ugt <4 x float>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpnle_ps(__a, __b);
}

__m128 test_mm_cmpnle_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpnle_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 6)
  return _mm_cmpnle_ss(__a, __b);
}

__m128 test_mm_cmpnlt_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpnlt_ps
  // CHECK:         [[CMP:%.*]] = fcmp uge <4 x float>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpnlt_ps(__a, __b);
}

__m128 test_mm_cmpnlt_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpnlt_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 5)
  return _mm_cmpnlt_ss(__a, __b);
}

__m128 test_mm_cmpord_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpord_ps
  // CHECK:         [[CMP:%.*]] = fcmp ord <4 x float>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpord_ps(__a, __b);
}

__m128 test_mm_cmpord_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpord_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 7)
  return _mm_cmpord_ss(__a, __b);
}

__m128 test_mm_cmpunord_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpunord_ps
  // CHECK:         [[CMP:%.*]] = fcmp uno <4 x float>
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpunord_ps(__a, __b);
}

__m128 test_mm_cmpunord_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: test_mm_cmpunord_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 3)
  return _mm_cmpunord_ss(__a, __b);
}

int test_mm_comieq_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_comieq_ss
  // CHECK: call i32 @llvm.x86.sse.comieq.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_comieq_ss(A, B);
}

int test_mm_comige_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_comige_ss
  // CHECK: call i32 @llvm.x86.sse.comige.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_comige_ss(A, B);
}

int test_mm_comigt_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_comigt_ss
  // CHECK: call i32 @llvm.x86.sse.comigt.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_comigt_ss(A, B);
}

int test_mm_comile_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_comile_ss
  // CHECK: call i32 @llvm.x86.sse.comile.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_comile_ss(A, B);
}

int test_mm_comilt_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_comilt_ss
  // CHECK: call i32 @llvm.x86.sse.comilt.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_comilt_ss(A, B);
}

int test_mm_comineq_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_comineq_ss
  // CHECK: call i32 @llvm.x86.sse.comineq.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_comineq_ss(A, B);
}

int test_mm_cvt_ss2si(__m128 A) {
  // CHECK-LABEL: test_mm_cvt_ss2si
  // CHECK: call i32 @llvm.x86.sse.cvtss2si(<4 x float> %{{.*}})
  return _mm_cvt_ss2si(A);
}

__m128 test_mm_cvtsi32_ss(__m128 A, int B) {
  // CHECK-LABEL: test_mm_cvtsi32_ss
  // CHECK: sitofp i32 %{{.*}} to float
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 0
  return _mm_cvtsi32_ss(A, B);
}

#ifdef __x86_64__
__m128 test_mm_cvtsi64_ss(__m128 A, long long B) {
  // CHECK-LABEL: test_mm_cvtsi64_ss
  // CHECK: sitofp i64 %{{.*}} to float
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 0
  return _mm_cvtsi64_ss(A, B);
}
#endif

float test_mm_cvtss_f32(__m128 A) {
  // CHECK-LABEL: test_mm_cvtss_f32
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  return _mm_cvtss_f32(A);
}

int test_mm_cvtss_si32(__m128 A) {
  // CHECK-LABEL: test_mm_cvtss_si32
  // CHECK: call i32 @llvm.x86.sse.cvtss2si(<4 x float> %{{.*}})
  return _mm_cvtss_si32(A);
}

#ifdef __x86_64__
long long test_mm_cvtss_si64(__m128 A) {
  // CHECK-LABEL: test_mm_cvtss_si64
  // CHECK: call i64 @llvm.x86.sse.cvtss2si64(<4 x float> %{{.*}})
  return _mm_cvtss_si64(A);
}
#endif

int test_mm_cvtt_ss2si(__m128 A) {
  // CHECK-LABEL: test_mm_cvtt_ss2si
  // CHECK: call i32 @llvm.x86.sse.cvttss2si(<4 x float> %{{.*}})
  return _mm_cvtt_ss2si(A);
}

int test_mm_cvttss_si32(__m128 A) {
  // CHECK-LABEL: test_mm_cvttss_si32
  // CHECK: call i32 @llvm.x86.sse.cvttss2si(<4 x float> %{{.*}})
  return _mm_cvttss_si32(A);
}

#ifdef __x86_64__
long long test_mm_cvttss_si64(__m128 A) {
  // CHECK-LABEL: test_mm_cvttss_si64
  // CHECK: call i64 @llvm.x86.sse.cvttss2si64(<4 x float> %{{.*}})
  return _mm_cvttss_si64(A);
}
#endif

__m128 test_mm_div_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_div_ps
  // CHECK: fdiv <4 x float>
  return _mm_div_ps(A, B);
}

__m128 test_mm_div_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_div_ss
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: fdiv float
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 0
  return _mm_div_ss(A, B);
}

unsigned int test_MM_GET_EXCEPTION_MASK(void) {
  // CHECK-LABEL: test_MM_GET_EXCEPTION_MASK
  // CHECK: call void @llvm.x86.sse.stmxcsr(ptr %{{.*}})
  // CHECK: and i32 %{{.*}}, 8064
  return _MM_GET_EXCEPTION_MASK();
}

unsigned int test_MM_GET_EXCEPTION_STATE(void) {
  // CHECK-LABEL: test_MM_GET_EXCEPTION_STATE
  // CHECK: call void @llvm.x86.sse.stmxcsr(ptr %{{.*}})
  // CHECK: and i32 %{{.*}}, 63
  return _MM_GET_EXCEPTION_STATE();
}

unsigned int test_MM_GET_FLUSH_ZERO_MODE(void) {
  // CHECK-LABEL: test_MM_GET_FLUSH_ZERO_MODE
  // CHECK: call void @llvm.x86.sse.stmxcsr(ptr %{{.*}})
  // CHECK: and i32 %{{.*}}, 32768
  return _MM_GET_FLUSH_ZERO_MODE();
}

unsigned int test_MM_GET_ROUNDING_MODE(void) {
  // CHECK-LABEL: test_MM_GET_ROUNDING_MODE
  // CHECK: call void @llvm.x86.sse.stmxcsr(ptr %{{.*}})
  // CHECK: and i32 %{{.*}}, 24576
  return _MM_GET_ROUNDING_MODE();
}

unsigned int test_mm_getcsr(void) {
  // CHECK-LABEL: test_mm_getcsr
  // CHECK: call void @llvm.x86.sse.stmxcsr(ptr %{{.*}})
  // CHECK: load i32
  return _mm_getcsr();
}

__m128 test_mm_load_ps(float* y) {
  // CHECK-LABEL: test_mm_load_ps
  // CHECK: load <4 x float>, ptr {{.*}}, align 16
  return _mm_load_ps(y);
}

__m128 test_mm_load_ps1(float* y) {
  // CHECK-LABEL: test_mm_load_ps1
  // CHECK: load float, ptr %{{.*}}, align 4
  // CHECK: insertelement <4 x float> undef, float %{{.*}}, i32 0
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 1
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 2
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 3
  return _mm_load_ps1(y);
}

__m128 test_mm_load_ss(float* y) {
  // CHECK-LABEL: test_mm_load_ss
  // CHECK: load float, ptr {{.*}}, align 1{{$}}
  // CHECK: insertelement <4 x float> undef, float %{{.*}}, i32 0
  // CHECK: insertelement <4 x float> %{{.*}}, float 0.000000e+00, i32 1
  // CHECK: insertelement <4 x float> %{{.*}}, float 0.000000e+00, i32 2
  // CHECK: insertelement <4 x float> %{{.*}}, float 0.000000e+00, i32 3
  return _mm_load_ss(y);
}

__m128 test_mm_load1_ps(float* y) {
  // CHECK-LABEL: test_mm_load1_ps
  // CHECK: load float, ptr %{{.*}}, align 4
  // CHECK: insertelement <4 x float> undef, float %{{.*}}, i32 0
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 1
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 2
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 3
  return _mm_load1_ps(y);
}

__m128 test_mm_loadh_pi(__m128 x, __m64* y) {
  // CHECK-LABEL: test_mm_loadh_pi
  // CHECK: load <2 x float>, ptr {{.*}}, align 1{{$}}
  // CHECK: shufflevector {{.*}} <4 x i32> <i32 0, i32 1
  // CHECK: shufflevector {{.*}} <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  return _mm_loadh_pi(x,y);
}

__m128 test_mm_loadl_pi(__m128 x, __m64* y) {
  // CHECK-LABEL: test_mm_loadl_pi
  // CHECK: load <2 x float>, ptr {{.*}}, align 1{{$}}
  // CHECK: shufflevector {{.*}} <4 x i32> <i32 0, i32 1
  // CHECK: shufflevector {{.*}} <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  return _mm_loadl_pi(x,y);
}

__m128 test_mm_loadr_ps(float* A) {
  // CHECK-LABEL: test_mm_loadr_ps
  // CHECK: load <4 x float>, ptr %{{.*}}, align 16
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  return _mm_loadr_ps(A);
}

__m128 test_mm_loadu_ps(float* A) {
  // CHECK-LABEL: test_mm_loadu_ps
  // CHECK: load <4 x float>, ptr %{{.*}}, align 1{{$}}
  return _mm_loadu_ps(A);
}

__m128 test_mm_max_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_max_ps
  // CHECK: @llvm.x86.sse.max.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_max_ps(A, B);
}

__m128 test_mm_max_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_max_ss
  // CHECK: @llvm.x86.sse.max.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_max_ss(A, B);
}

__m128 test_mm_min_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_min_ps
  // CHECK: @llvm.x86.sse.min.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_min_ps(A, B);
}

__m128 test_mm_min_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_min_ss
  // CHECK: @llvm.x86.sse.min.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_min_ss(A, B);
}

__m128 test_mm_move_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_move_ss
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 0
  return _mm_move_ss(A, B);
}

__m128 test_mm_movehl_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_movehl_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 6, i32 7, i32 2, i32 3>
  return _mm_movehl_ps(A, B);
}

__m128 test_mm_movelh_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_movelh_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  return _mm_movelh_ps(A, B);
}

int test_mm_movemask_ps(__m128 A) {
  // CHECK-LABEL: test_mm_movemask_ps
  // CHECK: call i32 @llvm.x86.sse.movmsk.ps(<4 x float> %{{.*}})
  return _mm_movemask_ps(A);
}

__m128 test_mm_mul_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_mul_ps
  // CHECK: fmul <4 x float>
  return _mm_mul_ps(A, B);
}

__m128 test_mm_mul_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_mul_ss
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: fmul float
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 0
  return _mm_mul_ss(A, B);
}

__m128 test_mm_or_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_or_ps
  // CHECK: or <4 x i32>
  return _mm_or_ps(A, B);
}

void test_mm_prefetch(char const* p) {
  // CHECK-LABEL: test_mm_prefetch
  // CHECK: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 0, i32 1)
  _mm_prefetch(p, 0);
}

__m128 test_mm_rcp_ps(__m128 x) {
  // CHECK-LABEL: test_mm_rcp_ps
  // CHECK: call <4 x float> @llvm.x86.sse.rcp.ps(<4 x float> {{.*}})
  return _mm_rcp_ps(x);
}

__m128 test_mm_rcp_ss(__m128 x) {
  // CHECK-LABEL: test_mm_rcp_ss
  // CHECK: call <4 x float> @llvm.x86.sse.rcp.ss(<4 x float> {{.*}})
  return _mm_rcp_ss(x);
}

__m128 test_mm_rsqrt_ps(__m128 x) {
  // CHECK-LABEL: test_mm_rsqrt_ps
  // CHECK: call <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float> {{.*}})
  return _mm_rsqrt_ps(x);
}

__m128 test_mm_rsqrt_ss(__m128 x) {
  // CHECK-LABEL: test_mm_rsqrt_ss
  // CHECK: call <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float> {{.*}})
  return _mm_rsqrt_ss(x);
}

void test_MM_SET_EXCEPTION_MASK(unsigned int A) {
  // CHECK-LABEL: test_MM_SET_EXCEPTION_MASK
  // CHECK: call void @llvm.x86.sse.stmxcsr(ptr {{.*}})
  // CHECK: load i32
  // CHECK: and i32 {{.*}}, -8065
  // CHECK: or i32
  // CHECK: store i32
  // CHECK: call void @llvm.x86.sse.ldmxcsr(ptr {{.*}})
  _MM_SET_EXCEPTION_MASK(A);
}

void test_MM_SET_EXCEPTION_STATE(unsigned int A) {
  // CHECK-LABEL: test_MM_SET_EXCEPTION_STATE
  // CHECK: call void @llvm.x86.sse.stmxcsr(ptr {{.*}})
  // CHECK: load i32
  // CHECK: and i32 {{.*}}, -64
  // CHECK: or i32
  // CHECK: store i32
  // CHECK: call void @llvm.x86.sse.ldmxcsr(ptr {{.*}})
  _MM_SET_EXCEPTION_STATE(A);
}

void test_MM_SET_FLUSH_ZERO_MODE(unsigned int A) {
  // CHECK-LABEL: test_MM_SET_FLUSH_ZERO_MODE
  // CHECK: call void @llvm.x86.sse.stmxcsr(ptr {{.*}})
  // CHECK: load i32
  // CHECK: and i32 {{.*}}, -32769
  // CHECK: or i32
  // CHECK: store i32
  // CHECK: call void @llvm.x86.sse.ldmxcsr(ptr {{.*}})
  _MM_SET_FLUSH_ZERO_MODE(A);
}

__m128 test_mm_set_ps(float A, float B, float C, float D) {
  // CHECK-LABEL: test_mm_set_ps
  // CHECK: insertelement <4 x float> undef, float {{.*}}, i32 0
  // CHECK: insertelement <4 x float> {{.*}}, float {{.*}}, i32 1
  // CHECK: insertelement <4 x float> {{.*}}, float {{.*}}, i32 2
  // CHECK: insertelement <4 x float> {{.*}}, float {{.*}}, i32 3
  return _mm_set_ps(A, B, C, D);
}

__m128 test_mm_set_ps1(float A) {
  // CHECK-LABEL: test_mm_set_ps1
  // CHECK: insertelement <4 x float> undef, float {{.*}}, i32 0
  // CHECK: insertelement <4 x float> {{.*}}, float {{.*}}, i32 1
  // CHECK: insertelement <4 x float> {{.*}}, float {{.*}}, i32 2
  // CHECK: insertelement <4 x float> {{.*}}, float {{.*}}, i32 3
  return _mm_set_ps1(A);
}

void test_MM_SET_ROUNDING_MODE(unsigned int A) {
  // CHECK-LABEL: test_MM_SET_ROUNDING_MODE
  // CHECK: call void @llvm.x86.sse.stmxcsr(ptr {{.*}})
  // CHECK: load i32
  // CHECK: and i32 {{.*}}, -24577
  // CHECK: or i32
  // CHECK: store i32
  // CHECK: call void @llvm.x86.sse.ldmxcsr(ptr {{.*}})
  _MM_SET_ROUNDING_MODE(A);
}

__m128 test_mm_set_ss(float A) {
  // CHECK-LABEL: test_mm_set_ss
  // CHECK: insertelement <4 x float> undef, float {{.*}}, i32 0
  // CHECK: insertelement <4 x float> {{.*}}, float 0.000000e+00, i32 1
  // CHECK: insertelement <4 x float> {{.*}}, float 0.000000e+00, i32 2
  // CHECK: insertelement <4 x float> {{.*}}, float 0.000000e+00, i32 3
  return _mm_set_ss(A);
}

__m128 test_mm_set1_ps(float A) {
  // CHECK-LABEL: test_mm_set1_ps
  // CHECK: insertelement <4 x float> undef, float {{.*}}, i32 0
  // CHECK: insertelement <4 x float> {{.*}}, float {{.*}}, i32 1
  // CHECK: insertelement <4 x float> {{.*}}, float {{.*}}, i32 2
  // CHECK: insertelement <4 x float> {{.*}}, float {{.*}}, i32 3
  return _mm_set1_ps(A);
}

void test_mm_setcsr(unsigned int A) {
  // CHECK-LABEL: test_mm_setcsr
  // CHECK: store i32
  // CHECK: call void @llvm.x86.sse.ldmxcsr(ptr {{.*}})
  _mm_setcsr(A);
}

__m128 test_mm_setr_ps(float A, float B, float C, float D) {
  // CHECK-LABEL: test_mm_setr_ps
  // CHECK: insertelement <4 x float> undef, float {{.*}}, i32 0
  // CHECK: insertelement <4 x float> {{.*}}, float {{.*}}, i32 1
  // CHECK: insertelement <4 x float> {{.*}}, float {{.*}}, i32 2
  // CHECK: insertelement <4 x float> {{.*}}, float {{.*}}, i32 3
  return _mm_setr_ps(A, B, C, D);
}

__m128 test_mm_setzero_ps(void) {
  // CHECK-LABEL: test_mm_setzero_ps
  // CHECK: store <4 x float> zeroinitializer
  return _mm_setzero_ps();
}

void test_mm_sfence(void) {
  // CHECK-LABEL: test_mm_sfence
  // CHECK: call void @llvm.x86.sse.sfence()
  _mm_sfence();
}

__m128 test_mm_shuffle_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_shuffle_ps
  // CHECK: shufflevector <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x i32> <i32 0, i32 0, i32 4, i32 4>
  return _mm_shuffle_ps(A, B, 0);
}

__m128 test_mm_sqrt_ps(__m128 x) {
  // CHECK-LABEL: test_mm_sqrt_ps
  // CHECK: call <4 x float> @llvm.sqrt.v4f32(<4 x float> {{.*}})
  return _mm_sqrt_ps(x);
}

__m128 test_mm_sqrt_ss(__m128 x) {
  // CHECK-LABEL: test_mm_sqrt_ss
  // CHECK: extractelement <4 x float> {{.*}}, i64 0
  // CHECK: call float @llvm.sqrt.f32(float {{.*}})
  // CHECK: insertelement <4 x float> {{.*}}, float {{.*}}, i64 0
  return _mm_sqrt_ss(x);
}

void test_mm_store_ps(float* x, __m128 y) {
  // CHECK-LABEL: test_mm_store_ps
  // CHECK: store <4 x float> %{{.*}}, ptr {{.*}}, align 16
  _mm_store_ps(x, y);
}

void test_mm_store_ps1(float* x, __m128 y) {
  // CHECK-LABEL: test_mm_store_ps1
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: store <4 x float> %{{.*}}, ptr %{{.*}}, align 16
  _mm_store_ps1(x, y);
}

void test_mm_store_ss(float* x, __m128 y) {
  // CHECK-LABEL: test_mm_store_ss
  // CHECK: extractelement <4 x float> {{.*}}, i32 0
  // CHECK: store float %{{.*}}, ptr {{.*}}, align 1{{$}}
  _mm_store_ss(x, y);
}

void test_mm_store1_ps(float* x, __m128 y) {
  // CHECK-LABEL: test_mm_store1_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: store <4 x float> %{{.*}}, ptr %{{.*}}, align 16
  _mm_store1_ps(x, y);
}

void test_mm_storeh_pi(__m64* x,  __m128 y) {
  // CHECK-LABEL: test_mm_storeh_pi
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <2 x i32> <i32 2, i32 3>
  // CHECK: store <2 x float> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  _mm_storeh_pi(x, y);
}

void test_mm_storel_pi(__m64* x,  __m128 y) {
  // CHECK-LABEL: test_mm_storel_pi
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // CHECK: store <2 x float> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  _mm_storel_pi(x, y);
}

void test_mm_storer_ps(float* x,  __m128 y) {
  // CHECK-LABEL: test_mm_storer_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  // CHECK: store <4 x float> %{{.*}}, ptr {{.*}}, align 16
  _mm_storer_ps(x, y);
}

void test_mm_storeu_ps(float* x,  __m128 y) {
  // CHECK-LABEL: test_mm_storeu_ps
  // CHECK: store <4 x float> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  // CHECK-NEXT: ret void
  _mm_storeu_ps(x, y);
}

void test_mm_stream_ps(float*A, __m128 B) {
  // CHECK-LABEL: test_mm_stream_ps
  // CHECK: store <4 x float> %{{.*}}, ptr %{{.*}}, align 16, !nontemporal
  _mm_stream_ps(A, B);
}

void test_mm_stream_ps_void(void *A, __m128 B) {
  // CHECK-LABEL: test_mm_stream_ps_void
  // CHECK: store <4 x float> %{{.*}}, ptr %{{.*}}, align 16, !nontemporal
  _mm_stream_ps(A, B);
}

__m128 test_mm_sub_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_sub_ps
  // CHECK: fsub <4 x float>
  return _mm_sub_ps(A, B);
}

__m128 test_mm_sub_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_sub_ss
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: fsub float
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i32 0
  return _mm_sub_ss(A, B);
}

void test_MM_TRANSPOSE4_PS(__m128 *A, __m128 *B, __m128 *C, __m128 *D) {
  // CHECK-LABEL: test_MM_TRANSPOSE4_PS
  // CHECK: shufflevector <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  // CHECK: shufflevector <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  // CHECK: shufflevector <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  // CHECK: shufflevector <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  // CHECK: shufflevector <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  // CHECK: shufflevector <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x i32> <i32 6, i32 7, i32 2, i32 3>
  // CHECK: shufflevector <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  // CHECK: shufflevector <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x i32> <i32 6, i32 7, i32 2, i32 3>
  _MM_TRANSPOSE4_PS(*A, *B, *C, *D);
}

int test_mm_ucomieq_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_ucomieq_ss
  // CHECK: call i32 @llvm.x86.sse.ucomieq.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_ucomieq_ss(A, B);
}

int test_mm_ucomige_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_ucomige_ss
  // CHECK: call i32 @llvm.x86.sse.ucomige.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_ucomige_ss(A, B);
}

int test_mm_ucomigt_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_ucomigt_ss
  // CHECK: call i32 @llvm.x86.sse.ucomigt.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_ucomigt_ss(A, B);
}

int test_mm_ucomile_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_ucomile_ss
  // CHECK: call i32 @llvm.x86.sse.ucomile.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_ucomile_ss(A, B);
}

int test_mm_ucomilt_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_ucomilt_ss
  // CHECK: call i32 @llvm.x86.sse.ucomilt.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_ucomilt_ss(A, B);
}

int test_mm_ucomineq_ss(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_ucomineq_ss
  // CHECK: call i32 @llvm.x86.sse.ucomineq.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_ucomineq_ss(A, B);
}

__m128 test_mm_undefined_ps(void) {
  // CHECK-LABEL: test_mm_undefined_ps
  // CHECK: ret <4 x float> zeroinitializer
  return _mm_undefined_ps();
}

__m128 test_mm_unpackhi_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_unpackhi_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  return _mm_unpackhi_ps(A, B);
}

__m128 test_mm_unpacklo_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_unpacklo_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  return _mm_unpacklo_ps(A, B);
}

__m128 test_mm_xor_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_xor_ps
  // CHECK: xor <4 x i32>
  return _mm_xor_ps(A, B);
}
