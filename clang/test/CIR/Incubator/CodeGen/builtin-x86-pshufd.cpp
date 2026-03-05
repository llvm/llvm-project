// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Test that __builtin_ia32_pshufd and __builtin_ia32_vpermilp generates correct CIR vec.shuffle operations
// This verifies the fix for SIMD intrinsic support that was previously NYI

typedef int __v4si __attribute__((__vector_size__(16)));
typedef float __v4sf __attribute__((__vector_size__(16)));
typedef double __v2df __attribute__((__vector_size__(16)));
typedef float __v8sf __attribute__((__vector_size__(32)));
typedef double __v4df __attribute__((__vector_size__(32)));
typedef float __v16sf __attribute__((__vector_size__(64)));
typedef double __v8df __attribute__((__vector_size__(64)));

typedef __v4si __m128i;
typedef __v4sf __m128;
typedef __v2df __m128d;
typedef __v8sf __m256;
typedef __v4df __m256d;
typedef __v16sf __m512;
typedef __v8df __m512d;

// CHECK-LABEL: @_Z11test_pshufdv
void test_pshufd() {
    __m128i vec = {1, 2, 3, 4};
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s32i x 4>) [#cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<!s32i x 4>
    __m128i result = __builtin_ia32_pshufd(vec, 0x4E);
}

// CHECK-LABEL: @_Z19test_different_maskv  
void test_different_mask() {
    __m128i vec = {10, 20, 30, 40};
    // Test different immediate value: 0x1B = 00011011 = [3,2,1,0] reversed
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s32i x 4>) [#cir.int<3> : !s32i, #cir.int<2> : !s32i, #cir.int<1> : !s32i, #cir.int<0> : !s32i] : !cir.vector<!s32i x 4>
    __m128i result = __builtin_ia32_pshufd(vec, 0x1B);
}

// CHECK-LABEL: @_Z9test_casev
void test_case() {
    __m128i p0 = {1, 2, 3, 4};
    
    // This reproduces the exact pattern from stb_image.h:2685 that was failing:
    // _mm_storel_epi64((__m128i *) out, _mm_shuffle_epi32(p0, 0x4e));
    // Which expands to: __builtin_ia32_pshufd(p0, 0x4e)
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s32i x 4>) [#cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<!s32i x 4>
    __m128i out_vec = __builtin_ia32_pshufd(p0, 0x4e);
}

// CHECK-LABEL: @_Z15test_vpermilps4v
void test_vpermilps4() {
    __m128 vec = {1.0f, 2.0f, 3.0f, 4.0f};
    // vpermilps with immediate 0x4E = 01001110 = [1,3,2,0] for 4 elements
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} :  !cir.vector<!cir.float x 4>) [#cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<!cir.float x 4>  
    __m128 result = __builtin_ia32_vpermilps(vec, 0x4E);
}

// CHECK-LABEL: @_Z15test_vpermilpd2v
void test_vpermilpd2() {
    __m128d vec = {1.0, 2.0};
    // vpermilpd with immediate 0x1 = 01 = [1,0] for 2 elements
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.double x 2>) [#cir.int<1> : !s32i, #cir.int<0> : !s32i] : !cir.vector<!cir.double x 2>
    __m128d result = __builtin_ia32_vpermilpd(vec, 0x1);
}

// CHECK-LABEL: @_Z17test_vpermilps256v
void test_vpermilps256() {
    __m256 vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    // vpermilps256 with immediate 0x1B = 00011011 = [3,2,1,0] for each 128-bit lane
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.float x 8>) [#cir.int<3> : !s32i, #cir.int<2> : !s32i, #cir.int<1> : !s32i, #cir.int<0> : !s32i, #cir.int<7> : !s32i, #cir.int<6> : !s32i, #cir.int<5> : !s32i, #cir.int<4> : !s32i] : !cir.vector<!cir.float x 8>
    __m256 result = __builtin_ia32_vpermilps256(vec, 0x1B);
}

// CHECK-LABEL: @_Z17test_vpermilpd256v
void test_vpermilpd256() {
    __m256d vec = {1.0, 2.0, 3.0, 4.0};
    // vpermilpd256 with immediate 0x5 = 0101 = [1,0,1,0] for 4 elements
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.double x 4>) [#cir.int<1> : !s32i, #cir.int<0> : !s32i, #cir.int<3> : !s32i, #cir.int<2> : !s32i] : !cir.vector<!cir.double x 4> 
    __m256d result = __builtin_ia32_vpermilpd256(vec, 0x5);
}

// CHECK-LABEL: @_Z17test_vpermilps512v
void test_vpermilps512() {
    __m512 vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 
                  9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    // vpermilps512 with immediate 0x4E = 01001110 = [1,3,2,0] for each 128-bit lane
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.float x 16>) [#cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i] : !cir.vector<!cir.float x 16>
    __m512 result = __builtin_ia32_vpermilps512(vec, 0x4E);
}

// CHECK-LABEL: @_Z17test_vpermilpd512v
void test_vpermilpd512() {
    __m512d vec = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    // vpermilpd512 with immediate 0x55 = 01010101 = [1,0,1,0,1,0,1,0] for 8 elements
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.double x 8>) [#cir.int<1> : !s32i, #cir.int<0> : !s32i, #cir.int<3> : !s32i, #cir.int<2> : !s32i, #cir.int<5> : !s32i, #cir.int<4> : !s32i, #cir.int<7> : !s32i, #cir.int<6> : !s32i] : !cir.vector<!cir.double x 8> 
    __m512d result = __builtin_ia32_vpermilpd512(vec, 0x55);
}

// Test different immediate values
// CHECK-LABEL: @_Z24test_vpermilps_differentv
void test_vpermilps_different() {
    __m128 vec = {10.0f, 20.0f, 30.0f, 40.0f};
    // Test different immediate value: 0x1B = 00011011 = [3,2,1,0] reversed
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.float x 4>) [#cir.int<3> : !s32i, #cir.int<2> : !s32i, #cir.int<1> : !s32i, #cir.int<0> : !s32i] : !cir.vector<!cir.float x 4> 
    __m128 result = __builtin_ia32_vpermilps(vec, 0x1B);
}

// CHECK-LABEL: @_Z24test_vpermilpd_differentv
void test_vpermilpd_different() {
    __m128d vec = {100.0, 200.0};
    // Test immediate 0x0 = 00 = [0,0] - duplicate first element
    // CHECK: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!cir.double x 2>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i] : !cir.vector<!cir.double x 2>
    __m128d result = __builtin_ia32_vpermilpd(vec, 0x0);
}
