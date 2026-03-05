// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -target-feature +avx512f -target-feature +avx512bw
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -target-feature +avx512f -target-feature +avx512bw
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll -target-feature +avx512f -target-feature +avx512bw
// RUN: FileCheck --input-file=%t.og.ll %s -check-prefix=OGCG

// Tests PSLLDQI byte shift intrinsics implementation in ClangIR
// Compares CIR emission, LLVM lowering, and original CodeGen output

typedef long long __m128i __attribute__((__vector_size__(16)));
typedef long long __m256i __attribute__((__vector_size__(32)));
typedef long long __m512i __attribute__((__vector_size__(64)));

// ============================================================================
// Core Functionality Tests
// ============================================================================

// CIR-LABEL: @_Z22test_pslldqi128_shift4Dv2_x
// LLVM-LABEL: @_Z22test_pslldqi128_shift4Dv2_x
// OGCG-LABEL: @_Z22test_pslldqi128_shift4Dv2_x
__m128i test_pslldqi128_shift4(__m128i a) {
    // Should shift left by 4 bytes, filling with zeros
    // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 16>) [#cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i, #cir.int<24> : !s32i, #cir.int<25> : !s32i, #cir.int<26> : !s32i, #cir.int<27> : !s32i] : !cir.vector<!s8i x 16>
    // LLVM: %{{.*}} = shufflevector <16 x i8> zeroinitializer, <16 x i8> %{{.*}}, <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
    // OGCG: %{{.*}} = shufflevector <16 x i8> zeroinitializer, <16 x i8> %{{.*}}, <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
    return __builtin_ia32_pslldqi128_byteshift(a, 4);
}

// CIR-LABEL: @_Z22test_pslldqi128_shift0Dv2_x
// LLVM-LABEL: @_Z22test_pslldqi128_shift0Dv2_x
// OGCG-LABEL: @_Z22test_pslldqi128_shift0Dv2_x
__m128i test_pslldqi128_shift0(__m128i a) {
    // Should return input unchanged (shift by 0)
    // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 16>) [#cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i, #cir.int<24> : !s32i, #cir.int<25> : !s32i, #cir.int<26> : !s32i, #cir.int<27> : !s32i, #cir.int<28> : !s32i, #cir.int<29> : !s32i, #cir.int<30> : !s32i, #cir.int<31> : !s32i] : !cir.vector<!s8i x 16>
    // LLVM: %{{.*}} = shufflevector <16 x i8> zeroinitializer, <16 x i8> %{{.*}}, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
    // OGCG: %{{.*}} = shufflevector <16 x i8> zeroinitializer, <16 x i8> %{{.*}}, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
    return __builtin_ia32_pslldqi128_byteshift(a, 0);
}

// CIR-LABEL: @_Z23test_pslldqi128_shift16Dv2_x
// LLVM-LABEL: @_Z23test_pslldqi128_shift16Dv2_x
// OGCG-LABEL: @_Z23test_pslldqi128_shift16Dv2_x
__m128i test_pslldqi128_shift16(__m128i a) {
    // Entire vector shifted out, should return zero
    // CIR: %{{.*}} = cir.const #cir.zero : !cir.vector<!s8i x 16>
    // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !cir.vector<!s8i x 16> -> !cir.vector<!s64i x 2>
    // LLVM: store <2 x i64> zeroinitializer, ptr %{{.*}}, align 16
    // OGCG: ret <2 x i64> zeroinitializer
    return __builtin_ia32_pslldqi128_byteshift(a, 16);
}

// ============================================================================
// 256-bit Tests (Two Independent 128-bit Lanes)
// ============================================================================

// CIR-LABEL: @_Z22test_pslldqi256_shift4Dv4_x
// LLVM-LABEL: @_Z22test_pslldqi256_shift4Dv4_x
// OGCG-LABEL: @_Z22test_pslldqi256_shift4Dv4_x
__m256i test_pslldqi256_shift4(__m256i a) {
    // Each 128-bit lane shifts independently by 4 bytes
    // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 32>) [#cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<32> : !s32i, #cir.int<33> : !s32i, #cir.int<34> : !s32i, #cir.int<35> : !s32i, #cir.int<36> : !s32i, #cir.int<37> : !s32i, #cir.int<38> : !s32i, #cir.int<39> : !s32i, #cir.int<40> : !s32i, #cir.int<41> : !s32i, #cir.int<42> : !s32i, #cir.int<43> : !s32i, #cir.int<28> : !s32i, #cir.int<29> : !s32i, #cir.int<30> : !s32i, #cir.int<31> : !s32i, #cir.int<48> : !s32i, #cir.int<49> : !s32i, #cir.int<50> : !s32i, #cir.int<51> : !s32i, #cir.int<52> : !s32i, #cir.int<53> : !s32i, #cir.int<54> : !s32i, #cir.int<55> : !s32i, #cir.int<56> : !s32i, #cir.int<57> : !s32i, #cir.int<58> : !s32i, #cir.int<59> : !s32i] : !cir.vector<!s8i x 32>
    // LLVM: %{{.*}} = shufflevector <32 x i8> zeroinitializer, <32 x i8> %{{.*}}, <32 x i32> <i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59>
    // OGCG: %{{.*}} = shufflevector <32 x i8> zeroinitializer, <32 x i8> %{{.*}}, <32 x i32> <i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59>
    return __builtin_ia32_pslldqi256_byteshift(a, 4);
}

// CIR-LABEL: @_Z23test_pslldqi256_shift16Dv4_x
// LLVM-LABEL: @_Z23test_pslldqi256_shift16Dv4_x
// OGCG-LABEL: @_Z23test_pslldqi256_shift16Dv4_x
__m256i test_pslldqi256_shift16(__m256i a) {
    // Both lanes completely shifted out, returns zero
    // CIR: %{{.*}} = cir.const #cir.zero : !cir.vector<!s8i x 32>
    // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !cir.vector<!s8i x 32> -> !cir.vector<!s64i x 4>
    // LLVM: store <4 x i64> zeroinitializer, ptr %{{.*}}, align 32
    // OGCG: ret <4 x i64> zeroinitializer
    return __builtin_ia32_pslldqi256_byteshift(a, 16);
}

// ============================================================================
// 512-bit Tests (Four Independent 128-bit Lanes)
// ============================================================================

// CIR-LABEL: @_Z22test_pslldqi512_shift4Dv8_x
// LLVM-LABEL: @_Z22test_pslldqi512_shift4Dv8_x
// OGCG-LABEL: @_Z22test_pslldqi512_shift4Dv8_x
__m512i test_pslldqi512_shift4(__m512i a) {
    // All 4 lanes shift independently by 4 bytes
    // CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 64>)
    // LLVM: shufflevector <64 x i8> zeroinitializer, <64 x i8> %{{.*}}, <64 x i32> <i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 28, i32 29, i32 30, i32 31, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 44, i32 45, i32 46, i32 47, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 60, i32 61, i32 62, i32 63, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123>
    // OGCG: shufflevector <64 x i8> zeroinitializer, <64 x i8> %{{.*}}, <64 x i32> <i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 28, i32 29, i32 30, i32 31, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 44, i32 45, i32 46, i32 47, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 60, i32 61, i32 62, i32 63, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123>
    return __builtin_ia32_pslldqi512_byteshift(a, 4);
}

// CIR-LABEL: @_Z23test_pslldqi512_shift16Dv8_x
// LLVM-LABEL: @_Z23test_pslldqi512_shift16Dv8_x
// OGCG-LABEL: @_Z23test_pslldqi512_shift16Dv8_x
__m512i test_pslldqi512_shift16(__m512i a) {
    // All 4 lanes completely cleared
    // CIR: %{{.*}} = cir.const #cir.zero : !cir.vector<!s8i x 64>
    // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !cir.vector<!s8i x 64> -> !cir.vector<!s64i x 8>
    // LLVM: store <8 x i64> zeroinitializer, ptr %{{.*}}, align 64
    // OGCG: ret <8 x i64> zeroinitializer
    return __builtin_ia32_pslldqi512_byteshift(a, 16);
}

// ============================================================================
// Input-Output Verification Tests
// ============================================================================

// Test with specific input values to verify correct data transformation
// CIR-LABEL: @_Z26test_input_output_shift4_1Dv2_x
// LLVM-LABEL: @_Z26test_input_output_shift4_1Dv2_x
// OGCG-LABEL: @_Z26test_input_output_shift4_1Dv2_x
__m128i test_input_output_shift4_1(__m128i a) {
    // Input:  [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] (bytes)
    // Shift left by 4 bytes (insert 4 zeros at start)
    // Output: [0, 0, 0, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4] (bytes)
    // CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 16>) [#cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i, #cir.int<24> : !s32i, #cir.int<25> : !s32i, #cir.int<26> : !s32i, #cir.int<27> : !s32i] : !cir.vector<!s8i x 16>
    // LLVM: shufflevector <16 x i8> zeroinitializer, <16 x i8> %{{.*}}, <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
    // OGCG: shufflevector <16 x i8> zeroinitializer, <16 x i8> %{{.*}}, <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
    return __builtin_ia32_pslldqi128_byteshift(a, 4);
}

// Test 256-bit lane independence with specific input pattern
// CIR-LABEL: @_Z34test_input_output_256_independenceDv4_x
// LLVM-LABEL: @_Z34test_input_output_256_independenceDv4_x
// OGCG-LABEL: @_Z34test_input_output_256_independenceDv4_x
__m256i test_input_output_256_independence(__m256i a) {
    // Input: Two 128-bit lanes, each with pattern [15,14,13,...,2,1,0]
    // Lane 0: [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    // Lane 1: [31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16]
    // After shift by 4 bytes:
    // Lane 0: [0, 0, 0, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]
    // Lane 1: [0, 0, 0, 0, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20]
    // CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 32>)
    // LLVM: shufflevector <32 x i8> zeroinitializer, <32 x i8> %{{.*}}, <32 x i32> <i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59>
    // OGCG: shufflevector <32 x i8> zeroinitializer, <32 x i8> %{{.*}}, <32 x i32> <i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59>
    return __builtin_ia32_pslldqi256_byteshift(a, 4);
}

// ============================================================================
// Edge Cases
// ============================================================================

// Test with concrete constant values to verify exact transformation
// CIR-LABEL: @_Z28test_concrete_input_constantv
// LLVM-LABEL: @_Z28test_concrete_input_constantv
// OGCG-LABEL: @_Z28test_concrete_input_constantv
__m128i test_concrete_input_constant() {
    // Create a known input pattern: 0x0F0E0D0C0B0A09080706050403020100
    // This represents bytes [15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
    __m128i input = (__m128i){0x0706050403020100LL, 0x0F0E0D0C0B0A0908LL};

    // Shift left by 4 bytes - should produce: 0x0B0A090807060504030201000000000
    // This represents bytes [0,0,0,0,15,14,13,12,11,10,9,8,7,6,5,4]
    __m128i result = __builtin_ia32_pslldqi128_byteshift(input, 4);

    // CIR: cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<!s8i x 16>) [#cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i, #cir.int<24> : !s32i, #cir.int<25> : !s32i, #cir.int<26> : !s32i, #cir.int<27> : !s32i] : !cir.vector<!s8i x 16>
    // LLVM: shufflevector <16 x i8> zeroinitializer, <16 x i8> %{{.*}}, <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
    // OGCG: shufflevector <16 x i8> zeroinitializer, <16 x i8> %{{.*}}, <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>

    return result;
}

// CIR-LABEL: @_Z22test_large_shift_valueDv2_x
// LLVM-LABEL: @_Z22test_large_shift_valueDv2_x
// OGCG-LABEL: @_Z22test_large_shift_valueDv2_x
__m128i test_large_shift_value(__m128i a) {
    // 240 & 0xFF = 240, so this should return zero (240 > 16)
    // CIR: %{{.*}} = cir.const #cir.zero : !cir.vector<!s8i x 16>
    // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !cir.vector<!s8i x 16> -> !cir.vector<!s64i x 2>
    // LLVM: store <2 x i64> zeroinitializer, ptr %{{.*}}, align 16
    // OGCG: ret <2 x i64> zeroinitializer
    return __builtin_ia32_pslldqi128_byteshift(a, 240);
}
