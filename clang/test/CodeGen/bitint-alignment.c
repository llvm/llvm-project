// RUN: %clang_cc1 -triple=i386-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-I386
// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-X86_64

// Test that _BitInt alignment is handled correctly, especially for 64-bit types
// This test verifies the fix for the alignment discrepancy between Clang and GCC

#include <stdint.h>

// Test variables
long long vll;
_BitInt(64) vbi;

// Test struct to capture alignment values
struct result {
    int uppercase_alignof_bi64;      // _Alignof(_BitInt(64))
    int uppercase_alignof_vbi;       // _Alignof(vbi)
    int uppercase_alignof_longlong;  // _Alignof(long long)
    int uppercase_alignof_vll;       // _Alignof(vll)
    int lowercase_alignof_bi64;      // __alignof(_BitInt(64))
    int lowercase_alignof_vbi;       // __alignof(vbi)
    int lowercase_alignof_longlong;  // __alignof(long long)
    int lowercase_alignof_vll;       // __alignof(vll)
} result = {
    .uppercase_alignof_bi64 = _Alignof(_BitInt(64)),
    .uppercase_alignof_vbi = _Alignof(vbi),
    .uppercase_alignof_longlong = _Alignof(long long),
    .uppercase_alignof_vll = _Alignof(vll),
    .lowercase_alignof_bi64 = __alignof(_BitInt(64)),
    .lowercase_alignof_vbi = __alignof(vbi),
    .lowercase_alignof_longlong = __alignof(long long),
    .lowercase_alignof_vll = __alignof(vll),
};

// Test struct layout to verify alignment is respected
struct test_struct {
    char c;
    _BitInt(64) bi;
    long long ll;
    double d;
};

// Function to test alignment calculations
void test_alignments() {
    // These should all compile and have correct alignment
    _Static_assert(_Alignof(_BitInt(8)) == 1, "_BitInt(8) should have 1-byte alignment");
    _Static_assert(_Alignof(_BitInt(16)) == 2, "_BitInt(16) should have 2-byte alignment");
    _Static_assert(_Alignof(_BitInt(32)) == 4, "_BitInt(32) should have 4-byte alignment");
    
    // The key test: _BitInt(64) should have 8-byte alignment on both i386 and x86_64
    _Static_assert(_Alignof(_BitInt(64)) == 8, "_BitInt(64) should have 8-byte alignment");
    
    // Verify struct alignment
    _Static_assert(_Alignof(struct test_struct) >= 8, "test_struct should have at least 8-byte alignment");
}

// CHECK-I386: @result = global %struct.result { i32 4, i32 4, i32 4, i32 8, i32 8, i32 8, i32 8, i32 8 }
// CHECK-X86_64: @result = global %struct.result { i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8 }

// Verify that the struct layout respects alignment
// CHECK-I386: %struct.test_struct = type { i8, [3 x i8], i64, i64, double }
// CHECK-X86_64: %struct.test_struct = type { i8, [7 x i8], i64, i64, double } 