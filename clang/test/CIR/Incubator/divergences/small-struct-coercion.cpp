// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Small structs (8 bytes or less) should be coerced to integer types per x86_64 ABI.
// This is a divergence from CodeGen's behavior.
//
// Per the System V AMD64 ABI:
// - Structs up to 8 bytes containing only INTEGER class should be passed in one register
// - The struct should be coerced to i64 for 8-byte structs
// - This applies to both parameters and return values
//
// CodeGen correctly coerces small structs:
//   define i64 @return_small()    // Returns i64, not struct
//
// CIR incorrectly returns the struct directly:
//   define %struct.SmallStruct @return_small()   // Wrong!
//
// This affects:
// - Struct returns (8 bytes or less)
// - Struct parameters (8 bytes or less)
// - Any struct containing two i32 fields (8 bytes total)
// - Struct with single i64 field
// - Struct with single double field (floating point class, different rules)
//
// Impact: ABI incompatibility between ClangIR-compiled and CodeGen-compiled code

// DIFF: -define {{.*}} i64 @{{.*}}return_small
// DIFF: +define {{.*}} %struct.SmallStruct @{{.*}}return_small

struct SmallStruct {
    int a, b;  // 8 bytes total
};

// Should return i64 per ABI, but CIR returns struct
SmallStruct return_small() {
    return {1, 2};
}

// Should take i64 parameter per ABI, but CIR takes struct
int take_small(SmallStruct s) {
    return s.a + s.b;
}

// Test with 4-byte struct (should be coerced to i32)
struct TinyStruct {
    int x;  // 4 bytes
};

// DIFF: -define {{.*}} i32 @{{.*}}return_tiny
// DIFF: +define {{.*}} %struct.TinyStruct @{{.*}}return_tiny

TinyStruct return_tiny() {
    return {42};
}

// Test with struct containing long long (8 bytes, should be i64)
struct LongStruct {
    long long value;
};

// DIFF: -define {{.*}} i64 @{{.*}}return_long
// DIFF: +define {{.*}} %struct.LongStruct @{{.*}}return_long

LongStruct return_long() {
    return {123456789LL};
}
