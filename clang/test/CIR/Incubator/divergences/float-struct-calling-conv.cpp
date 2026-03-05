// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Floating-point struct calling conventions diverge from CodeGen.
//
// Per the System V AMD64 ABI, structs containing only floating-point types
// are classified as SSE and passed in XMM registers, with special handling.
//
// This affects:
// - Structs with single float (should be in XMM register)
// - Structs with double (should be in XMM register)
// - Structs with two floats (should be in XMM registers)
// - Mixed integer/float structs (different classification)
//
// Impact: May affect ABI compatibility for floating-point structs

// DIFF: Check for differences in float/double struct handling

// Single float struct
struct FloatStruct {
    float f;
};

FloatStruct return_float() {
    return {3.14f};
}

int test_float() {
    FloatStruct s = return_float();
    return s.f > 3.0f ? 1 : 0;
}

// Single double struct
struct DoubleStruct {
    double d;
};

DoubleStruct return_double() {
    return {3.14159};
}

int test_double() {
    DoubleStruct s = return_double();
    return s.d > 3.0 ? 1 : 0;
}

// Two floats
struct TwoFloats {
    float a, b;
};

TwoFloats return_two_floats() {
    return {1.0f, 2.0f};
}

int test_two_floats() {
    TwoFloats s = return_two_floats();
    return s.a + s.b > 2.5f ? 1 : 0;
}

// Mixed int and float
struct MixedStruct {
    int i;
    float f;
};

MixedStruct return_mixed() {
    return {42, 3.14f};
}

int test_mixed() {
    MixedStruct s = return_mixed();
    return s.i;
}

// Three floats (larger than 16 bytes, needs sret)
struct ThreeFloats {
    float a, b, c;
};

ThreeFloats return_three_floats() {
    return {1.0f, 2.0f, 3.0f};
}

int test_three_floats() {
    ThreeFloats s = return_three_floats();
    return s.a + s.b + s.c > 5.0f ? 1 : 0;
}

// Four floats (definitely needs sret)
struct FourFloats {
    float a, b, c, d;
};

FourFloats return_four_floats() {
    return {1.0f, 2.0f, 3.0f, 4.0f};
}

int test_four_floats() {
    FourFloats s = return_four_floats();
    return s.a + s.b + s.c + s.d > 9.0f ? 1 : 0;
}
