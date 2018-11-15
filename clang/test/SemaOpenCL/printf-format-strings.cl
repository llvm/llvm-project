// RUN: %clang_cc1 -cl-std=CL1.2 -fsyntax-only -verify %s

typedef __attribute__((ext_vector_type(2))) float float2;
typedef __attribute__((ext_vector_type(4))) float float4;
typedef __attribute__((ext_vector_type(4))) int int4;

int printf(__constant const char* st, ...) __attribute__((format(printf, 1, 2)));

kernel void format_v4f32(float4 arg)
{
    printf("%v4f\n", arg); // expected-no-diagnostics
}

kernel void format_v4f32_wrong_num_elts(float2 arg)
{
    printf("%v4f\n", arg); // expected-no-diagnostics
}

kernel void vector_precision_modifier_v4f32(float4 arg)
{
    printf("%.2v4f\n", arg); // expected-no-diagnostics
}

// FIXME: This should warn
kernel void format_missing_num_elts(float4 arg)
{
    printf("%vf\n", arg); // expected-no-diagnostics
}

// FIXME: This should warn
kernel void vector_precision_modifier_v4i32(int4 arg)
{
    printf("%.2v4f\n", arg); // expected-no-diagnostics
}
