// RUN: %clang_cc1 -cl-std=CL1.2 -cl-ext=+cl_khr_fp64 -fsyntax-only -verify %s
// RUN: %clang_cc1 -cl-std=CL1.2 -cl-ext=-cl_khr_fp64 -fsyntax-only -verify %s

typedef __attribute__((ext_vector_type(2))) float float2;
typedef __attribute__((ext_vector_type(4))) float float4;

typedef __attribute__((ext_vector_type(2))) int int2;
typedef __attribute__((ext_vector_type(4))) int int4;
typedef __attribute__((ext_vector_type(16))) int int16;

int printf(__constant const char* st, ...) __attribute__((format(printf, 1, 2)));

kernel void format_v4f32(float4 arg)
{
#ifdef cl_khr_fp64
    printf("%v4f\n", arg);

    // Precision modifier
    printf("%.2v4f\n", arg);
#else
    // FIXME: These should not warn, and the type should be expected to be float.
    printf("%v4f\n", arg);  // expected-warning {{double __attribute__((ext_vector_type(4)))' but the argument has type 'float4' (vector of 4 'float' values)}}

    // Precision modifier
    printf("%.2v4f\n", arg); // expected-warning {{double __attribute__((ext_vector_type(4)))' but the argument has type 'float4' (vector of 4 'float' values)}}
#endif
}

kernel void format_only_v(int arg)
{
    printf("%v", arg); // expected-warning {{incomplete format specifier}}
}

kernel void format_missing_num(int arg)
{
    printf("%v4", arg); // expected-warning {{incomplete format specifier}}
}

kernel void format_not_num(int arg)
{
    printf("%vNd", arg); // expected-warning {{incomplete format specifier}}
    printf("%v*d", arg); // expected-warning {{incomplete format specifier}}
}

kernel void format_v16i32(int16 arg)
{
    printf("%v16d\n", arg);
}

kernel void format_v4i32_scalar(int arg)
{
   printf("%v4d\n", arg); // expected-warning  {{format specifies type 'int __attribute__((ext_vector_type(4)))' but the argument has type 'int'}}
}

kernel void format_v4i32_wrong_num_elts_2_to_4(int2 arg)
{
    printf("%v4d\n", arg); // expected-warning {{format specifies type 'int __attribute__((ext_vector_type(4)))' but the argument has type 'int2' (vector of 2 'int' values)}}
}

kernel void format_missing_num_elts_format(int4 arg)
{
    printf("%vd\n", arg); // expected-warning {{incomplete format specifier}}
}

kernel void format_v4f32_scalar(float arg)
{
    printf("%v4f\n", arg); // expected-warning {{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'float'}}
}

kernel void format_v4f32_wrong_num_elts(float2 arg)
{
    printf("%v4f\n", arg); // expected-warning {{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'float2' (vector of 2 'float' values)}}
}

kernel void format_missing_num_elts(float4 arg)
{
    printf("%vf\n", arg); // expected-warning {{incomplete format specifier}}
}

kernel void vector_precision_modifier_v4i32_to_v4f32(int4 arg)
{
    printf("%.2v4f\n", arg); // expected-warning {{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'int4' (vector of 4 'int' values)}}
}

kernel void invalid_Y(int4 arg)
{
    printf("%v4Y\n", arg); // expected-warning {{invalid conversion specifier 'Y'}}
}

// FIXME: This should warn
kernel void crash_on_s(int4 arg)
{
    printf("%v4s\n", arg);
}
