// RUN: %clang_cc1 -cl-std=CL1.2 -cl-ext=+cl_khr_fp64 -fsyntax-only -verify %s
// RUN: %clang_cc1 -cl-std=CL1.2 -cl-ext=-cl_khr_fp64 -fsyntax-only -verify %s

typedef __attribute__((ext_vector_type(4))) half half4;

typedef __attribute__((ext_vector_type(2))) float float2;
typedef __attribute__((ext_vector_type(4))) float float4;

#ifdef cl_khr_fp64
typedef __attribute__((ext_vector_type(4))) double double4;
#endif

typedef __attribute__((ext_vector_type(4))) char char4;
typedef __attribute__((ext_vector_type(4))) unsigned char uchar4;

typedef __attribute__((ext_vector_type(4))) short short4;
typedef __attribute__((ext_vector_type(4))) unsigned short ushort4;

typedef __attribute__((ext_vector_type(2))) int int2;
typedef __attribute__((ext_vector_type(4))) int int4;
typedef __attribute__((ext_vector_type(16))) int int16;

typedef __attribute__((ext_vector_type(4))) long long4;
typedef __attribute__((ext_vector_type(4))) unsigned int uint4;
typedef __attribute__((ext_vector_type(4))) unsigned long ulong4;

int printf(__constant const char* st, ...) __attribute__((format(printf, 1, 2)));


#ifdef cl_khr_fp64
kernel void format_v4f64(half4 arg_h, float4 arg_f, double4 arg_d)
{
  printf("%v4lf", arg_d);
  printf("%v4lf", arg_f); // expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'float4' (vector of 4 'float' values)}}
  printf("%v4lf", arg_h); // expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'half4' (vector of 4 'half' values)}}

  printf("%v4lF", arg_d);
  printf("%v4lF", arg_f); // expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'float4' (vector of 4 'float' values)}}
  printf("%v4lF", arg_h); // expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'half4' (vector of 4 'half' values)}}

  printf("%v4le", arg_d);
  printf("%v4le", arg_f); // expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'float4' (vector of 4 'float' values)}}
  printf("%v4le", arg_h); // expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'half4' (vector of 4 'half' values)}}

  printf("%v4lE", arg_d);
  printf("%v4lE", arg_f); // expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'float4' (vector of 4 'float' values)}}
  printf("%v4lE", arg_h); // expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'half4' (vector of 4 'half' values)}}

  printf("%v4lg", arg_d);
  printf("%v4lg", arg_f); // expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'float4' (vector of 4 'float' values)}}
  printf("%v4lg", arg_h); // expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'half4' (vector of 4 'half' values)}}

  printf("%v4lG", arg_d);
  printf("%v4lG", arg_f); // expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'float4' (vector of 4 'float' values)}}
  printf("%v4lG", arg_h); // expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'half4' (vector of 4 'half' values)}}

  printf("%v4la", arg_d);
  printf("%v4la", arg_f); // expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'float4' (vector of 4 'float' values)}}
  printf("%v4la", arg_h); // expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'half4' (vector of 4 'half' values)}}

  printf("%v4lA", arg_d);
  printf("%v4lA", arg_f); // expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'float4' (vector of 4 'float' values)}}
  printf("%v4lA", arg_h); // expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'half4' (vector of 4 'half' values)}}
}

kernel void format_v4f16(half4 arg_h, float4 arg_f, double4 arg_d)
{
  printf("%v4hf\n", arg_d); // expected-warning{{format specifies type '__fp16 __attribute__((ext_vector_type(4)))' but the argument has type 'double4' (vector of 4 'double' values)}}
  printf("%v4hf\n", arg_f); // expected-warning{{format specifies type '__fp16 __attribute__((ext_vector_type(4)))' but the argument has type 'float4' (vector of 4 'float' values)}}
  printf("%v4hf\n", arg_h);
}

kernel void no_length_modifier_scalar_fp(float f) {
  printf("%hf", f); // expected-warning{{length modifier 'h' results in undefined behavior or no effect with 'f' conversion specifier}}
  printf("%hlf", f); // expected-warning{{length modifier 'hl' results in undefined behavior or no effect with 'f' conversion specifier}}
  printf("%lf", f); // expected-warning{{length modifier 'l' results in undefined behavior or no effect with 'f' conversion specifier}}
}

#endif

kernel void format_v4f32(float4 arg)
{
#ifdef cl_khr_fp64
    printf("%v4f\n", arg); // expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'float4' (vector of 4 'float' values)}}

    // Precision modifier
    printf("%.2v4f\n", arg); //expected-warning{{format specifies type 'double __attribute__((ext_vector_type(4)))' but the argument has type 'float4' (vector of 4 'float' values)}}
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


kernel void printf_int_length_modifiers(char4 arg_c, short4 arg_s, int4 arg_i, long4 arg_l, uchar4 arg_uc, ushort4 arg_us, uint4 arg_ui, ulong4 arg_ul) {
  printf("%v4hhd", arg_c);
  printf("%v4hhd", arg_s);
  printf("%v4hhd", arg_i);
  printf("%v4hhd", arg_l);

  printf("%v4hd", arg_c); // expected-warning{{format specifies type 'short __attribute__((ext_vector_type(4)))' but the argument has type 'char4' (vector of 4 'char' values)}}
  printf("%v4hd", arg_s);
  printf("%v4hd", arg_i); // expected-warning{{format specifies type 'short __attribute__((ext_vector_type(4)))' but the argument has type 'int4' (vector of 4 'int' values)}}
  printf("%v4hd", arg_l); // expected-warning{{format specifies type 'short __attribute__((ext_vector_type(4)))' but the argument has type 'long4' (vector of 4 'long' values)}}

  printf("%v4hld", arg_c); // expected-warning{{format specifies type 'int __attribute__((ext_vector_type(4)))' but the argument has type 'char4' (vector of 4 'char' values)}}
  printf("%v4hld", arg_s); // expected-warning{{format specifies type 'int __attribute__((ext_vector_type(4)))' but the argument has type 'short4' (vector of 4 'short' values)}}
  printf("%v4hld", arg_i);
  printf("%v4hld", arg_l); // expected-warning{{format specifies type 'int __attribute__((ext_vector_type(4)))' but the argument has type 'long4' (vector of 4 'long' values)}}

  printf("%v4ld", arg_c); // expected-warning{{format specifies type 'long __attribute__((ext_vector_type(4)))' but the argument has type 'char4' (vector of 4 'char' values)}}
  printf("%v4ld", arg_s); // expected-warning{{format specifies type 'long __attribute__((ext_vector_type(4)))' but the argument has type 'short4' (vector of 4 'short' values)}}
  printf("%v4ld", arg_i); // expected-warning{{format specifies type 'long __attribute__((ext_vector_type(4)))' but the argument has type 'int4' (vector of 4 'int' values)}}
  printf("%v4ld", arg_l);



  printf("%v4hhu", arg_uc);
  printf("%v4hhu", arg_us); // expected-warning{{format specifies type 'unsigned char __attribute__((ext_vector_type(4)))' but the argument has type 'ushort4' (vector of 4 'unsigned short' values)}}
  printf("%v4hhu", arg_ui); // expected-warning{{format specifies type 'unsigned char __attribute__((ext_vector_type(4)))' but the argument has type 'uint4' (vector of 4 'unsigned int' values)}}
  printf("%v4hhu", arg_ul); // expected-warning{{format specifies type 'unsigned char __attribute__((ext_vector_type(4)))' but the argument has type 'ulong4' (vector of 4 'unsigned long' values)}}

  printf("%v4hu", arg_uc); // expected-warning{{format specifies type 'unsigned short __attribute__((ext_vector_type(4)))' but the argument has type 'uchar4' (vector of 4 'unsigned char' values)}}
  printf("%v4hu", arg_us);
  printf("%v4hu", arg_ui); // expected-warning{{format specifies type 'unsigned short __attribute__((ext_vector_type(4)))' but the argument has type 'uint4' (vector of 4 'unsigned int' values)}}
  printf("%v4hu", arg_ul); // expected-warning{{format specifies type 'unsigned short __attribute__((ext_vector_type(4)))' but the argument has type 'ulong4' (vector of 4 'unsigned long' values)}}

  printf("%v4hlu", arg_uc); // expected-warning{{format specifies type 'unsigned int __attribute__((ext_vector_type(4)))' but the argument has type 'uchar4' (vector of 4 'unsigned char' values)}}
  printf("%v4hlu", arg_us); // expected-warning{{format specifies type 'unsigned int __attribute__((ext_vector_type(4)))' but the argument has type 'ushort4' (vector of 4 'unsigned short' values)}}
  printf("%v4hlu", arg_ui);
  printf("%v4hlu", arg_ul); // expected-warning{{format specifies type 'unsigned int __attribute__((ext_vector_type(4)))' but the argument has type 'ulong4' (vector of 4 'unsigned long' values)}}

  printf("%v4lu", arg_uc); // expected-warning{{format specifies type 'unsigned long __attribute__((ext_vector_type(4)))' but the argument has type 'uchar4' (vector of 4 'unsigned char' values)}}
  printf("%v4lu", arg_us); // expected-warning{{format specifies type 'unsigned long __attribute__((ext_vector_type(4)))' but the argument has type 'ushort4' (vector of 4 'unsigned short' values)}}
  printf("%v4lu", arg_ui); // expected-warning{{format specifies type 'unsigned long __attribute__((ext_vector_type(4)))' but the argument has type 'uint4' (vector of 4 'unsigned int' values)}}
  printf("%v4lu", arg_ul);


  printf("%v4n", &arg_i); // expected-warning{{invalid conversion specifier 'n'}}
  printf("%v4hln", &arg_i); // expected-warning{{invalid conversion specifier 'n'}}
}
