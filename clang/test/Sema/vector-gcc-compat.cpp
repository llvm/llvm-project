// RUN: %clang_cc1 %s -verify -fsyntax-only -Weverything -std=c++11 -triple x86_64-apple-darwin10

// Test the compatibility of clang++'s vector extensions with g++'s vector
// extensions. In comparison to the extensions available in C, the !, ?:, && and
// || operators work on vector types.

typedef long long v2i64 __attribute__((vector_size(16))); // expected-warning {{'long long' is incompatible with C++98}}
typedef int v2i32 __attribute__((vector_size(8)));
typedef short v2i16 __attribute__((vector_size(4)));
typedef char v2i8 __attribute__((vector_size(2)));

typedef unsigned long long v2u64 __attribute__((vector_size(16))); // expected-warning {{'long long' is incompatible with C++98}}
typedef unsigned int v2u32 __attribute__((vector_size(8)));
typedef unsigned short v2u16 __attribute__((vector_size(4)));
typedef unsigned char v2u8 __attribute__((vector_size(2)));

typedef float v4f32 __attribute__((vector_size(16)));
typedef double v2f64 __attribute__((vector_size(16)));
typedef double v4f64 __attribute__((vector_size(32)));
typedef int v4i32 __attribute((vector_size(16)));

void arithmeticTest(void);
void logicTest(void);
void comparisonTest(void);
void floatTestSignedType(char a, short b, int c, long long d); // expected-warning {{'long long' is incompatible with C++98}}
void floatTestUnsignedType(unsigned char a, unsigned short b, unsigned int c,
                           unsigned long long d); // expected-warning {{'long long' is incompatible with C++98}}
void floatTestConstant(void);
void intTestType(char a, short b, int c, long long d); // expected-warning {{'long long' is incompatible with C++98}}
void intTestTypeUnsigned(unsigned char a, unsigned short b, unsigned int c,
                         unsigned long long d); // expected-warning {{'long long' is incompatible with C++98}}
void uintTestType(char a, short b, int c, long long d); // expected-warning {{'long long' is incompatible with C++98}}
void uintTestTypeUnsigned(unsigned char a, unsigned short b, unsigned int c,
                          unsigned long long d); // expected-warning {{'long long' is incompatible with C++98}}
void uintTestConstant(v2u64 v2u64_a, v2u32 v2u32_a, v2u16 v2u16_a, v2u8 v2u8_a);
void intTestConstant(v2i64 v2i64_a, v2i32 v2i32_a, v2i16 v2i16_a, v2i8 v2i8_a);

void arithmeticTest(void) {
  v2i64 v2i64_a = (v2i64){0, 1}; // expected-warning {{compound literals are a C99-specific feature}}
  v2i64 v2i64_r;

  v2i64_r = v2i64_a + 1;
  v2i64_r = v2i64_a - 1;
  v2i64_r = v2i64_a * 1;
  v2i64_r = v2i64_a / 1;
  v2i64_r = v2i64_a % 1;

  v2i64_r = 1 + v2i64_a;
  v2i64_r = 1 - v2i64_a;
  v2i64_r = 1 * v2i64_a;
  v2i64_r = 1 / v2i64_a;
  v2i64_r = 1 % v2i64_a;

  v2i64_a += 1;
  v2i64_a -= 1;
  v2i64_a *= 1;
  v2i64_a /= 1;
  v2i64_a %= 1;
}

void comparisonTest(void) {
  v2i64 v2i64_a = (v2i64){0, 1}; // expected-warning {{compound literals are a C99-specific feature}}
  v2i64 v2i64_r;

  v2i64_r = v2i64_a == 1;
  v2i64_r = v2i64_a != 1;
  v2i64_r = v2i64_a < 1;
  v2i64_r = v2i64_a > 1;
  v2i64_r = v2i64_a <= 1;
  v2i64_r = v2i64_a >= 1;

  v2i64_r = 1 == v2i64_a;
  v2i64_r = 1 != v2i64_a;
  v2i64_r = 1 < v2i64_a;
  v2i64_r = 1 > v2i64_a;
  v2i64_r = 1 <= v2i64_a;
  v2i64_r = 1 >= v2i64_a;
}

void logicTest(void) {
  v2i64 v2i64_a = (v2i64){0, 1}; // expected-warning {{compound literals are a C99-specific feature}}
  v2i64 v2i64_b = (v2i64){2, 1}; // expected-warning {{compound literals are a C99-specific feature}}
  v2i64 v2i64_c = (v2i64){3, 1}; // expected-warning {{compound literals are a C99-specific feature}}
  v2i64 v2i64_r;

  v2i64_r = !v2i64_a;  // expected-error {{invalid argument type 'v2i64' (vector of 2 'long long' values) to unary expression}}
  v2i64_r = ~v2i64_a;

  v2i64_r = v2i64_a ? v2i64_b : v2i64_c; // expected-error {{value of type 'v2i64' (vector of 2 'long long' values) is not contextually convertible to 'bool'}}

  v2i64_r = v2i64_a & 1;
  v2i64_r = v2i64_a | 1;
  v2i64_r = v2i64_a ^ 1;

  v2i64_r = 1 & v2i64_a;
  v2i64_r = 1 | v2i64_a;
  v2i64_r = 1 ^ v2i64_a;
  v2i64_a &= 1;
  v2i64_a |= 1;
  v2i64_a ^= 1;

  v2i64_r = v2i64_a && 1;
  v2i64_r = v2i64_a || 1;

  v2i64_r = v2i64_a << 1;
  v2i64_r = v2i64_a >> 1;

  v2i64_r = 1 << v2i64_a;
  v2i64_r = 1 >> v2i64_a;

  v2i64_a <<= 1;
  v2i64_a >>= 1;
}

// For operations with floating point types, we check that interger constants
// can be respresented, or failing that checking based on the integer types.
void floatTestConstant(void) {
  // Test that constants added to floats must be expressible as floating point
  // numbers.
  v4f32 v4f32_a = {0.4f, 0.4f, 0.4f, 0.4f};
  v4f32_a = v4f32_a + 1;
  v4f32_a = v4f32_a + 0xFFFFFF;
  v4f32_a = v4f32_a + (-1567563LL); // expected-warning {{'long long' is incompatible with C++98}}
  v4f32_a = v4f32_a + (16777208);
  v4f32_a = v4f32_a + (16777219); // expected-error {{cannot convert between scalar type 'int' and vector type 'v4f32' (vector of 4 'float' values) as implicit conversion would cause truncation}}
}

void floatTestConstantComparison(void);
void doubleTestConstantComparison(void);

void floatTestConstantComparison(void) {
  v4f32 v4f32_a = {0.4f, 0.4f, 0.4f, 0.4f};
  v4i32 v4i32_r;
  v4i32_r = v4f32_a > 0.4f;
  v4i32_r = v4f32_a >= 0.4f;
  v4i32_r = v4f32_a < 0.4f;
  v4i32_r = v4f32_a <= 0.4f;
  v4i32_r = v4f32_a == 0.4f; // expected-warning {{comparing floating point with == or != is unsafe}}
  v4i32_r = v4f32_a != 0.4f; // expected-warning {{comparing floating point with == or != is unsafe}}
}

void doubleTestConstantComparison(void) {
  v2f64 v2f64_a = {0.4, 0.4};
  v2i64 v2i64_r;
  v2i64_r = v2f64_a > 0.4;
  v2i64_r = v2f64_a >= 0.4;
  v2i64_r = v2f64_a < 0.4;
  v2i64_r = v2f64_a <= 0.4;
  v2i64_r = v2f64_a == 0.4; // expected-warning {{comparing floating point with == or != is unsafe}}
  v2i64_r = v2f64_a != 0.4; // expected-warning {{comparing floating point with == or != is unsafe}}
}

void floatTestUnsignedType(unsigned char a, unsigned short b, unsigned int c,
                           unsigned long long d) { // expected-warning {{'long long' is incompatible with C++98}}
  v4f32 v4f32_a = {0.4f, 0.4f, 0.4f, 0.4f};
  v4f64 v4f64_b = {0.4, 0.4, 0.4, 0.4};

  v4f32_a = v4f32_a + a;
  v4f32_a = v4f32_a + b;
  v4f32_a = v4f32_a + c; // expected-error {{cannot convert between scalar type 'unsigned int' and vector type 'v4f32' (vector of 4 'float' values) as implicit conversion would cause truncation}}
  v4f32_a = v4f32_a + d; // expected-error {{cannot convert between scalar type 'unsigned long long' and vector type 'v4f32' (vector of 4 'float' values) as implicit conversion would cause truncation}}

  v4f64_b = v4f64_b + a;
  v4f64_b = v4f64_b + b;
  v4f64_b = v4f64_b + c;
  v4f64_b = v4f64_b + d; // expected-error {{cannot convert between scalar type 'unsigned long long' and vector type 'v4f64' (vector of 4 'double' values) as implicit conversion would cause truncation}}
}

void floatTestSignedType(char a, short b, int c, long long d) { // expected-warning {{'long long' is incompatible with C++98}}
  v4f32 v4f32_a = {0.4f, 0.4f, 0.4f, 0.4f};
  v4f64 v4f64_b = {0.4, 0.4, 0.4, 0.4};

  v4f32_a = v4f32_a + a;
  v4f32_a = v4f32_a + b;
  v4f32_a = v4f32_a + c; // expected-error {{cannot convert between scalar type 'int' and vector type 'v4f32' (vector of 4 'float' values) as implicit conversion would cause truncation}}
  v4f32_a = v4f32_a + d; // expected-error {{cannot convert between scalar type 'long long' and vector type 'v4f32' (vector of 4 'float' values) as implicit conversion would cause truncation}}

  v4f64_b = v4f64_b + a;
  v4f64_b = v4f64_b + b;
  v4f64_b = v4f64_b + c;
  v4f64_b = v4f64_b + d; // expected-error {{cannot convert between scalar type 'long long' and vector type 'v4f64' (vector of 4 'double' values) as implicit conversion would cause truncation}}
}

void intTestType(char a, short b, int c, long long d) { // expected-warning {{'long long' is incompatible with C++98}}
  v2i64 v2i64_a = {1, 2};
  v2i32 v2i32_a = {1, 2};
  v2i16 v2i16_a = {1, 2};
  v2i8 v2i8_a = {1, 2};

  v2i64_a = v2i64_a + d;
  v2i64_a = v2i64_a + c;
  v2i64_a = v2i64_a + b;
  v2i64_a = v2i64_a + a;

  v2i32_a = v2i32_a + d; // expected-warning {{implicit conversion loses integer precision: 'long long' to 'v2i32' (vector of 2 'int' values)}}
  v2i32_a = v2i32_a + c;
  v2i32_a = v2i32_a + b;
  v2i32_a = v2i32_a + a;

  v2i16_a = v2i16_a + d; // expected-error {{cannot convert between scalar type 'long long' and vector type 'v2i16' (vector of 2 'short' values) as implicit conversion would cause truncation}}
  v2i16_a = v2i16_a + c; // expected-warning {{implicit conversion loses integer precision: 'int' to 'v2i16' (vector of 2 'short' values)}}
  v2i16_a = v2i16_a + b;
  v2i16_a = v2i16_a + a;

  v2i8_a = v2i8_a + d; // expected-error {{cannot convert between scalar type 'long long' and vector type 'v2i8' (vector of 2 'char' values) as implicit conversion would cause truncation}}
  v2i8_a = v2i8_a + c; // expected-error {{cannot convert between scalar type 'int' and vector type 'v2i8' (vector of 2 'char' values) as implicit conversion would cause truncation}}
  v2i8_a = v2i8_a + b; // expected-warning {{implicit conversion loses integer precision: 'short' to 'v2i8' (vector of 2 'char' values)}}
  v2i8_a = v2i8_a + a;
}

void intTestTypeUnsigned(unsigned char a, unsigned short b, unsigned int c,
                         unsigned long long d) { // expected-warning {{'long long' is incompatible with C++98}}
  v2i64 v2i64_a = {1, 2};
  v2i32 v2i32_a = {1, 2};
  v2i16 v2i16_a = {1, 2};
  v2i8 v2i8_a = {1, 2};

  v2i64_a = v2i64_a + d; // expected-error {{cannot convert between scalar type 'unsigned long long' and vector type 'v2i64' (vector of 2 'long long' values) as implicit conversion would cause truncation}}

  v2i64_a = v2i64_a + c;
  v2i64_a = v2i64_a + b;
  v2i64_a = v2i64_a + a;

  v2i32_a = v2i32_a + d; // expected-warning {{implicit conversion loses integer precision: 'unsigned long long' to 'v2i32' (vector of 2 'int' values)}}
  v2i32_a = v2i32_a + c; // expected-error {{cannot convert between scalar type 'unsigned int' and vector type 'v2i32' (vector of 2 'int' values) as implicit conversion would cause truncation}}
  v2i32_a = v2i32_a + b;
  v2i32_a = v2i32_a + a;

  v2i16_a = v2i16_a + d; // expected-error {{cannot convert between scalar type 'unsigned long long' and vector type 'v2i16' (vector of 2 'short' values) as implicit conversion would cause truncation}}
  v2i16_a = v2i16_a + c; // expected-warning {{implicit conversion loses integer precision: 'unsigned int' to 'v2i16' (vector of 2 'short' values)}}
  v2i16_a = v2i16_a + b; // expected-error {{cannot convert between scalar type 'unsigned short' and vector type 'v2i16' (vector of 2 'short' values) as implicit conversion would cause truncation}}
  v2i16_a = v2i16_a + a;

  v2i8_a = v2i8_a + d; // expected-error {{cannot convert between scalar type 'unsigned long long' and vector type 'v2i8' (vector of 2 'char' values) as implicit conversion would cause truncation}}
  v2i8_a = v2i8_a + c; // expected-error {{cannot convert between scalar type 'unsigned int' and vector type 'v2i8' (vector of 2 'char' values) as implicit conversion would cause truncation}}
  v2i8_a = v2i8_a + b; // expected-warning {{implicit conversion loses integer precision: 'unsigned short' to 'v2i8' (vector of 2 'char' values)}}
  v2i8_a = v2i8_a + a; // expected-error {{cannot convert between scalar type 'unsigned char' and vector type 'v2i8' (vector of 2 'char' values) as implicit conversion would cause truncation}}
}

void uintTestType(char a, short b, int c, long long d) { // expected-warning {{'long long' is incompatible with C++98}}
  v2u64 v2u64_a = {1, 2};
  v2u32 v2u32_a = {1, 2};
  v2u16 v2u16_a = {1, 2};
  v2u8 v2u8_a = {1, 2};

  v2u64_a = v2u64_a + d; // expected-warning {{implicit conversion changes signedness: 'long long' to 'v2u64' (vector of 2 'unsigned long long' values)}}
  v2u64_a = v2u64_a + c; // expected-warning {{implicit conversion changes signedness: 'int' to 'v2u64' (vector of 2 'unsigned long long' values)}}
  v2u64_a = v2u64_a + b; // expected-warning {{implicit conversion changes signedness: 'short' to 'v2u64' (vector of 2 'unsigned long long' values)}}
  v2u64_a = v2u64_a + a; // expected-warning {{implicit conversion changes signedness: 'char' to 'v2u64' (vector of 2 'unsigned long long' values)}}

  v2u32_a = v2u32_a + d; // expected-warning {{implicit conversion loses integer precision: 'long long' to 'v2u32' (vector of 2 'unsigned int' values)}}
  v2u32_a = v2u32_a + c; // expected-warning {{implicit conversion changes signedness: 'int' to 'v2u32' (vector of 2 'unsigned int' values)}}
  v2u32_a = v2u32_a + b; // expected-warning {{implicit conversion changes signedness: 'short' to 'v2u32' (vector of 2 'unsigned int' values)}}
  v2u32_a = v2u32_a + a; // expected-warning {{implicit conversion changes signedness: 'char' to 'v2u32' (vector of 2 'unsigned int' values)}}

  v2u16_a = v2u16_a + d; // expected-error {{cannot convert between scalar type 'long long' and vector type 'v2u16' (vector of 2 'unsigned short' values) as implicit conversion would cause truncation}}
  v2u16_a = v2u16_a + c; // expected-warning {{implicit conversion loses integer precision: 'int' to 'v2u16' (vector of 2 'unsigned short' values)}}
  v2u16_a = v2u16_a + b; // expected-warning {{implicit conversion changes signedness: 'short' to 'v2u16' (vector of 2 'unsigned short' values)}}
  v2u16_a = v2u16_a + a; // expected-warning {{implicit conversion changes signedness: 'char' to 'v2u16' (vector of 2 'unsigned short' values)}}

  v2u8_a = v2u8_a + d; // expected-error {{cannot convert between scalar type 'long long' and vector type 'v2u8' (vector of 2 'unsigned char' values) as implicit conversion would cause truncation}}
  v2u8_a = v2u8_a + c; // expected-error {{cannot convert between scalar type 'int' and vector type 'v2u8' (vector of 2 'unsigned char' values) as implicit conversion would cause truncation}}
  v2u8_a = v2u8_a + b; // expected-warning {{implicit conversion loses integer precision: 'short' to 'v2u8' (vector of 2 'unsigned char' values)}}
  v2u8_a = v2u8_a + a; // expected-warning {{implicit conversion changes signedness: 'char' to 'v2u8' (vector of 2 'unsigned char' values)}}
}

void uintTestTypeUnsigned(unsigned char a, unsigned short b, unsigned int c,
                          unsigned long long d) { // expected-warning {{'long long' is incompatible with C++98}}
  v2u64 v2u64_a = {1, 2};
  v2u32 v2u32_a = {1, 2};
  v2u16 v2u16_a = {1, 2};
  v2u8 v2u8_a = {1, 2};

  v2u64_a = v2u64_a + d;
  v2u64_a = v2u64_a + c;
  v2u64_a = v2u64_a + b;
  v2u64_a = v2u64_a + a;

  v2u32_a = v2u32_a + d; // expected-warning {{implicit conversion loses integer precision: 'unsigned long long' to 'v2u32' (vector of 2 'unsigned int' values)}}
  v2u32_a = v2u32_a + c;
  v2u32_a = v2u32_a + b;
  v2u32_a = v2u32_a + a;

  v2u16_a = v2u16_a + d; // expected-error {{cannot convert between scalar type 'unsigned long long' and vector type 'v2u16' (vector of 2 'unsigned short' values) as implicit conversion would cause truncation}}
  v2u16_a = v2u16_a + c; // expected-warning {{implicit conversion loses integer precision: 'unsigned int' to 'v2u16' (vector of 2 'unsigned short' values)}}
  v2u16_a = v2u16_a + b;
  v2u16_a = v2u16_a + a;

  v2u8_a = v2u8_a + d; // expected-error {{cannot convert between scalar type 'unsigned long long' and vector type 'v2u8' (vector of 2 'unsigned char' values) as implicit conversion would cause truncation}}
  v2u8_a = v2u8_a + c; // expected-error {{cannot convert between scalar type 'unsigned int' and vector type 'v2u8' (vector of 2 'unsigned char' values) as implicit conversion would cause truncation}}
  v2u8_a = v2u8_a + b; // expected-warning {{implicit conversion loses integer precision: 'unsigned short' to 'v2u8' (vector of 2 'unsigned char' values)}}
  v2u8_a = v2u8_a + a;
}

void uintTestConstant(v2u64 v2u64_a, v2u32 v2u32_a, v2u16 v2u16_a,
                      v2u8 v2u8_a) {
  v2u64_a = v2u64_a + 0xFFFFFFFFFFFFFFFF;
  v2u32_a = v2u32_a + 0xFFFFFFFF;
  v2u16_a = v2u16_a + 0xFFFF;
  v2u8_a = v2u8_a + 0xFF;

  v2u32_a = v2u32_a + 0x1FFFFFFFF; // expected-warning {{implicit conversion from 'long' to 'v2u32' (vector of 2 'unsigned int' values) changes value from 8589934591 to 4294967295}}
  v2u16_a = v2u16_a + 0x1FFFF;     // expected-warning {{implicit conversion from 'int' to 'v2u16' (vector of 2 'unsigned short' values) changes value from 131071 to 65535}}
  v2u8_a = v2u8_a + 0x1FF;         // expected-error {{cannot convert between scalar type 'int' and vector type 'v2u8' (vector of 2 'unsigned char' values) as implicit conversion would cause truncation}}
}

void intTestConstant(v2i64 v2i64_a, v2i32 v2i32_a, v2i16 v2i16_a, v2i8 v2i8_a) {
  // Legal upper bounds.
  v2i64_a = v2i64_a + static_cast<long long>(0x7FFFFFFFFFFFFFFF); // expected-warning {{'long long' is incompatible with C++98}}
  v2i32_a = v2i32_a + static_cast<int>(0x7FFFFFFF);
  v2i16_a = v2i16_a + static_cast<short>(0x7FFF);
  v2i8_a = v2i8_a + static_cast<char>(0x7F);

  // Legal lower bounds.
  v2i64_a = v2i64_a + (-9223372036854775807);
  v2i32_a = v2i32_a + (-2147483648);
  v2i16_a = v2i16_a + (-32768);
  v2i8_a = v2i8_a + (-128);

  // One increment/decrement more than the type can hold
  v2i32_a = v2i32_a + 2147483648; // expected-warning {{implicit conversion from 'long' to 'v2i32' (vector of 2 'int' values) changes value from 2147483648 to -2147483648}}
  v2i16_a = v2i16_a + 32768;      // expected-warning {{implicit conversion from 'int' to 'v2i16' (vector of 2 'short' values) changes value from 32768 to -32768}}
  v2i8_a = v2i8_a + 128;          // expected-warning {{implicit conversion from 'int' to 'v2i8' (vector of 2 'char' values) changes value from 128 to -128}}

  v2i32_a = v2i32_a + (-2147483649); // expected-warning {{implicit conversion from 'long' to 'v2i32' (vector of 2 'int' values) changes value from -2147483649 to 2147483647}}
  v2i16_a = v2i16_a + (-32769);      // expected-warning {{implicit conversion from 'int' to 'v2i16' (vector of 2 'short' values) changes value from -32769 to 32767}}
  v2i8_a = v2i8_a + (-129);          // expected-error {{cannot convert between scalar type 'int' and vector type 'v2i8' (vector of 2 'char' values) as implicit conversion would cause truncation}}
}
