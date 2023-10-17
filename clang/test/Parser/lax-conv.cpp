// RUN: %clang_cc1 -triple=powerpc64-unknown-linux-gnu -target-feature +altivec -target-feature +vsx -target-cpu pwr8 -fsyntax-only -verify=expected,nonaix %s
// RUN: %clang_cc1 -triple=powerpc64le-unknown-linux-gnu -target-feature +altivec -target-feature +vsx -target-cpu pwr8 -fsyntax-only -verify=expected,novsx %s
// RUN: %clang_cc1 -triple=powerpc64-ibm-aix -target-feature +altivec -target-feature +vsx -target-cpu pwr8 -fsyntax-only -verify=expected,aix %s

vector bool short vbs;
vector signed short vss;
vector unsigned short vus;
vector bool int vbi;
vector signed int vsi;
vector unsigned int vui;
vector bool long long vbl;
vector signed long long vsl;
vector unsigned long long vul;
vector bool char vbc;
vector signed char vsc;
vector unsigned char vuc;
vector pixel vp;

void dummy(vector unsigned int a);
template <typename VEC> VEC __attribute__((noinline)) test(vector unsigned char a, vector unsigned char b) {
    return (VEC)(a * b);
}
vector unsigned int test1(vector unsigned char RetImplicitConv) {
  return RetImplicitConv; // expected-warning {{Implicit conversion between vector types (''__vector unsigned char' (vector of 16 'unsigned char' values)' and ''__vector unsigned int' (vector of 4 'unsigned int' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}} 
}
vector unsigned int test2(vector unsigned char RetImplicitConvAddConst) {
  return RetImplicitConvAddConst + 5; // expected-warning {{Implicit conversion between vector types (''__vector unsigned char' (vector of 16 'unsigned char' values)' and ''__vector unsigned int' (vector of 4 'unsigned int' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}} 
}
vector unsigned int test3(vector unsigned char RetExplicitConv) {
  return (vector unsigned int)RetExplicitConv;
}
vector unsigned int test4(vector unsigned char RetExplicitConvAddConst) {
  return (vector unsigned int)RetExplicitConvAddConst + 5;
}
vector unsigned int test5(vector unsigned char RetImplicitConvAddSame1,
                          vector unsigned char RetImplicitConvAddSame2) {
  return RetImplicitConvAddSame1 + RetImplicitConvAddSame2; // expected-warning {{Implicit conversion between vector types (''__vector unsigned char' (vector of 16 'unsigned char' values)' and ''__vector unsigned int' (vector of 4 'unsigned int' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}} 
}
vector unsigned int test6(vector unsigned char RetExplicitConvAddSame1,
                          vector unsigned char RetExplicitConvAddSame2) {
  return (vector unsigned int)RetExplicitConvAddSame1 +
         (vector unsigned int)RetExplicitConvAddSame2;
}
vector unsigned int test7(vector unsigned char RetExplicitConvAddSame1Full,
                          vector unsigned char RetExplicitConvAddSame2Full) {
  return (vector unsigned int)(RetExplicitConvAddSame1Full +
                               RetExplicitConvAddSame2Full);
}
vector unsigned char test8(vector unsigned char a, vector unsigned char b) {
    return test<vector unsigned char>(a, b);
}

vector unsigned long long test9(vector unsigned char a, vector unsigned char b) {
    return test<vector unsigned long long>(a, b);
}
void test1a(vector unsigned char ArgImplicitConv) {
  return dummy(ArgImplicitConv); // expected-warning {{Implicit conversion between vector types (''__vector unsigned char' (vector of 16 'unsigned char' values)' and ''__vector unsigned int' (vector of 4 'unsigned int' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}
}
void test2a(vector unsigned char ArgImplicitConvAddConst) {
  return dummy(ArgImplicitConvAddConst + 5); // expected-warning {{Implicit conversion between vector types (''__vector unsigned char' (vector of 16 'unsigned char' values)' and ''__vector unsigned int' (vector of 4 'unsigned int' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}
}
void test3a(vector unsigned char ArgExplicitConv) {
  return dummy((vector unsigned int)ArgExplicitConv);
}
void test4a(vector unsigned char ArgExplicitConvAddConst) {
  return dummy((vector unsigned int)ArgExplicitConvAddConst + 5);
}
void test5a(vector unsigned char ArgImplicitConvAddSame1,
            vector unsigned char ArgImplicitConvAddSame2) {
  return dummy(ArgImplicitConvAddSame1 + ArgImplicitConvAddSame2); // expected-warning {{Implicit conversion between vector types (''__vector unsigned char' (vector of 16 'unsigned char' values)' and ''__vector unsigned int' (vector of 4 'unsigned int' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}
}
void test6a(vector unsigned char ArgExplicitConvAddSame1,
            vector unsigned char ArgExplicitConvAddSame2) {
  return dummy((vector unsigned int)ArgExplicitConvAddSame1 +
               (vector unsigned int)ArgExplicitConvAddSame2);
}
void test7a(vector unsigned char ArgExplicitConvAddSame1Full,
            vector unsigned char ArgExplicitConvAddSame2Full) {
  return dummy((vector unsigned int)(ArgExplicitConvAddSame1Full +
                                     ArgExplicitConvAddSame2Full));
}
void test_bool_compat(void) {
  vbs = vss; // expected-warning {{Implicit conversion between vector types (''__vector short' (vector of 8 'short' values)' and ''__vector __bool unsigned short' (vector of 8 'unsigned short' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}
  vbs = vus; // expected-warning {{Implicit conversion between vector types (''__vector unsigned short' (vector of 8 'unsigned short' values)' and ''__vector __bool unsigned short' (vector of 8 'unsigned short' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}

  vbi = vsi; // expected-warning {{Implicit conversion between vector types (''__vector int' (vector of 4 'int' values)' and ''__vector __bool unsigned int' (vector of 4 'unsigned int' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}
  vbi = vui; // expected-warning {{Implicit conversion between vector types (''__vector unsigned int' (vector of 4 'unsigned int' values)' and ''__vector __bool unsigned int' (vector of 4 'unsigned int' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}

  vbl = vsl; // expected-warning {{Implicit conversion between vector types (''__vector long long' (vector of 2 'long long' values)' and ''__vector __bool unsigned long long' (vector of 2 'unsigned long long' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}
  vbl = vul; // expected-warning {{Implicit conversion between vector types (''__vector unsigned long long' (vector of 2 'unsigned long long' values)' and ''__vector __bool unsigned long long' (vector of 2 'unsigned long long' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}

  vbc = vsc; // expected-warning {{Implicit conversion between vector types (''__vector signed char' (vector of 16 'signed char' values)' and ''__vector __bool unsigned char' (vector of 16 'unsigned char' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}
  vbc = vuc; // expected-warning {{Implicit conversion between vector types (''__vector unsigned char' (vector of 16 'unsigned char' values)' and ''__vector __bool unsigned char' (vector of 16 'unsigned char' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}
}

void test_pixel_compat(void) {
  vp = vbs; // expected-warning {{Implicit conversion between vector types (''__vector __bool unsigned short' (vector of 8 'unsigned short' values)' and ''__vector __pixel ' (vector of 8 'unsigned short' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}
  vp = vss; // expected-warning {{Implicit conversion between vector types (''__vector short' (vector of 8 'short' values)' and ''__vector __pixel ' (vector of 8 'unsigned short' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}
  vp = vus; // expected-warning {{Implicit conversion between vector types (''__vector unsigned short' (vector of 8 'unsigned short' values)' and ''__vector __pixel ' (vector of 8 'unsigned short' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}

  vp = vbi; // expected-warning {{Implicit conversion between vector types (''__vector __bool unsigned int' (vector of 4 'unsigned int' values)' and ''__vector __pixel ' (vector of 8 'unsigned short' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}
  vp = vsi; // expected-warning {{Implicit conversion between vector types (''__vector int' (vector of 4 'int' values)' and ''__vector __pixel ' (vector of 8 'unsigned short' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}
  vp = vui; // expected-warning {{Implicit conversion between vector types (''__vector unsigned int' (vector of 4 'unsigned int' values)' and ''__vector __pixel ' (vector of 8 'unsigned short' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}

  vp = vbl; // expected-warning {{Implicit conversion between vector types (''__vector __bool unsigned long long' (vector of 2 'unsigned long long' values)' and ''__vector __pixel ' (vector of 8 'unsigned short' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}
  vp = vsl; // expected-warning {{Implicit conversion between vector types (''__vector long long' (vector of 2 'long long' values)' and ''__vector __pixel ' (vector of 8 'unsigned short' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}
  vp = vul; // expected-warning {{Implicit conversion between vector types (''__vector unsigned long long' (vector of 2 'unsigned long long' values)' and ''__vector __pixel ' (vector of 8 'unsigned short' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}

  vp = vbc; // expected-warning {{Implicit conversion between vector types (''__vector __bool unsigned char' (vector of 16 'unsigned char' values)' and ''__vector __pixel ' (vector of 8 'unsigned short' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}
  vp = vsc; // expected-warning {{Implicit conversion between vector types (''__vector signed char' (vector of 16 'signed char' values)' and ''__vector __pixel ' (vector of 8 'unsigned short' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}
  vp = vuc; // expected-warning {{Implicit conversion between vector types (''__vector unsigned char' (vector of 16 'unsigned char' values)' and ''__vector __pixel ' (vector of 8 'unsigned short' values)') is deprecated. In the future, the behavior implied by '-fno-lax-vector-conversions' will be the default.}}
}
