// RUN: %clang_cc1 -triple=powerpc64-unknown-linux-gnu -target-feature +altivec -target-feature +vsx -target-cpu pwr8 -fsyntax-only -verify=expected,nonaix %s
// RUN: %clang_cc1 -triple=powerpc64le-unknown-linux-gnu -target-feature +altivec -target-feature +vsx -target-cpu pwr8 -fsyntax-only -verify=expected,novsx %s
// RUN: %clang_cc1 -triple=powerpc64-ibm-aix -target-feature +altivec -target-feature +vsx -target-cpu pwr8 -fsyntax-only -verify=expected,aix %s

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
