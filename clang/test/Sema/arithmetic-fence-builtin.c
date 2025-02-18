// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm -o - -verify -x c++ %s
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm -o - -verify -x c++ %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -triple ppc64le -DPPC     -emit-llvm -o - -verify -x c++ %s
// RUN: %clang_cc1 -triple ppc64le -DPPC     -emit-llvm -o - -verify -x c++ %s -fexperimental-new-constant-interpreter
// RUN: not %clang_cc1 -triple ppc64le -DPPC     -emit-llvm -o - -x c++ %s \
// RUN:            -fprotect-parens 2>&1 | FileCheck -check-prefix=PPC %s
// RUN: not %clang_cc1 -triple ppc64le -DPPC     -emit-llvm -o - -x c++ %s -fexperimental-new-constant-interpreter \
// RUN:            -fprotect-parens 2>&1 | FileCheck -check-prefix=PPC %s
// RUN: %clang_cc1 -triple spir64 -emit-llvm -o - -verify -x c++ %s
// RUN: %clang_cc1 -triple spir64 -emit-llvm -o - -verify -x c++ %s -fexperimental-new-constant-interpreter
#ifndef PPC
int v;
template <typename T> T addT(T a, T b) {
  T *q = __arithmetic_fence(&a);
  // expected-error@-1 {{invalid operand of type 'float *' where floating, complex or a vector of such types is required}}
  // expected-error@-2 {{invalid operand of type 'int *' where floating, complex or a vector of such types is required}}
  return __arithmetic_fence(a + b);
  // expected-error@-1 {{invalid operand of type 'int' where floating, complex or a vector of such types is required}}
}
int addit(int a, int b) {
  float x, y;
  typedef struct {
    int a, b;
  } stype;
  stype s;
  s = __arithmetic_fence(s);    // expected-error {{invalid operand of type 'stype' where floating, complex or a vector of such types is required}}
  x = __arithmetic_fence();     // expected-error {{too few arguments to function call, expected 1, have 0}}
  x = __arithmetic_fence(x, y); // expected-error {{too many arguments to function call, expected 1, have 2}}
  // Complex is supported.
  _Complex double cd, cd1;
  cd = __arithmetic_fence(cd1);
  // Vector is supported.
  typedef float __v4hi __attribute__((__vector_size__(8)));
  __v4hi vec1, vec2;
  vec1 = __arithmetic_fence(vec2);

  v = __arithmetic_fence(a + b); // expected-error {{invalid operand of type 'int' where floating, complex or a vector of such types is required}}
  float f = addT<float>(a, b);   // expected-note {{in instantiation of function template specialization 'addT<float>' requested here}}
  int i = addT<int>(1, 2);       // expected-note {{in instantiation of function template specialization 'addT<int>' requested here}}
  constexpr float d = 1.0 + 2.0;
  constexpr float c = __arithmetic_fence(1.0 + 2.0);
  constexpr float e = __arithmetic_fence(d);
  return 0;
}
bool func(float f1, float f2, float f3) {
  return (f1 == f2 && f1 == f3) || f2 == f3; // Should not warn here
}
static_assert( __arithmetic_fence(1.0 + 2.0), "message" );
#else
float addit(float a, float b) {
  return __arithmetic_fence(a+b); // expected-error {{builtin is not supported on this target}}
}
#endif
//PPC: error: option '-fprotect-parens' cannot be specified on this target
