// RUN: %clang_cc1 %s -fsyntax-only -verify
// expected-no-diagnostics

// This example uncovered a bug in Sema::BuiltinVectorMath, where we should be
// using ASTContext::hasSameUnqualifiedType().

typedef float vec3 __attribute__((ext_vector_type(3)));

typedef struct {
  vec3 b;
} struc;

vec3 foo(vec3 a,const struc* hi) {
  vec3 b = __builtin_elementwise_max((vec3)(0.0f), a);
  return __builtin_elementwise_pow(b, hi->b.yyy);
}
