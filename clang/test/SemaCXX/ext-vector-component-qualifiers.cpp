// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

typedef float float2 __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(3))) float2 LocalFloat2;

// A component gets the qualifiers of the vector, also via '->'. A swizzle that
// repeats a component is a computed value, so it doesn't get any qualifiers.
void f(const float2 *p, LocalFloat2 *q, const float2 v) {
  static_assert(__is_same(decltype((p->x)), const float &), "");
  static_assert(__is_same(decltype((v.x)), const float &), "");
  static_assert(__is_same(decltype((q->x)), __attribute__((address_space(3))) float &), "");
  static_assert(__is_same(decltype(p->xx), float2), "");
  static_assert(__is_same(decltype(v.xx), float2), "");
  static_assert(__is_same(decltype(q->xx), float2), "");
  const float2 &r = q->xx;
  (void)r;
}
