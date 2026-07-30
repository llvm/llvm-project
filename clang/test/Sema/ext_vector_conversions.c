// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -fsyntax-only -verify -Wconversion %s

typedef __attribute__((ext_vector_type(4))) char char4;
typedef __attribute__((ext_vector_type(4))) short short4;
typedef __attribute__((ext_vector_type(1))) float float1;

static void test(void) {
  char4 vc4;
  float f;
  // Not allowed.  There's no splatting conversion between float and int vector,
  // and we don't want to bitcast f to vector-of-char (as would happen with the
  // old-style vector types).
  vc4 += f; // expected-error {{cannot convert between vector values of different size}}
  short4 vs4;
  long long ll;
  // This one is OK; we don't re-interpret ll as short4, rather we splat its
  // value, which should produce a warning about clamping.
  vs4 += ll; // expected-warning {{implicit conversion loses integer precision}}
}
