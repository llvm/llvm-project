// RUN: %clang_cc1 -fsyntax-only -verify %s
// pr7390

void f(const SEL& v2) {}
void g(SEL* _Nonnull);
void h() {
  f(@selector(dealloc));

  SEL s = @selector(dealloc);
  SEL* ps = &s;

  @selector(dealloc) = s;  // expected-error {{expression is not assignable}}

  SEL* ps2 = &@selector(dealloc);

  // Shouldn't crash.
  g(&@selector(foo));
  g(&(@selector(foo)));
}

