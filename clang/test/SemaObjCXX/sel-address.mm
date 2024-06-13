// RUN: %clang_cc1 -fsyntax-only -verify %s
// pr7390

void f(const SEL& v2) {}
void g(SEL* _Nonnull);
void h() {
  f(@selector(dealloc));

  SEL s = @selector(dealloc);
  SEL* ps = &s;

  /*
   FIXME: https://github.com/llvm/llvm-project/pull/94159
   
   TLDR; This is about inserting '*' to deref.
   
   This would assign the value of s to the SEL object pointed to by
   @selector(dealloc). However, in Objective-C, selectors are not pointers,
   they are special compile-time constructs representing method names, and
   they are immutable, so you cannot assign values to them.

   Therefore, this syntax is not valid for selectors in Objective-C.
   */
  @selector(dealloc) = s;  // expected-error {{expression is not assignable}}
  // expected-note@-1 {{add '*' to dereference it}}

  SEL* ps2 = &@selector(dealloc);

  // Shouldn't crash.
  g(&@selector(foo));
  g(&(@selector(foo)));
}

