// RUN: %clang_cc1 -fsyntax-only -verify -triple i686-linux-pc %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple i686-linux-pc -target-feature +sse2 %s -DHAVE
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-linux-pc %s -DHAVE
// RUN: %clang_cc1 -fsyntax-only -verify -triple spir-unknown-unknown %s -DHAVE
// RUN: %clang_cc1 -fsyntax-only -verify -triple armv7a-linux-gnu %s -DHAVE
// RUN: %clang_cc1 -fsyntax-only -verify -triple aarch64-linux-gnu %s -DHAVE

#ifndef HAVE
// expected-error@+2{{_Float16 is not supported on this target}}
#endif // HAVE
_Float16 f;

#ifdef HAVE
_Complex _Float16 a;
void builtin_complex(void) {
  _Float16 a = 0;
  (void)__builtin_complex(a, a); // expected-error {{'_Complex _Float16' is invalid}}
}
#endif
