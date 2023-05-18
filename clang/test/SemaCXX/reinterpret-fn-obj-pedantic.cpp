// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 -pedantic %s

void fnptrs()
{
  typedef void (*fnptr)();
  fnptr fp = 0;
  void *vp = reinterpret_cast<void*>(fp); // expected-warning {{cast between pointer-to-function and pointer-to-object is an extension}}
  (void)reinterpret_cast<fnptr>(vp); // expected-warning {{cast between pointer-to-function and pointer-to-object is an extension}}
}
