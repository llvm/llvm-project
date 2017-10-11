// RUN: %clang_cc1 -fmodules -verify -fno-modules-error-recovery -fno-spell-checking %s
// RUN: %clang_cc1 -fmodules -verify -fno-modules-error-recovery -DONLY_Y %s

#pragma clang module build a
module a {
  explicit module x {}
  explicit module y {}
}
#pragma clang module contents
#pragma clang module begin a.x
namespace N {
  template<typename T> extern int f(T) { return 0; }
}
#pragma clang module end

#pragma clang module begin a.y
#pragma clang module import a.x
using N::f;
#pragma clang module end
#pragma clang module endbuild

namespace N { struct A {}; }
struct B {};

#ifndef ONLY_Y
#pragma clang module import a.x
void test1() {
  f(N::A());
  f(B()); // expected-error {{use of undeclared identifier 'f'}}
}
#else
// expected-no-diagnostics
#endif

#pragma clang module import a.y
void test2() {
  // These are OK even if a.x is not imported.
  f(N::A());
  f(B());
}
