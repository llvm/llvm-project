// RUN:     %clang_cc1 -std=c++20 -emit-module-interface %s -o %t.pcm -verify
// RUN:     %clang_cc1 -std=c++20 -emit-module-interface %s -o %t.pcm -verify -DERRORS

export module foo;
#ifndef ERRORS
// expected-no-diagnostics
#else
// FIXME: diagnose missing module-declaration when building module interface

// FIXME: proclaimed-ownership-declarations?

export {
  int a;
  int b;
}
export int c;

namespace N {
export void f() {}
} // namespace N

export struct T {
} t;

struct S {
  export int n;        // expected-error {{expected member name or ';'}}
  export static int n; // expected-error {{expected member name or ';'}}
};
void f() {
  export int n; // expected-error {{expected expression}}
}
#endif
