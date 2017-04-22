// RUN:     %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %s -o %t.pcm -verify
// RUN:     %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %s -o %t.pcm -verify -Dmodule=int -DERRORS

export module foo;
#ifndef ERRORS
// expected-no-diagnostics
#else
// FIXME: diagnose 'export' declaration in non-module
// FIXME: diagnose missing module-declaration when building module interface

// FIXME: proclaimed-ownership-declarations?

export {
  int a;
  int b;
}
export int c;

namespace N {
  export void f() {}
}

export struct T {} t;

struct S {
  export int n; // expected-error {{expected member name or ';'}}
  export static int n; // expected-error {{expected member name or ';'}}
};
void f() {
  export int n; // expected-error {{expected expression}}
}
#endif
