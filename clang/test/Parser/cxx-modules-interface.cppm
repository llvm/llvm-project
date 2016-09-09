// RUN:     %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %s -o %t.pcm -verify
// RUN:     %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %s -o %t.pcm -verify -Dmodule=int -DERRORS

module foo;
#ifndef ERRORS
// expected-no-diagnostics
#else
// expected-error@-4 {{expected module declaration at start of module interface}}

// FIXME: support 'export module X;' and 'export { int n; module X; }'
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
