// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN:     %clang_cc1 -std=c++20 -emit-module-interface %t/test0.cpp -o %t/test0.pcm -verify
// RUN:     %clang_cc1 -std=c++20 -emit-module-interface %t/test1.cpp -o %t/test1.pcm -verify
// RUN:     %clang_cc1 -std=c++20 -emit-module-interface %t/test2.cpp -fmodule-file=foo=%t/test0.pcm -o %t/test2.pcm -verify
// RUN:     %clang_cc1 -std=c++20 -emit-module-interface %t/test3.cpp -fmodule-file=foo=%t/test0.pcm -o %t/test3.pcm -verify

//--- test0.cpp
// expected-no-diagnostics
export module foo;

static int m;

int n;

export {
  int a;
  int b;
  constexpr int *p = &n;
}
export int c;

namespace N {
export void f() {}
} // namespace N

export struct T {
} t;

//--- test1.cpp
export module foo;

static int m;

int n;

struct S {
  export int n;        // expected-error {{expected member name or ';'}}
  export static int n; // expected-error {{expected member name or ';'}}
};

int main() {} // expected-warning {{'main' never has module linkage}}

// FIXME: Exports of declarations without external linkage are disallowed.
// Exports of declarations with non-external-linkage types are disallowed.

// Cannot export within another export. This isn't precisely covered by the
// language rules right now, but (per personal correspondence between zygoloid
// and gdr) is the intent.
export { // expected-note {{export block begins here}}
  extern "C++" {
  namespace NestedExport {
  export { // expected-error {{export declaration appears within another export declaration}}
    int q;
  }
  } // namespace NestedExport
  }
}

//--- test2.cpp
// expected-no-diagnostics
export module foo;

static int m;

int n;

//--- test3.cpp
export module bar;

extern "C++" int main() {}

static int m;

int n;

int use_a = a; // expected-error {{use of undeclared identifier 'a'}}

import foo; // expected-error {{imports must immediately follow the module declaration}}

export {}
export {
  ;       // No diagnostic after P2615R1 DR
}
export {
  static_assert(true); // No diagnostic after P2615R1 DR
}

int use_b = b; // expected-error{{use of undeclared identifier 'b'}}
int use_n = n; // FIXME: this should not be visible, because it is not exported

extern int n;
static_assert(&n != p); // expected-error{{use of undeclared identifier 'p'}}
