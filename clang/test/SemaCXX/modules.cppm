// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/A.cppm -o %t.0.pcm -verify
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/B.cppm -o %t.1.pcm -verify
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/C.cppm -fmodule-file=foo=%t.0.pcm -o %t.2.pcm -verify
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/D.cppm -fmodule-file=foo=%t.0.pcm -o %t.3.pcm -verify
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/E.cppm -fmodule-file=foo=%t.0.pcm -o %t.3.pcm -verify -Dfoo=bar

//--- A.cppm
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
// expected-no-diagnostics

//--- B.cppm
export module foo;
static int m;
int n;
struct S {
  export int n;        // expected-error {{expected member name or ';'}}
  export static int n; // expected-error {{expected member name or ';'}}
};

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

//--- C.cppm
export module foo;
static int m;
int n;
// expected-no-diagnostics

//--- D.cppm
export module foo;
static int m;
int n;
int use_a = a; // expected-error {{use of undeclared identifier 'a'}}

#undef foo
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

//--- E.cppm
export module foo; // expected-error {{the module name in a module declaration cannot contain an object-like macro 'foo'}}
static int m;
int n;
int use_a = a; // expected-error {{use of undeclared identifier 'a'}}

#undef foo
import foo; // expected-error {{imports must immediately follow the module declaration}}
