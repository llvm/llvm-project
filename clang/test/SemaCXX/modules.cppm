// RUN:     %clang_cc1 -std=c++20 -emit-module-interface %s -o %t.0.pcm -verify -DTEST=0
// RUN:     %clang_cc1 -std=c++20 -emit-module-interface %s -o %t.1.pcm -verify -DTEST=1
// RUN:     %clang_cc1 -std=c++20 -emit-module-interface %s -fmodule-file=foo=%t.0.pcm -o %t.2.pcm -verify -DTEST=2
// RUN:     %clang_cc1 -std=c++20 -emit-module-interface %s -fmodule-file=foo=%t.0.pcm -o %t.3.pcm -verify -Dfoo=bar -DTEST=3

#if TEST == 0 || TEST == 2
// expected-no-diagnostics
#endif

export module foo;

static int m;

int n;

#if TEST == 0
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
#elif TEST == 3
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
#endif

#if TEST == 1
struct S {
  export int n;        // expected-error {{expected member name or ';'}}
  export static int n; // expected-error {{expected member name or ';'}}
};
#endif

// FIXME: Exports of declarations without external linkage are disallowed.
// Exports of declarations with non-external-linkage types are disallowed.

// Cannot export within another export. This isn't precisely covered by the
// language rules right now, but (per personal correspondence between zygoloid
// and gdr) is the intent.
#if TEST == 1
export { // expected-note {{export block begins here}}
  extern "C++" {
  namespace NestedExport {
  export { // expected-error {{export declaration appears within another export declaration}}
    int q;
  }
  } // namespace NestedExport
  }
}
#endif
