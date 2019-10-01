// RUN:     %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %s -o %t.0.pcm -verify -DTEST=0
// RUN:     %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %s -o %t.1.pcm -verify -DTEST=1
// RUN:     %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %s -fmodule-file=%t.0.pcm -o %t.2.pcm -verify -DTEST=2
// RUN:     %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %s -fmodule-file=%t.0.pcm -o %t.3.pcm -verify -Dfoo=bar -DTEST=3

#if TEST == 0
// expected-no-diagnostics
#endif

export module foo;
#if TEST == 2
// expected-error@-2 {{redefinition of module 'foo'}}
// expected-note@modules-ts.cppm:* {{loaded from}}
#endif

static int m;
#if TEST == 2
// expected-error@-2 {{redefinition of '}}
// expected-note@-3 {{unguarded header; consider using #ifdef guards or #pragma once}}
// FIXME: We should drop the "header from" in this diagnostic.
// expected-note-re@modules-ts.cppm:1 {{'{{.*}}modules-ts.cppm' included multiple times, additional include site in header from module 'foo'}}
#endif
int n;
#if TEST == 2
// expected-error@-2 {{redefinition of '}}
// expected-note@-3 {{unguarded header; consider using #ifdef guards or #pragma once}}
// FIXME: We should drop the "header from" in this diagnostic.
// expected-note-re@modules-ts.cppm:1 {{'{{.*}}modules-ts.cppm' included multiple times, additional include site in header from module 'foo'}}
#endif

#if TEST == 0
export {
  int a;
  int b;
  constexpr int *p = &n;
}
export int c;

namespace N {
  export void f() {}
}

export struct T {} t;
#elif TEST == 3
int use_a = a; // expected-error {{declaration of 'a' must be imported from module 'foo' before it is required}}
// expected-note@-13 {{previous}}

#undef foo
import foo;

export {} // expected-error {{export declaration cannot be empty}}
export { // expected-note {{begins here}}
  ; // expected-warning {{ISO C++20 does not permit an empty declaration to appear in an export block}}
}
export { // expected-note {{begins here}}
  static_assert(true); // expected-warning {{ISO C++20 does not permit a static_assert declaration to appear in an export block}}
}

int use_b = b;
int use_n = n; // FIXME: this should not be visible, because it is not exported

extern int n;
static_assert(&n != p);
#endif


#if TEST == 1
struct S {
  export int n; // expected-error {{expected member name or ';'}}
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
      export { // expected-error {{appears within another export}}
        int q;
      }
    }
  }
}
#endif
