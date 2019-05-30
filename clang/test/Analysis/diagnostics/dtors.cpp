// RUN: %clang_analyze_cc1 -w -analyzer-checker=core,cplusplus -analyzer-output=text -verify %s

namespace no_crash_on_delete_dtor {
// We were crashing when producing diagnostics for this code, but not for the
// report that it currently emits. Instead, Static Analyzer was thinking that
// p.get()->foo() is a null dereference because it was dropping
// constraints over x too early and took a different branch next time
// we call .get().
struct S {
  void foo();
  ~S();
};

struct smart_ptr {
  int x;
  S *s;
  smart_ptr(S *);
  S *get() {
    return (x || 0) ? nullptr : s; // expected-note{{Field 'x' is 0}}
                                   // expected-note@-1{{Left side of '||' is false}}
                                   // expected-note@-2{{'?' condition is false}}
                                   // expected-warning@-3{{Use of memory after it is freed}}
                                   // expected-note@-4{{Use of memory after it is freed}}
  }
};

void bar(smart_ptr p) {
  delete p.get(); // expected-note{{Memory is released}}
  p.get()->foo(); // expected-note{{Calling 'smart_ptr::get'}}
}
} // namespace no_crash_on_delete_dtor
