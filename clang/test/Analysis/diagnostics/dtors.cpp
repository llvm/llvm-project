// RUN: %clang_analyze_cc1 -w -analyzer-checker=core,cplusplus -verify %s

// expected-no-diagnostics

namespace no_crash_on_delete_dtor {
// We were crashing when producing diagnostics for this code.
struct S {
  void foo();
  ~S();
};

struct smart_ptr {
  int x;
  S *s;
  smart_ptr(S *);
  S *get() {
    return (x || 0) ? nullptr : s;
  }
};

void bar(smart_ptr p) {
  delete p.get();
  p.get()->foo();
}
} // namespace no_crash_on_delete_dtor
