// RUN: %clang_analyze_cc1 -w -verify %s\
// RUN:   -analyzer-checker=core\
// RUN:   -analyzer-checker=debug.ExprInspection -std=c++17
// RUN: %clang_analyze_cc1 -w -verify %s\
// RUN:   -analyzer-checker=core\
// RUN:   -analyzer-checker=debug.ExprInspection -std=c++11\
// RUN:   -DTEST_INLINABLE_ALLOCATORS
// RUN: %clang_analyze_cc1 -w -verify %s\
// RUN:   -analyzer-checker=core\
// RUN:   -analyzer-checker=debug.ExprInspection -std=c++17\
// RUN:   -DTEST_INLINABLE_ALLOCATORS

void clang_analyzer_eval(bool);

using size_t = decltype(sizeof(int));

namespace CustomClassType {
struct S {
  int x;
  static void* operator new(size_t size) {
    return ::operator new(size);
  }
};
void F() {
  S *s = new S;
  clang_analyzer_eval(s->x); // expected-warning{{UNKNOWN}} FIXME: should be an undefined warning

  S *s2 = new S{};
  clang_analyzer_eval(0 == s2->x); // expected-warning{{TRUE}}

  S *s3 = new S{1};
  clang_analyzer_eval(1 == s3->x); // expected-warning{{TRUE}}
}
} // namespace CustomClassType
