// RUN: %clang_analyze_cc1 -verify %s\
// RUN:   -analyzer-checker=core,debug.ExprInspection

void clang_analyzer_eval(bool);

using size_t = decltype(sizeof(int));

template <class FirstT, class... Rest>
void escape(FirstT first, Rest... args);

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

  escape(s, s2, s3);
}
} // namespace CustomClassType
