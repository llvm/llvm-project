// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++11 -fexceptions -fcxx-exceptions -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++11 -verify %s

void clang_analyzer_eval(bool);

typedef __typeof__(sizeof(int)) size_t;
extern "C" void *malloc(size_t);

// This is the standard placement new.
inline void* operator new(size_t, void* __p) throw()
{
  return __p;
}

struct NoThrow {
  void *operator new(size_t) throw();
};

struct NoExcept {
  void *operator new(size_t) noexcept;
};

struct DefaultThrow {
  void *operator new(size_t);
};

struct ExplicitThrow {
  void *operator new(size_t) throw(int);
};

void testNew() {
  clang_analyzer_eval(new NoThrow); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(new NoExcept); // expected-warning{{UNKNOWN}}

  clang_analyzer_eval(new DefaultThrow); // expected-warning{{TRUE}}
  clang_analyzer_eval(new ExplicitThrow); // expected-warning{{TRUE}}
}

void testNewArray() {
  clang_analyzer_eval(new NoThrow[2]); // expected-warning{{TRUE}}
  clang_analyzer_eval(new NoExcept[2]); // expected-warning{{TRUE}}
  clang_analyzer_eval(new DefaultThrow[2]); // expected-warning{{TRUE}}
  clang_analyzer_eval(new ExplicitThrow[2]); // expected-warning{{TRUE}}
}

extern void *operator new[](size_t, int) noexcept;

void testNewArrayNoThrow() {
  clang_analyzer_eval(new (1) NoThrow[2]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(new (1) NoExcept[2]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(new (1) DefaultThrow[2]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(new (1) ExplicitThrow[2]); // expected-warning{{UNKNOWN}}
}
