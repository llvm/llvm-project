// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config c++-allocator-inlining=true -std=c++11 -verify -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(bool);
void clang_analyzer_warnOnDeadSymbol(int);

typedef __typeof__(sizeof(int)) size_t;

int conjure();
void exit(int);

struct S {
  S() {}
  ~S() {}

  static S buffer[1000];

  // This operator allocates stuff within the buffer. Additionally, it never
  // places anything at the beginning of the buffer.
  void *operator new(size_t size) {
    int i = conjure();
    if (i == 0)
      exit(1);
    // Let's see if the symbol dies before new-expression is evaluated.
    // It shouldn't.
    clang_analyzer_warnOnDeadSymbol(i);
    return buffer + i;
  }
};

void testIndexLiveness() {
  S *s = new S();
  clang_analyzer_eval(s == S::buffer); // expected-warning{{FALSE}}
} // expected-warning{{SYMBOL DEAD}}
