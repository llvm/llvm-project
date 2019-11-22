// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

struct S {
  S();
  ~S();
};

bool bar(S);

// no-crash during diagnostic construction.
void foo() {
  int x;
  if (true && bar(S()))
    ++x; // expected-warning{{The expression is an uninitialized value. The computed value will also be garbage}}
}
