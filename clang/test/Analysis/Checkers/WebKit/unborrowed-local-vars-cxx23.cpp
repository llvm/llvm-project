// RUN: %clang_analyze_cc1 -std=c++23 -analyzer-checker=alpha.webkit.UnborrowedLocalVarsChecker -verify %s

#include "mock-canborrow.h"

void someFunction();

void borrow_function_get_loop(Vector<char> &vec) {
  for (char &c : borrow(vec).get()) {
    someFunction();
    (void)c;
  }
}

struct ReversedChars {
  char *b;
  char *e;
  char *begin() const;
  char *end() const;
};
struct ReverseAdaptor {};
ReversedChars operator|(Vector<char> &vec, ReverseAdaptor);

void borrow_get_pipe_loop(Vector<char> &vec) {
  for (char &c : borrow(vec).get() | ReverseAdaptor()) {
    someFunction();
    (void)c;
  }
}

void unguarded_pipe_loop(Vector<char> &vec) {
  for (char &c : vec | ReverseAdaptor()) {
    someFunction();
    (void)c;
  }
}

void reference_loop(Vector<char> &vec) {
  for (char &c : vec) {
    // expected-warning@-1{{Local variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLocalVarsChecker]}}
    someFunction();
    (void)c;
  }
}
