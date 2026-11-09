// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,alpha.cplusplus.IteratorRange,debug.DebugIteratorModeling,debug.ExprInspection -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=false -verify %s
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,alpha.cplusplus.IteratorRange,debug.DebugIteratorModeling,debug.ExprInspection -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=true -verify %s

#include "Inputs/system-header-simulator-cxx.h"

template <typename Iterator>
long clang_analyzer_iterator_position(const Iterator &);
template <typename Iterator>
void *clang_analyzer_iterator_container(const Iterator &);
void clang_analyzer_eval(bool);

// Constrain the abstract offset so that advancing it produces a concrete value.
void prefix_increment(const std::vector<int> &V) {
  auto I = V.begin();
  if (clang_analyzer_iterator_position(I) != 1)
    return;

  auto J = ++I;
  clang_analyzer_eval(clang_analyzer_iterator_container(I) == &V); // expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_container(J) == &V); // expected-warning{{TRUE}}
  // expected-warning@+1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(I) ==
                      clang_analyzer_iterator_position(J));
  clang_analyzer_eval(clang_analyzer_iterator_position(I) == 2); // expected-warning{{TRUE}} expected-warning{{FALSE}} FIXME: This should be only TRUE.
}

void postfix_increment(const std::vector<int> &V) {
  auto I = V.begin();
  if (clang_analyzer_iterator_position(I) != 1)
    return;

  auto J = I++;
  clang_analyzer_eval(clang_analyzer_iterator_container(I) == &V); // expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_container(J) == &V); // expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(J) == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(I) == 2); // expected-warning{{TRUE}} expected-warning{{FALSE}} FIXME: This should be only TRUE.
}

void prefix_decrement(const std::vector<int> &V) {
  auto I = V.end();
  if (clang_analyzer_iterator_position(I) != 1)
    return;

  auto J = --I;
  clang_analyzer_eval(clang_analyzer_iterator_container(I) == &V); // expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_container(J) == &V); // expected-warning{{TRUE}}
  // expected-warning@+1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(I) ==
                      clang_analyzer_iterator_position(J));
  clang_analyzer_eval(clang_analyzer_iterator_position(I) == 0); // expected-warning{{TRUE}} expected-warning{{FALSE}} FIXME: This should be only TRUE.
}

void postfix_decrement(const std::vector<int> &V) {
  auto I = V.end();
  if (clang_analyzer_iterator_position(I) != 1)
    return;

  auto J = I--;
  clang_analyzer_eval(clang_analyzer_iterator_container(I) == &V); // expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_container(J) == &V); // expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(J) == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(I) == 0); // expected-warning{{TRUE}} expected-warning{{FALSE}} FIXME: This should be only TRUE.
}

void compound_assignment(const std::vector<int> &V) {
  auto I = V.begin();
  if (clang_analyzer_iterator_position(I) != 1)
    return;

  I += 2;
  clang_analyzer_eval(clang_analyzer_iterator_container(I) == &V); // expected-warning{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(I) == 3); // expected-warning{{TRUE}} expected-warning{{FALSE}} FIXME: This should be only TRUE.
}
