// RUN: %clang_analyze_cc1 -std=c++20 -analyzer-checker=core,debug.ExprInspection -analyzer-config inline-lambdas=true -verify %s

#include "Inputs/system-header-simulator-cxx.h"
void clang_analyzer_warnIfReached();
void clang_analyzer_eval(int);

void capture_structured_binding_to_array_byref() {
  int arr[] {5};
  auto& [i] = arr;
  [i]() mutable {
    if (i != 5)
      clang_analyzer_warnIfReached();
    ++i;
  }();
  [&i] {
    if (i != 5)
      clang_analyzer_warnIfReached();
  }();
  [&i] {
    if (i != 5)
      clang_analyzer_warnIfReached();
    i++;
  }();
  clang_analyzer_eval(i == 6); // expected-warning{{TRUE}}
}

void capture_structured_binding_to_array_byvalue() {
  int arr[] {5};
  auto [i] = arr;
  [i]() mutable {
    if (i != 5)
      clang_analyzer_warnIfReached();
    ++i;
  }();
  [&i] {
    if (i != 5)
      clang_analyzer_warnIfReached();
  }();
  [&i] {
    if (i != 5)
      clang_analyzer_warnIfReached();
    i++;
  }();
  clang_analyzer_eval(i == 6); // expected-warning{{TRUE}}
}

void capture_structured_binding_to_tuple_like_byref() {
  std::pair<int, int> p {5, 6};
  auto& [i, _] = p;
  [i]() mutable {
    if (i != 5)
      clang_analyzer_warnIfReached();
    ++i;
  }();
  [&i] {
    if (i != 5)
      clang_analyzer_warnIfReached();
  }();
  [&i] {
    if (i != 5)
      clang_analyzer_warnIfReached();
    i++;
  }();
  clang_analyzer_eval(i == 6); // expected-warning{{TRUE}}
}

void capture_structured_binding_to_tuple_like_byvalue() {
  std::pair<int, int> p {5, 6};
  auto [i, _] = p;
  [i]() mutable {
    if (i != 5)
      clang_analyzer_warnIfReached();
    ++i;
  }();
  [&i] {
    if (i != 5)
      clang_analyzer_warnIfReached();
  }();
  [&i] {
    if (i != 5)
      clang_analyzer_warnIfReached();
    i++;
  }();
  clang_analyzer_eval(i == 6); // expected-warning{{TRUE}}
}

struct S { int x; };

void capture_structured_binding_to_data_member_byref() {
  S s{5};
  auto& [i] = s;
  [i]() mutable {
    if (i != 5)
      clang_analyzer_warnIfReached();
    ++i;
  }();
  [&i] {
    if (i != 5)
      clang_analyzer_warnIfReached();
  }();
  [&i] {
    if (i != 5)
      clang_analyzer_warnIfReached();
    i++;
  }();
  clang_analyzer_eval(i == 6); // expected-warning{{TRUE}}
}

void capture_structured_binding_to_data_member_byvalue() {
  S s{5};
  auto [i] = s;
  [i]() mutable {
    if (i != 5)
      clang_analyzer_warnIfReached();
    ++i;
  }();
  [&i] {
    if (i != 5)
      clang_analyzer_warnIfReached();
  }();
  [&i] {
    if (i != 5)
      clang_analyzer_warnIfReached();
    i++;
  }();
  clang_analyzer_eval(i == 6); // expected-warning{{TRUE}}
}
