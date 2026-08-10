// RUN: %clang_analyze_cc1 %s -analyzer-checker=core.uninitialized.NewArraySize -analyzer-output=text -verify

#include "Inputs/system-header-simulator-cxx.h"

void checkUndefinedElmenetCountValue() {
  int n;
  // expected-note@-1{{'n' declared without an initial value}}

  int *arr = new int[n]; // expected-warning{{Element count in new[] is a garbage value}}
  // expected-note@-1{{Element count in new[] is a garbage value}}
}

void checkUndefinedElmenetCountMultiDimensionalValue() {
  int n;
  // expected-note@-1{{'n' declared without an initial value}}

  auto *arr = new int[n][5]; // expected-warning{{Element count in new[] is a garbage value}}
  // expected-note@-1{{Element count in new[] is a garbage value}}
}

void checkUndefinedElmenetCountReference() {
  int n;
  // expected-note@-1{{'n' declared without an initial value}}

  int &ref = n;
  // expected-note@-1{{'ref' initialized here}}

  int *arr = new int[ref]; // expected-warning{{Element count in new[] is a garbage value}}
  // expected-note@-1{{Element count in new[] is a garbage value}}
}

void checkUndefinedElmenetCountMultiDimensionalReference() {
  int n;
  // expected-note@-1{{'n' declared without an initial value}}

  int &ref = n;
  // expected-note@-1{{'ref' initialized here}}

  auto *arr = new int[ref][5]; // expected-warning{{Element count in new[] is a garbage value}}
  // expected-note@-1{{Element count in new[] is a garbage value}}
}

int foo() {
  int n;

  return n;
}

void checkUndefinedElmenetCountFunction() {
  int *arr = new int[foo()]; // expected-warning{{Element count in new[] is a garbage value}}
  // expected-note@-1{{Element count in new[] is a garbage value}}
}

void checkUndefinedElmenetCountMultiDimensionalFunction() {
  auto *arr = new int[foo()][5]; // expected-warning{{Element count in new[] is a garbage value}}
  // expected-note@-1{{Element count in new[] is a garbage value}}
}

void *malloc(size_t);

void checkUndefinedPlacementElementCount() {
  int n;
  // expected-note@-1{{'n' declared without an initial value}}
  
  void *buffer = malloc(sizeof(std::string) * 10);
  std::string *p =
      ::new (buffer) std::string[n]; // expected-warning{{Element count in new[] is a garbage value}}
  // expected-note@-1{{Element count in new[] is a garbage value}}
}
