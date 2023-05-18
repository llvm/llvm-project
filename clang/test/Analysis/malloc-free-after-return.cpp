// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.NewDelete -verify %s

#include "Inputs/system-header-simulator-cxx.h"

struct S {
  S() : Data(new int) {}
  ~S() { delete Data; }
  int *getData() { return Data; }

private:
  int *Data;
};

int *freeAfterReturnTemp() {
  return S().getData(); // expected-warning {{Use of memory after it is freed}}
}

int *freeAfterReturnLocal() {
  S X;
  return X.getData(); // expected-warning {{Use of memory after it is freed}}
}
