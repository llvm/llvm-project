// RUN: %clang_cc1 -analyze -fms-extensions -std=c++1z -analyzer-checker=alpha.clone.CloneChecker -verify %s

// expected-no-diagnostics

bool foo1(int x) {
  if (x < 0) {
    __if_exists(x) { return false; }
  }
  return true;
}

// Same as above, but __if_not_exists
bool foo2(int x) {
  if (x < 0) {
    __if_not_exists(x) { return false; }
  }
  return true;
}
