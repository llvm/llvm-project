// RUN: %clang_cc1 -analyze -std=c++1z -analyzer-checker=alpha.clone.CloneChecker -verify %s

// expected-no-diagnostics

int foo1(int n) {
  int result = 0;
  switch (n) {
  case 33:
    result += 33;
    [[clang::fallthrough]];
  case 44:
    result += 44;
  }
  return result;
}

// Identical to foo1 except the missing attribute.
int foo2(int n) {
  int result = 0;
  switch (n) {
  case 33:
    result += 33;
    ;
  case 44:
    result += 44;
  }
  return result;
}
