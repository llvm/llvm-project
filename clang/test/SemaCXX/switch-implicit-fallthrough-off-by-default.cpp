// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -DUNREACHABLE=1 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -DUNREACHABLE=0 -Wimplicit-fallthrough %s

void fallthrough(int n) {
  switch (n) {
  case 1:
    if (UNREACHABLE)
      return;
    [[fallthrough]]; // expected-no-diagnostics, only checked when UNREACHABLE=0
  case 2:
    break;
  }
}
