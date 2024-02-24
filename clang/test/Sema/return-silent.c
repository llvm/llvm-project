// RUN: %clang_cc1 %s -Wno-return-type -Wno-return-mismatch -fsyntax-only -verify
// expected-no-diagnostics

int t14(void) {
  return;
}

void t15(void) {
  return 1;
}
