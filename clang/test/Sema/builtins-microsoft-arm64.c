// RUN: %clang_cc1 -triple arm64-windows -fsyntax-only -verify \
// RUN: -fms-compatibility -ffreestanding -fms-compatibility-version=17.00 %s

#include <intrin.h>

void check__break(int x) {
  __break(-1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __break(65536); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __break(x); // expected-error {{argument to '__break' must be a constant integer}}
}

void check__getReg(void) {
  __getReg(-1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __getReg(32); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void check_ReadWriteStatusReg(int v) {
  int x;
  _ReadStatusReg(x); // expected-error {{argument to '_ReadStatusReg' must be a constant integer}}
  _WriteStatusReg(x, v); // expected-error {{argument to '_WriteStatusReg' must be a constant integer}}
}
