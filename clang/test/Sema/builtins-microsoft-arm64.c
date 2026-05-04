// RUN: %clang_cc1 -triple arm64-windows -fsyntax-only -verify \
// RUN: -fms-compatibility -ffreestanding -fms-compatibility-version=17.00 %s

#include <intrin.h>

void check__break(int x) {
  __break(-1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __break(65536); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __break(x); // expected-error {{argument to '__break' must be a constant integer}}
}

void check__hlt() {
  __hlt(-1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __hlt(65536); // expected-error-re {{argument value {{.*}} is outside the valid range}}
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

void check_ReadWriteStatusReg_range(int v) {
  _ReadStatusReg(0x3fff);      // expected-error-re {{argument value {{.*}} is outside the valid range}}
  _ReadStatusReg(0x8000); // expected-error-re {{argument value {{.*}} is outside the valid range}}

  _WriteStatusReg(0x3fff, v);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  _WriteStatusReg(0x8000, v); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void check__sys(int v) {
  int x;
  __sys(x, v); // expected-error {{argument to '__sys' must be a constant integer}}
}

void check__sys_range(int v) {
  __sys(-1, v);      // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __sys(0x4000, v);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

unsigned int check__sys_retval() {
  return __sys(0, 1); // builtin has superfluous return value for MSVC compatibility
}
