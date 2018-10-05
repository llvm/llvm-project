// RUN: %clang_cc1 -triple arm64-windows -fsyntax-only -verify \
// RUN: -fms-compatibility -ffreestanding -fms-compatibility-version=17.00 %s

#include <intrin.h>

void check__getReg() {
  __getReg(-1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __getReg(32); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}
