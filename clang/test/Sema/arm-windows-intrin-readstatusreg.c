// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=19.00 \
// RUN:         -triple thumbv7--windows -fsyntax-only -verify %s

// _ReadStatusReg takes only 0 (APSR) or 1 (SPSR), as with MSVC.

// intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;
#include <intrin.h>

int ok(void) { return _ReadStatusReg(0) + _ReadStatusReg(1); }

int out_of_range(void) {
  return _ReadStatusReg(2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
}

int not_constant(int r) {
  return _ReadStatusReg(r); // expected-error {{argument to '_ReadStatusReg' must be a constant integer}}
}
