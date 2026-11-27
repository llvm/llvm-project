// RUN: %clang_cc1 -triple armv7 -fms-extensions -fsyntax-only -ffreestanding -verify %s

unsigned int test_MoveFromCoprocessor(const unsigned int value) {
  return _MoveFromCoprocessor(value, 1, 2, 3, 4); // expected-error-re {{argument to {{.*}} must be a constant integer}}
}

void test_MoveToCoprocessor(const unsigned int value) {
  _MoveToCoprocessor(1, 2, value, 3, 4, 5); // expected-error-re {{argument to {{.*}} must be a constant integer}}
}

int _ReadStatusReg(int);

int test_ReadStatusReg(int r) {
  return _ReadStatusReg(0) + _ReadStatusReg(1) +
         _ReadStatusReg(2) + // expected-error {{argument value 2 is outside the valid range [0, 1]}}
         _ReadStatusReg(r);  // expected-error {{argument to '_ReadStatusReg' must be a constant integer}}
}
