// RUN: %clang_cc1 -triple armv7 -fms-extensions -fsyntax-only -ffreestanding -verify %s

unsigned int test_MoveFromCoprocessor(const unsigned int value) {
  return _MoveFromCoprocessor(value, 1, 2, 3, 4); // expected-error-re {{argument to {{.*}} must be a constant integer}}
}

void test_MoveToCoprocessor(const unsigned int value) {
  _MoveToCoprocessor(1, 2, value, 3, 4, 5); // expected-error-re {{argument to {{.*}} must be a constant integer}}
}
