// RUN: %clang_cc1 %s -verify -fsyntax-only
// expected-no-diagnostics

_Atomic(unsigned int) data1;
int _Atomic data2;

// Shift operations

int func_01 (int x) {
  return data1 << x;
}

int func_02 (int x) {
  return x << data1;
}

int func_03 (int x) {
  return data2 << x;
}

int func_04 (int x) {
  return x << data2;
}

int func_05 (void) {
  return data2 << data1;
}

int func_06 (void) {
  return data1 << data2;
}

void func_07 (int x) {
  data1 <<= x;
}

void func_08 (int x) {
  data2 <<= x;
}

void func_09 (int* xp) {
  *xp <<= data1;
}

void func_10 (int* xp) {
  *xp <<= data2;
}

int func_11 (int x) {
  return data1 == x;
}

int func_12 (void) {
  return data1 < data2;
}

int func_13 (int x, unsigned y) {
  return x ? data1 : y;
}

int func_14 (void) {
  return data1 == 0;
}
