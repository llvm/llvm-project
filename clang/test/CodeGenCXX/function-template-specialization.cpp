// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
template<typename T, typename U>
T* next(T* ptr, const U& diff);

template<typename T, typename U>
T* next(T* ptr, const U& diff) { 
  return ptr + diff; 
}

void test(int *iptr, float *fptr, int diff) {
  // CHECK: _Z4nextIiiEPT_S1_RKT0_
  iptr = next(iptr, diff);

  // CHECK: _Z4nextIfiEPT_S1_RKT0_
  fptr = next(fptr, diff);
}

template<typename T, typename U>
T* next(T* ptr, const U& diff);

void test2(int *iptr, double *dptr, int diff) {
  iptr = next(iptr, diff);

  // CHECK: _Z4nextIdiEPT_S1_RKT0_
  dptr = next(dptr, diff);
}
