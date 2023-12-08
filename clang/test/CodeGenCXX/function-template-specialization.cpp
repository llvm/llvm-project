// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple %s -o - | FileCheck %s

// CHECK-DAG: _ZZN7PR219047GetDataIiEERKibE1i = internal global i32 4
// CHECK-DAG: _ZZN7PR219047GetDataIiEERKibE1i_0 = internal global i32 2

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

namespace PR21904 {
template <typename>
const int &GetData(bool);

template <>
const int &GetData<int>(bool b) {
  static int i = 4;
  if (b) {
    static int i = 2;
    return i;
  }
  return i;
}
}
