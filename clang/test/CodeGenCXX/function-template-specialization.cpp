// RUN: %clang_cc1 -emit-llvm -Wundefined-func-template -verify -triple %itanium_abi_triple %s -o - | FileCheck %s
// expected-no-diagnostics

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

namespace GH125747 {

template<typename F> constexpr int visit(F f) { return f(0); }
    
template <class T> int G(T t);
    
int main() { return visit([](auto s) -> int { return G(s); }); }
    
template <class T> int G(T t) {
  return 0;
}

// CHECK: define {{.*}} @_ZN8GH1257471GIiEEiT_

}
