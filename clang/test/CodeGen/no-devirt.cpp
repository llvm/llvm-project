// RUN: %clang_cc1 %s -DUSEIT -triple %itanium_abi_triple -emit-llvm -o - |  FileCheck %s

// Test with decls and template defs in pch, and just use in .cpp
// RUN:  %clang_cc1 %s -DTMPL_DEF_IN_HEADER -triple %itanium_abi_triple -emit-pch -o %t
// RUN:  %clang_cc1 %s -DTMPL_DEF_IN_HEADER -DUSEIT -triple %itanium_abi_triple -include-pch %t -emit-llvm -o - | FileCheck %s

// Test with A in pch, and B and C in main
// Test with just decls in pch, and template defs and use in .cpp
// RUN:  %clang_cc1 %s -triple %itanium_abi_triple -emit-pch -o %t
// RUN:  %clang_cc1 %s -DUSEIT -triple %itanium_abi_triple -include-pch %t -emit-llvm -o - | FileCheck %s

#ifndef HEADER
#define HEADER
template < typename T, int N = 0 > class TmplWithArray {
public:
  virtual T& operator [] (int idx);
  virtual T& func1 (int idx);
  virtual T& func2 (int idx);
  T ar[N+1];
};
struct Wrapper {
  TmplWithArray<bool, 10> data;
  bool indexIt(int a) {
    if (a > 6) return data[a] ;      // Should devirtualize
    if (a > 4) return data.func1(a); // Should devirtualize
    return data.func2(a);            // Should devirtualize
  }
};

#ifdef TMPL_DEF_IN_HEADER
template <typename T, int N> T& TmplWithArray<T, N >::operator[](int idx) {
  return ar[idx];
}
template <typename T, int N> T& TmplWithArray<T, N >::func1(int idx) {
  return ar[idx];
}
#endif // TMPL_DEF_IN_HEADER
#endif // HEADER

#ifdef USEIT
#ifndef TMPL_DEF_IN_HEADER
template <typename T, int N> T& TmplWithArray<T, N >::operator[](int idx) {
  return ar[idx];
}
template <typename T, int N> T& TmplWithArray<T, N >::func1(int idx) {
  return ar[idx];
}
#endif // !TMPL_DEF_IN_HEADER
extern Wrapper ew;
bool stuff(int p)
{
  return ew.indexIt(p);
}
#endif

// CHECK-DAG: call {{.*}} @_ZN13TmplWithArrayIbLi10EEixEi
// CHECK-DAG: call {{.*}} @_ZN13TmplWithArrayIbLi10EE5func1Ei
// CHECK-DAG: call {{.*}} @_ZN13TmplWithArrayIbLi10EE5func2Ei

