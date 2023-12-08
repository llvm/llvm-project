// RUN: %clang_cc1 %s -std=c++14 -emit-llvm -o - -triple=x86_64-linux-gnu | FileCheck %s

struct AM {
  int f1, f2;
};
alignas(8) AM m;
AM load1() {
  AM am;
  // m is declared to align to 8bytes, so generate load atomic instead
  // of libcall.
  // CHECK-LABEL: @_Z5load1v
  // CHECK: load atomic {{.*}} monotonic, align 8
  __atomic_load(&m, &am, 0);
  return am;
}

struct BM {
  int f1;
  alignas(8) AM f2;
};
BM bm;
AM load2() {
  AM am;
  // BM::f2 is declared to align to 8bytes, so generate load atomic instead
  // of libcall.
  // CHECK-LABEL: @_Z5load2v
  // CHECK: load atomic {{.*}} monotonic, align 8
  __atomic_load(&bm.f2, &am, 0);
  return am;
}

namespace std {
  template <class _Tp>
  inline constexpr
  __attribute__ ((__visibility__("hidden"), __internal_linkage__))
  _Tp* __addressof(_Tp& __x) noexcept
  {
      return __builtin_addressof(__x);
  }
}

AM load3() {
  AM am;
  // m is declared to align to 8bytes, so generate load atomic instead
  // of libcall.
  // CHECK-LABEL: @_Z5load3v
  // CHECK: load atomic {{.*}} monotonic, align 8
  __atomic_load(std::__addressof(m), &am, 0);
  return am;
}
