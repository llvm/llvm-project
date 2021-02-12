// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-apple-darwin9 | FileCheck %s

namespace PR11411 {
  template<typename _Tp> struct Ptr {
    void f();
  };

  // CHECK-LABEL: define linkonce_odr void @_ZN7PR114113PtrIiE1fEv
  // CHECK-NOT: ret
  template<typename _Tp> inline void Ptr<_Tp>::f() {
    int* _refcount;
    // CHECK: atomicrmw add i32* {{.*}} seq_cst, align 4
    __sync_fetch_and_add(_refcount, 1);
    // CHECK-NEXT: ret void
  }
  void f(Ptr<int> *a) { a->f(); }
}
