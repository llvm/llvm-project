// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -emit-llvm -o %t
// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -fhidden-weak-vtables -emit-llvm -o %t.hidden
// RUN: FileCheck --check-prefix=CHECK-TEST1 %s < %t
// RUN: FileCheck --check-prefix=CHECK-TEST2 %s < %t
// RUN: FileCheck --check-prefix=CHECK-TEST2-HIDDEN %s < %t.hidden

#include <typeinfo>

namespace Test1 {
  // A is explicitly marked hidden, so all RTTI data should also be marked hidden.
  // CHECK-TEST1: @_ZTSN5Test11AE = linkonce_odr hidden constant
  // CHECK-TEST1: @_ZTIN5Test11AE = linkonce_odr hidden unnamed_addr constant
  // CHECK-TEST1: @_ZTSPN5Test11AE = linkonce_odr hidden constant
  // CHECK-TEST1: @_ZTIPN5Test11AE = linkonce_odr hidden unnamed_addr constant
  struct __attribute__((visibility("hidden"))) A { };

  void f() {
    (void)typeid(A);
    (void)typeid(A *);
  }
}

namespace Test2 {
  // A is weak, so its linkage should be linkoce_odr, but not marked hidden.
  // CHECK-TEST2: @_ZTSN5Test21AE = linkonce_odr constant
  // CHECK-TEST2: @_ZTIN5Test21AE = linkonce_odr unnamed_addr constant
  struct A { };

  // With -fhidden-weak-vtables, the typeinfo for A is marked hidden, but not its name.
  // CHECK-TEST2-HIDDEN: _ZTSN5Test21AE = linkonce_odr constant
  // CHECK-TEST2-HIDDEN: @_ZTIN5Test21AE = linkonce_odr hidden unnamed_addr constant
  void f() {
    (void)typeid(A);
  }
}
