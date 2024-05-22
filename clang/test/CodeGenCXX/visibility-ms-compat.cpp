// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-apple-darwin10 -fvisibility=hidden -ftype-visibility=default -emit-llvm -o %t
// RUN: FileCheck %s < %t
// RUN: FileCheck -check-prefix=CHECK-GLOBAL %s < %t

// The two visibility options above are how we translate
// -fvisibility-ms-compat in the driver.

#define HIDDEN __attribute__((visibility("hidden")))
#define PROTECTED __attribute__((visibility("protected")))
#define DEFAULT __attribute__((visibility("default")))

namespace std {
  class type_info;
};

namespace test0 {
  struct A {
    static void foo();
    static void bar();
  };

  void A::foo() { bar(); }
  // CHECK-LABEL: define hidden void @_ZN5test01A3fooEv()
  // CHECK: declare void @_ZN5test01A3barEv()

  const std::type_info &ti = typeid(A);
  // CHECK-GLOBAL: @_ZTSN5test01AE = linkonce_odr constant
  // CHECK-GLOBAL: @_ZTIN5test01AE = linkonce_odr constant
  // CHECK-GLOBAL: @_ZN5test02tiE = hidden constant
}

namespace test1 {
  struct HIDDEN A {
    static void foo();
    static void bar();
  };

  void A::foo() { bar(); }
  // CHECK-LABEL: define hidden void @_ZN5test11A3fooEv()
  // CHECK: declare hidden void @_ZN5test11A3barEv()

  const std::type_info &ti = typeid(A);
  // CHECK-GLOBAL: @_ZTSN5test11AE = linkonce_odr hidden constant
  // CHECK-GLOBAL: @_ZTIN5test11AE = linkonce_odr hidden constant
  // CHECK-GLOBAL: @_ZN5test12tiE = hidden constant
}

namespace test2 {
  struct DEFAULT A {
    static void foo();
    static void bar();
  };

  void A::foo() { bar(); }
  // CHECK-LABEL: define{{.*}} void @_ZN5test21A3fooEv()
  // CHECK: declare void @_ZN5test21A3barEv()

  const std::type_info &ti = typeid(A);
  // CHECK-GLOBAL: @_ZTSN5test21AE = linkonce_odr constant
  // CHECK-GLOBAL: @_ZTIN5test21AE = linkonce_odr constant
  // CHECK-GLOBAL: @_ZN5test22tiE = hidden constant
}

namespace test3 {
  struct A { int x; };
  template <class T> struct B {
    static void foo() { bar(); }
    static void bar();
  };

  template void B<A>::foo();
  // CHECK-LABEL: define weak_odr hidden void @_ZN5test31BINS_1AEE3fooEv()
  // CHECK: declare void @_ZN5test31BINS_1AEE3barEv()

  const std::type_info &ti = typeid(B<A>);
  // CHECK-GLOBAL: @_ZTSN5test31BINS_1AEEE = linkonce_odr constant
  // CHECK-GLOBAL: @_ZTIN5test31BINS_1AEEE = linkonce_odr constant
}

namespace test4 {
  struct A { int x; };
  template <class T> struct DEFAULT B {
    static void foo() { bar(); }
    static void bar();
  };

  template void B<A>::foo();
  // CHECK-LABEL: define weak_odr void @_ZN5test41BINS_1AEE3fooEv()
  // CHECK: declare void @_ZN5test41BINS_1AEE3barEv()

  const std::type_info &ti = typeid(B<A>);
  // CHECK-GLOBAL: @_ZTSN5test41BINS_1AEEE = linkonce_odr constant
  // CHECK-GLOBAL: @_ZTIN5test41BINS_1AEEE = linkonce_odr constant
}

namespace test5 {
  struct A { int x; };
  template <class T> struct HIDDEN B {
    static void foo() { bar(); }
    static void bar();
  };

  template void B<A>::foo();
  // CHECK-LABEL: define weak_odr hidden void @_ZN5test51BINS_1AEE3fooEv()
  // CHECK: declare hidden void @_ZN5test51BINS_1AEE3barEv()

  const std::type_info &ti = typeid(B<A>);
  // CHECK-GLOBAL: @_ZTSN5test51BINS_1AEEE = linkonce_odr hidden constant
  // CHECK-GLOBAL: @_ZTIN5test51BINS_1AEEE = linkonce_odr hidden constant
}
