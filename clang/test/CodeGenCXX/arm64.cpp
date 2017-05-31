// RUN: %clang_cc1 %s -triple=arm64-apple-ios -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=arm64-apple-ios -emit-llvm -o - | FileCheck -check-prefix=CHECK-GLOBALS %s

// __cxa_guard_acquire argument is 64-bit
// rdar://11540122
struct A {
  A();
};

void f() {
  // CHECK: call i32 @__cxa_guard_acquire(i64*
  static A a;
}

// ARM64 uses the C++11 definition of POD.
// rdar://12650514
namespace test1 {
  // This class is POD in C++11 and cannot have objects allocated in
  // its tail-padding.
  struct ABase {};
  struct A : ABase {
    int x;
    char c;
  };

  struct B : A {
    char d;
  };

  int test() {
    return sizeof(B);
  }
  // CHECK: define i32 @_ZN5test14testEv()
  // CHECK: ret i32 12
}

namespace std {
  class type_info;
}

// ARM64 uses string comparisons for what would otherwise be
// default-visibility weak RTTI.  rdar://12650568
namespace test2 {
  struct A {
    virtual void foo();
  };
  void A::foo() {}
  // CHECK-GLOBALS-DAG: @_ZTSN5test21AE = constant [11 x i8]
  // CHECK-GLOBALS-DAG: @_ZTIN5test21AE = constant { {{.*}}, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @_ZTSN5test21AE, i32 0, i32 0) }

  struct __attribute__((visibility("hidden"))) B {};
  const std::type_info &b0 = typeid(B);
  // CHECK-GLOBALS-DAG: @_ZTSN5test21BE = linkonce_odr hidden constant
  // CHECK-GLOBALS-DAG: @_ZTIN5test21BE = linkonce_odr hidden constant { {{.*}}, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @_ZTSN5test21BE, i32 0, i32 0) }

  const std::type_info &b1 = typeid(B*);
  // CHECK-GLOBALS-DAG: @_ZTSPN5test21BE = linkonce_odr hidden constant
  // CHECK-GLOBALS-DAG: @_ZTIPN5test21BE = linkonce_odr hidden constant { {{.*}}, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @_ZTSPN5test21BE, i32 0, i32 0), i32 0, i8* bitcast

  struct C {};
  const std::type_info &c0 = typeid(C);
  // CHECK-GLOBALS-DAG: @_ZTSN5test21CE = linkonce_odr hidden constant
  // CHECK-GLOBALS-DAG: @_ZTIN5test21CE = linkonce_odr hidden constant { {{.*}}, i8* inttoptr (i64 add (i64 ptrtoint ([11 x i8]* @_ZTSN5test21CE to i64), i64 -9223372036854775808) to i8*) }

  const std::type_info &c1 = typeid(C*);
  // CHECK-GLOBALS-DAG: @_ZTSPN5test21CE = linkonce_odr hidden constant
  // CHECK-GLOBALS-DAG: @_ZTIPN5test21CE = linkonce_odr hidden constant { {{.*}}, i8* inttoptr (i64 add (i64 ptrtoint ([12 x i8]* @_ZTSPN5test21CE to i64), i64 -9223372036854775808) to i8*), i32 0, i8* bitcast

  // This class is explicitly-instantiated, but that instantiation
  // doesn't guarantee to emit RTTI, so we can still demote the visibility.
  template <class T> class D {};
  template class D<int>;
  const std::type_info &d0 = typeid(D<int>);
  // CHECK-GLOBALS-DAG: @_ZTSN5test21DIiEE = linkonce_odr hidden constant
  // CHECK-GLOBALS-DAG: @_ZTIN5test21DIiEE = linkonce_odr hidden constant { {{.*}}, i8* inttoptr (i64 add (i64 ptrtoint ([14 x i8]* @_ZTSN5test21DIiEE to i64), i64 -9223372036854775808) to i8*) }

  // This class is explicitly-instantiated and *does* guarantee to
  // emit RTTI, so we're stuck with having to use default visibility.
  template <class T> class E {
    virtual void foo() {}
  };
  template class E<int>;
  // CHECK-GLOBALS-DAG: @_ZTSN5test21EIiEE = weak_odr constant [14 x i8]
  // CHECK-GLOBALS-DAG: @_ZTIN5test21EIiEE = weak_odr constant { {{.*}}, i8* inttoptr (i64 add (i64 ptrtoint ([14 x i8]* @_ZTSN5test21EIiEE to i64), i64 -9223372036854775808) to i8*) }

}
