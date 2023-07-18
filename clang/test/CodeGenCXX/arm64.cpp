// RUN: %clang_cc1 %s -triple=arm64-apple-ios -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=arm64-apple-ios -emit-llvm -o - | FileCheck -check-prefix=CHECK-GLOBALS %s

// __cxa_guard_acquire argument is 64-bit
// rdar://11540122
struct A {
  A();
};

void f() {
  // CHECK: call i32 @__cxa_guard_acquire(ptr
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
  // CHECK: define{{.*}} i32 @_ZN5test14testEv()
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
  // CHECK-GLOBALS-DAG: @_ZTSN5test21AE ={{.*}} constant [11 x i8]
  // CHECK-GLOBALS-DAG: @_ZTIN5test21AE ={{.*}} constant { {{.*}}, ptr @_ZTSN5test21AE }

  struct __attribute__((visibility("hidden"))) B {};
  const std::type_info &b0 = typeid(B);
  // CHECK-GLOBALS-DAG: @_ZTSN5test21BE = linkonce_odr hidden constant
  // CHECK-GLOBALS-DAG: @_ZTIN5test21BE = linkonce_odr hidden constant { {{.*}}, ptr @_ZTSN5test21BE }

  const std::type_info &b1 = typeid(B*);
  // CHECK-GLOBALS-DAG: @_ZTSPN5test21BE = linkonce_odr hidden constant
  // CHECK-GLOBALS-DAG: @_ZTIPN5test21BE = linkonce_odr hidden constant { {{.*}}, ptr @_ZTSPN5test21BE, i32 0, ptr @_ZTIN5test21BE

  struct C {};
  const std::type_info &c0 = typeid(C);
  // CHECK-GLOBALS-DAG: @_ZTSN5test21CE = linkonce_odr hidden constant
  // CHECK-GLOBALS-DAG: @_ZTIN5test21CE = linkonce_odr hidden constant { {{.*}}, ptr inttoptr (i64 add (i64 ptrtoint (ptr @_ZTSN5test21CE to i64), i64 -9223372036854775808) to ptr) }

  const std::type_info &c1 = typeid(C*);
  // CHECK-GLOBALS-DAG: @_ZTSPN5test21CE = linkonce_odr hidden constant
  // CHECK-GLOBALS-DAG: @_ZTIPN5test21CE = linkonce_odr hidden constant { {{.*}}, ptr inttoptr (i64 add (i64 ptrtoint (ptr @_ZTSPN5test21CE to i64), i64 -9223372036854775808) to ptr), i32 0, ptr @_ZTIN5test21CE

  // This class is explicitly-instantiated, but that instantiation
  // doesn't guarantee to emit RTTI, so we can still demote the visibility.
  template <class T> class D {};
  template class D<int>;
  const std::type_info &d0 = typeid(D<int>);
  // CHECK-GLOBALS-DAG: @_ZTSN5test21DIiEE = linkonce_odr hidden constant
  // CHECK-GLOBALS-DAG: @_ZTIN5test21DIiEE = linkonce_odr hidden constant { {{.*}}, ptr inttoptr (i64 add (i64 ptrtoint (ptr @_ZTSN5test21DIiEE to i64), i64 -9223372036854775808) to ptr) }

  // This class is explicitly-instantiated and *does* guarantee to
  // emit RTTI, so we're stuck with having to use default visibility.
  template <class T> class E {
    virtual void foo() {}
  };
  template class E<int>;
  // CHECK-GLOBALS-DAG: @_ZTSN5test21EIiEE = weak_odr constant [14 x i8]
  // CHECK-GLOBALS-DAG: @_ZTIN5test21EIiEE = weak_odr constant { {{.*}}, ptr inttoptr (i64 add (i64 ptrtoint (ptr @_ZTSN5test21EIiEE to i64), i64 -9223372036854775808) to ptr) }

}

// ARM64 reserves the top half of the vtable offset in virtual
// member pointers.
namespace test3 {
  struct A {
    virtual void foo();
    virtual void bar();
  };

  // The offset half of the pointer is still initialized to zero.
  // CHECK-GLOBALS-DAG: @_ZN5test34mptrE ={{.*}} global { i64, i64 } { i64 0, i64 1 }
  void (A::*mptr)() = &A::foo;

  // CHECK-LABEL: define{{.*}} void @_ZN5test34testEv()
  // CHECK:       [[TEMP:%.*]] = alloca [[A:.*]], align 8
  // CHECK:       [[MEMPTR:%.*]] = load { i64, i64 }, ptr @_ZN5test34mptrE, align 8
  // CHECK:       [[ADJUST_AND_IS_VIRTUAL:%.*]] = extractvalue { i64, i64 } [[MEMPTR]], 1
  // CHECK:       [[ADJUST:%.*]] = ashr i64 [[ADJUST_AND_IS_VIRTUAL]], 1
  // CHECK:       [[T1:%.*]] = getelementptr inbounds i8, ptr [[TEMP]], i64 [[ADJUST]]
  // CHECK:       [[MEMBER:%.*]] = extractvalue { i64, i64 } [[MEMPTR]], 0
  // CHECK:       [[T0:%.*]] = and i64 [[ADJUST_AND_IS_VIRTUAL]], 1
  // CHECK:       [[IS_VIRTUAL:%.*]] = icmp ne i64 [[T0]], 0
  // CHECK:       br i1 [[IS_VIRTUAL]],
  // CHECK:       [[VPTR:%.*]] = load ptr, ptr [[T1]], align 8
  // CHECK:       [[TRUNC:%.*]] = trunc i64 [[MEMBER]] to i32
  // CHECK:       [[ZEXT:%.*]] = zext i32 [[TRUNC]] to i64
  // CHECK:       [[T0:%.*]] = getelementptr i8, ptr [[VPTR]], i64 [[ZEXT]]
  // CHECK:       load ptr, ptr [[T0]],
  void test() {
    (A().*mptr)();
  }
}
