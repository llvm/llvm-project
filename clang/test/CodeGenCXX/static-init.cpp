// RUN: %clang_cc1 %s -triple=x86_64-pc-linuxs -emit-llvm -std=c++98 -o - | FileCheck -check-prefix=CHECK -check-prefix=CHECK98 %s
// RUN: %clang_cc1 %s -triple=x86_64-pc-linuxs -emit-llvm -std=c++11 -o - | FileCheck -check-prefix=CHECK -check-prefix=CHECK11 %s
// RUN: %clang_cc1 %s -triple=x86_64-pc-linuxs -emit-llvm -std=c++20 -o - | FileCheck -check-prefix=CHECK -check-prefix=CHECK20 %s

// CHECK: @_ZZ1hvE1i = internal global i32 0, align 4
// CHECK: @base_req ={{.*}} global [4 x i8] c"foo\00", align 1
// CHECK: @base_req_uchar ={{.*}} global [4 x i8] c"bar\00", align 1

// CHECK: @_ZZN5test31BC1EvE1u = internal global { i8, [3 x i8] } { i8 97, [3 x i8] undef }, align 4

// CHECK20: @_ZZN5test51fEvE1a = internal constant %"struct.test5::A" { i32 42 }

// CHECK: @_ZZ2h2vE1i = linkonce_odr global i32 0, comdat, align 4
// CHECK: @_ZGVZ2h2vE1i = linkonce_odr global i64 0, comdat, align 8{{$}}
// CHECK: @_ZZN5test1L6getvarEiE3var = internal constant [4 x i32] [i32 1, i32 0, i32 2, i32 4], align 16
// CHECK98: @_ZZN5test414useStaticLocalEvE3obj = linkonce_odr global %"struct.test4::HasVTable" zeroinitializer, comdat, align 8
// CHECK11: @_ZZN5test414useStaticLocalEvE3obj = linkonce_odr global %"struct.test4::HasVTable" { ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTVN5test49HasVTableE, i32 0, i32 0, i32 2) }, comdat, align 8

struct A {
  A();
  ~A();
};

void f() {
  // CHECK: load atomic i8, ptr @_ZGVZ1fvE1a acquire, align 8
  // CHECK: call i32 @__cxa_guard_acquire
  // CHECK: call void @_ZN1AC1Ev
  // CHECK: call i32 @__cxa_atexit(ptr @_ZN1AD1Ev, ptr @_ZZ1fvE1a, ptr @__dso_handle)
  // CHECK: call void @__cxa_guard_release
  static A a;
}

void g() {
  // CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 1)
  // CHECK: call void @_ZN1AC1Ev(
  static A& a = *new A;
}

int a();
void h() {
  static const int i = a();
}

// CHECK: define linkonce_odr void @_Z2h2v() {{.*}} comdat {
inline void h2() {
  static int i = a();
}

void h3() {
  h2();
}

// PR6980: this shouldn't crash
namespace test0 {
  struct A { A(); };
  __attribute__((noreturn)) int throw_exception();

  void test() {
    throw_exception();
    static A r;
  }
}

namespace test1 {
  // CHECK-LABEL: define internal noundef i32 @_ZN5test1L6getvarEi(
  static inline int getvar(int index) {
    static const int var[] = { 1, 0, 2, 4 };
    return var[index];
  }

  void test() { (void) getvar(2); }
}

// Make sure we emit the initializer correctly for the following:
char base_req[] = { "foo" };
unsigned char base_req_uchar[] = { "bar" };

namespace union_static_local {
  // CHECK-LABEL: define internal void @_ZZN18union_static_local4testEvEN1c4mainEv
  // CHECK: call void @_ZN18union_static_local1fEPNS_1xE(ptr noundef @_ZZN18union_static_local4testEvE3foo)
  union x { long double y; const char *x[2]; };
  void f(union x*);
  void test() {
    static union x foo = { .x = { "a", "b" } };
    struct c {
      static void main() {
        f(&foo);
      }
    };
    c::main();
  }
}

//   Static variables should be consistent across constructor
//   or destructor variants.
namespace test2 {
  struct A {
    A();
    ~A();
  };

  struct B : virtual A {
    B();
    ~B();
  };

  // If we ever implement this as a delegate ctor call, just change
  // this to take variadic arguments or something.
  extern int foo();
  B::B() {
    static int x = foo();
  }
  // CHECK-LABEL: define{{.*}} void @_ZN5test21BC2Ev
  // CHECK:   load atomic i8, ptr @_ZGVZN5test21BC1EvE1x acquire, align 8
  // CHECK:   call i32 @__cxa_guard_acquire(ptr @_ZGVZN5test21BC1EvE1x)
  // CHECK:   [[T0:%.*]] = call noundef i32 @_ZN5test23fooEv()
  // CHECK:   store i32 [[T0]], ptr @_ZZN5test21BC1EvE1x,
  // CHECK:   call void @__cxa_guard_release(ptr @_ZGVZN5test21BC1EvE1x)

  // CHECK-LABEL: define{{.*}} void @_ZN5test21BC1Ev
  // CHECK:   load atomic i8, ptr @_ZGVZN5test21BC1EvE1x acquire, align 8
  // CHECK:   call i32 @__cxa_guard_acquire(ptr @_ZGVZN5test21BC1EvE1x)
  // CHECK:   [[T0:%.*]] = call noundef i32 @_ZN5test23fooEv()
  // CHECK:   store i32 [[T0]], ptr @_ZZN5test21BC1EvE1x,
  // CHECK:   call void @__cxa_guard_release(ptr @_ZGVZN5test21BC1EvE1x)

  // This is just for completeness, because we actually emit this
  // using a delegate dtor call.
  B::~B() {
    static int y = foo();
  }
  // CHECK-LABEL: define{{.*}} void @_ZN5test21BD2Ev(
  // CHECK:   load atomic i8, ptr @_ZGVZN5test21BD1EvE1y acquire, align 8
  // CHECK:   call i32 @__cxa_guard_acquire(ptr @_ZGVZN5test21BD1EvE1y)
  // CHECK:   [[T0:%.*]] = call noundef i32 @_ZN5test23fooEv()
  // CHECK:   store i32 [[T0]], ptr @_ZZN5test21BD1EvE1y,
  // CHECK:   call void @__cxa_guard_release(ptr @_ZGVZN5test21BD1EvE1y)

  // CHECK-LABEL: define{{.*}} void @_ZN5test21BD1Ev(
  // CHECK:   call void @_ZN5test21BD2Ev(
}

// This shouldn't error out.
namespace test3 {
  struct A {
    A();
    ~A();
  };

  struct B : virtual A {
    B();
    ~B();
  };

  B::B() {
    union U { char x; int i; };
    static U u = { 'a' };
  }
  // CHECK-LABEL: define{{.*}} void @_ZN5test31BC2Ev(
  // CHECK-LABEL: define{{.*}} void @_ZN5test31BC1Ev(
}

// We forgot to set the comdat when replacing the global with a different type.
namespace test4 {
struct HasVTable {
  virtual void f();
};
inline HasVTable &useStaticLocal() {
  static HasVTable obj;
  return obj;
}
void useit() {
  useStaticLocal();
}
// CHECK: define linkonce_odr noundef nonnull align 8 dereferenceable(8) ptr @_ZN5test414useStaticLocalEv()
// CHECK: ret ptr{{.*}} @_ZZN5test414useStaticLocalEvE3obj
}

#if __cplusplus >= 202002L
// A const object with constexpr destructor can be emitted as a constant.
namespace test5 {
  struct A {
    constexpr A(int x) : x_(x) {}
    constexpr ~A() {}
    int x_;
  };
  const int *f() {
    static const A a{42};
    return &a.x_;
  }
}
#endif
