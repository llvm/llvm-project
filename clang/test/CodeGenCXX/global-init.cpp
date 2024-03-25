// RUN: %clang_cc1 %std_cxx98-14 -triple=x86_64-apple-darwin10 -emit-llvm -fexceptions %s -o - |FileCheck %s --check-prefixes=CHECK,PRE17
// RUN: %clang_cc1 %std_cxx98-14 -triple=x86_64-apple-darwin10 -emit-llvm %s -o - |FileCheck %s --check-prefixes=CHECK-NOEXC,PRE17
// RUN: %clang_cc1 %std_cxx98-14 -triple=x86_64-apple-darwin10 -emit-llvm -mframe-pointer=non-leaf %s -o - \
// RUN:   | FileCheck -check-prefix CHECK-FP %s
// RUN: %clang_cc1 %std_cxx98-14 -triple=x86_64-apple-darwin10 -emit-llvm %s -o - -fno-builtin \
// RUN:   | FileCheck -check-prefix CHECK-NOBUILTIN %s
// RUN: %clang_cc1 %std_cxx17- -triple=x86_64-apple-darwin10 -emit-llvm -fexceptions %s -o - | FileCheck %s

struct A {
  A();
  ~A();
};

struct B { B(); ~B(); };

struct C { void *field; };

struct D { ~D(); };

// CHECK: @__dso_handle = external hidden global i8
// CHECK: @c ={{.*}} global %struct.C zeroinitializer, align 8

// PR6205: The casts should not require global initializers
// CHECK: @_ZN6PR59741cE = external global %"struct.PR5974::C"
// CHECK: @_ZN6PR59741aE ={{.*}} global ptr @_ZN6PR59741cE
// CHECK: @_ZN6PR59741bE ={{.*}} global ptr getelementptr (i8, ptr @_ZN6PR59741cE, i64 4), align 8

// CHECK: call void @_ZN1AC1Ev(ptr {{[^,]*}} @a)
// CHECK: call i32 @__cxa_atexit(ptr @_ZN1AD1Ev, ptr @a, ptr @__dso_handle)
A a;

// CHECK: call void @_ZN1BC1Ev(ptr {{[^,]*}} @b)
// CHECK: call i32 @__cxa_atexit(ptr @_ZN1BD1Ev, ptr @b, ptr @__dso_handle)
B b;

// PR6205: this should not require a global initializer
// CHECK-NOT: call void @_ZN1CC1Ev(ptr @c)
C c;

// CHECK: call i32 @__cxa_atexit(ptr @_ZN1DD1Ev, ptr @d, ptr @__dso_handle)
D d;

namespace test1 {
  int f();
  const int x = f();   // This has side-effects and gets emitted immediately.
  const int y = x - 1; // This gets deferred.
  const int z = ~y;    // This also gets deferred, but gets "undeferred" before y.
  int test() { return z; }
// CHECK-LABEL:      define{{.*}} i32 @_ZN5test14testEv()

  // All of these initializers end up delayed, so we check them later.
}

namespace test2 {
  struct allocator { allocator(); ~allocator(); };
  struct A { A(const allocator &a = allocator()); ~A(); };

  A a;
// CHECK: call void @_ZN5test29allocatorC1Ev(
// CHECK: invoke void @_ZN5test21AC1ERKNS_9allocatorE(
// CHECK: call void @_ZN5test29allocatorD1Ev(
// CHECK: call i32 @__cxa_atexit({{.*}} @_ZN5test21AD1Ev, {{.*}} @_ZN5test21aE
}

namespace test3 {
  // Tested at the beginning of the file.
  const char * const var = "string";
  extern const char * const var;

  const char *test() { return var; }
}

namespace test4 {
  struct A {
    A();
  };
  extern int foo();

  // This needs an initialization function and guard variables.
  // CHECK: load i8, ptr @_ZGVN5test41xE
  // CHECK: store i8 1, ptr @_ZGVN5test41xE
  // CHECK-NEXT: [[CALL:%.*]] = call noundef i32 @_ZN5test43fooEv
  // CHECK-NEXT: store i32 [[CALL]], ptr @_ZN5test41xE
  __attribute__((weak)) int x = foo();
}

namespace PR5974 {
  struct A { int a; };
  struct B { int b; };
  struct C : A, B { int c; };

  extern C c;

  // These should not require global initializers.
  A* a = &c;
  B* b = &c;
}

// PR9570: the indirect field shouldn't crash IR gen.
namespace test5 {
  static union {
    unsigned bar[4096] __attribute__((aligned(128)));
  };
}

namespace std { struct type_info; }

namespace test6 {
  struct A { virtual ~A(); };
  struct B : A {};
  extern A *p;

  // We must emit a dynamic initializer for 'q', because it could throw.
  B *const q = &dynamic_cast<B&>(*p);
  // CHECK: call void @__cxa_bad_cast()
  // CHECK: store {{.*}} @_ZN5test6L1qE

  // We don't need to emit 'r' at all, because it has internal linkage, is
  // unused, and its initialization has no side-effects.
  B *const r = dynamic_cast<B*>(p);
  // CHECK-NOT: call void @__cxa_bad_cast()
  // CHECK-NOT: store {{.*}} @_ZN5test6L1rE

  // This can throw, so we need to emit it.
  const std::type_info *const s = &typeid(*p);
  // CHECK: store {{.*}} @_ZN5test6L1sE

  // This can't throw, so we don't.
  const std::type_info *const t = &typeid(p);
  // CHECK-NOT: @_ZN5test6L1tE

  extern B *volatile v;
  // CHECK: store {{.*}} @_ZN5test6L1wE
  B *const w = dynamic_cast<B*>(v);

  // CHECK: load volatile
  // CHECK: store {{.*}} @_ZN5test6L1xE
  const int x = *(volatile int*)0x1234;

  namespace {
    int a = int();
    volatile int b = int();
    int c = a;
    int d = b;
    // CHECK-NOT: store {{.*}} @_ZN5test6{{[A-Za-z0-9_]*}}1aE
    // CHECK-NOT: store {{.*}} @_ZN5test6{{[A-Za-z0-9_]*}}1bE
    // CHECK-NOT: store {{.*}} @_ZN5test6{{[A-Za-z0-9_]*}}1cE
    // CHECK: load volatile {{.*}} @_ZN5test6{{[A-Za-z0-9_]*}}1bE
    // CHECK: store {{.*}} @_ZN5test6{{[A-Za-z0-9_]*}}1dE
  }
}

namespace test7 {
  struct A { A(); };
  struct B { ~B(); int n; };
  struct C { C() = default; C(const C&); int n; };
  struct D {};

  // CHECK: call void @_ZN5test71AC1Ev({{.*}}@_ZN5test7L1aE)
  const A a = A();

  // CHECK: call i32 @__cxa_atexit({{.*}} @_ZN5test71BD1Ev{{.*}} @_ZN5test7L2b1E
  // CHECK: call i32 @__cxa_atexit({{.*}} @_ZN5test71BD1Ev{{.*}} @_ZGRN5test72b2E
  // CHECK: call void @_ZN5test71BD1Ev(
  // CHECK: store {{.*}} @_ZN5test7L2b3E
  const B b1 = B();
  const B &b2 = B();
  const int b3 = B().n;

  // CHECK-NOT: @_ZN5test7L2c1E
  // PRE17: call void @llvm.memset{{.*}} @_ZN5test7L2c1E
  // CHECK-NOT: @_ZN5test7L2c1E
  // CHECK: @_ZN5test7L2c2E
  // CHECK-NOT: @_ZN5test7L2c3E
  // PRE17: @_ZN5test7L2c4E
  const C c1 = C();
  const C c2 = static_cast<const C&>(C());
  const int c3 = C().n;
  const int c4 = C(C()).n;

  // CHECK-NOT: @_ZN5test7L1dE
  const D d = D();

  // CHECK: store {{.*}} @_ZN5test71eE
  int f(), e = f();
}


// At the end of the file, we check that y is initialized before z.

// CHECK:      define internal void [[TEST1_Z_INIT:@.*]]()
// CHECK:        load i32, ptr @_ZN5test1L1yE
// CHECK-NEXT:   xor
// CHECK-NEXT:   store i32 {{.*}}, ptr @_ZN5test1L1zE
// CHECK:      define internal void [[TEST1_Y_INIT:@.*]]()
// CHECK:        load i32, ptr @_ZN5test1L1xE
// CHECK-NEXT:   sub
// CHECK-NEXT:   store i32 {{.*}}, ptr @_ZN5test1L1yE

// CHECK: define internal void @_GLOBAL__sub_I_global_init.cpp() #{{[0-9]+}} section "__TEXT,__StaticInit,regular,pure_instructions" {
// CHECK:   call void [[TEST1_Y_INIT]]
// CHECK:   call void [[TEST1_Z_INIT]]

// this should be nounwind
// CHECK-NOEXC: define internal void @_GLOBAL__sub_I_global_init.cpp() [[NUW:#[0-9]+]] section "__TEXT,__StaticInit,regular,pure_instructions" {
// CHECK-NOEXC: attributes [[NUW]] = { noinline nounwind{{.*}} }

// Make sure we mark global initializers with the no-builtins attribute.
// CHECK-NOBUILTIN: define internal void @_GLOBAL__sub_I_global_init.cpp() [[NUW:#[0-9]+]] section "__TEXT,__StaticInit,regular,pure_instructions" {
// CHECK-NOBUILTIN: attributes [[NUW]] = { noinline nounwind{{.*}}"no-builtins"{{.*}} }

// PR21811: attach the appropriate attribute to the global init function
// CHECK-FP: define internal void @_GLOBAL__sub_I_global_init.cpp() [[NUX:#[0-9]+]] section "__TEXT,__StaticInit,regular,pure_instructions" {
// CHECK-FP: attributes [[NUX]] = { noinline nounwind {{.*}}"frame-pointer"="non-leaf"{{.*}} }
