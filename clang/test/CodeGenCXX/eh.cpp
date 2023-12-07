// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -triple x86_64-apple-macosx10.13.99 -std=c++11 -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=UNALIGNED %s
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -triple x86_64-apple-macosx10.14 -std=c++11 -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=ALIGNED %s

struct test1_D {
  double d;
} d1;

void test1() {
  throw d1;
}

// CHECK-LABEL:     define{{.*}} void @_Z5test1v()
// CHECK:       [[EXNOBJ:%.*]] = call ptr @__cxa_allocate_exception(i64 8)
// UNALIGNED-NEXT:  call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[EXNOBJ]], ptr align 8 @d1, i64 8, i1 false)
// ALIGNED-NEXT:  call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[EXNOBJ]], ptr align 8 @d1, i64 8, i1 false)
// CHECK-NEXT:  call void @__cxa_throw(ptr [[EXNOBJ]], ptr @_ZTI7test1_D, ptr null) [[NR:#[0-9]+]]
// CHECK-NEXT:  unreachable


struct test2_D {
  test2_D(const test2_D&o);
  test2_D();
  virtual void bar() { }
  int i; int j;
} d2;

void test2() {
  throw d2;
}

// CHECK-LABEL:     define{{.*}} void @_Z5test2v()
// CHECK:       [[EXNVAR:%.*]] = alloca ptr
// CHECK-NEXT:  [[SELECTORVAR:%.*]] = alloca i32
// CHECK-NEXT:  [[EXNOBJ:%.*]] = call ptr @__cxa_allocate_exception(i64 16)
// CHECK-NEXT:  invoke void @_ZN7test2_DC1ERKS_(ptr {{[^,]*}} [[EXNOBJ]], ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) @d2)
// CHECK-NEXT:     to label %[[CONT:.*]] unwind label %{{.*}}
//      :     [[CONT]]:   (can't check this in Release-Asserts builds)
// CHECK:       call void @__cxa_throw(ptr [[EXNOBJ]], ptr @_ZTI7test2_D, ptr null) [[NR]]
// CHECK-NEXT:  unreachable


struct test3_D {
  test3_D() { }
  test3_D(volatile test3_D&o);
  virtual void bar();
};

void test3() {
  throw (volatile test3_D *)0;
}

// CHECK-LABEL:     define{{.*}} void @_Z5test3v()
// CHECK:       [[EXNOBJ:%.*]] = call ptr @__cxa_allocate_exception(i64 8)
// CHECK-NEXT:  store ptr null, ptr [[EXNOBJ]]
// CHECK-NEXT:  call void @__cxa_throw(ptr [[EXNOBJ]], ptr @_ZTIPV7test3_D, ptr null) [[NR]]
// CHECK-NEXT:  unreachable


void test4() {
  throw;
}

// CHECK-LABEL:     define{{.*}} void @_Z5test4v()
// CHECK:        call void @__cxa_rethrow() [[NR]]
// CHECK-NEXT:   unreachable

namespace test5 {
  struct A {
    A();
    A(const A&);
    ~A();
  };

  void test() {
    try { throw A(); } catch (A &x) {}
  }
// CHECK-LABEL:      define{{.*}} void @_ZN5test54testEv()
// CHECK:      [[EXNOBJ:%.*]] = call ptr @__cxa_allocate_exception(i64 1)
// CHECK: invoke void @_ZN5test51AC1Ev(ptr {{[^,]*}} [[EXNOBJ]])
// CHECK:      invoke void @__cxa_throw(ptr [[EXNOBJ]], ptr @_ZTIN5test51AE, ptr @_ZN5test51AD1Ev) [[NR]]
// CHECK-NEXT:   to label {{%.*}} unwind label %[[HANDLER:[^ ]*]]
//      :    [[HANDLER]]:  (can't check this in Release-Asserts builds)
// CHECK:      {{%.*}} = call i32 @llvm.eh.typeid.for(ptr @_ZTIN5test51AE)
}

namespace test6 {
  template <class T> struct allocator {
    ~allocator() throw() { }
  };

  void foo() {
    allocator<int> a;
  }
}

// PR7127
namespace test7 {
// CHECK-LABEL:      define{{.*}} i32 @_ZN5test73fooEv() 
// CHECK-SAME:  personality ptr @__gxx_personality_v0
  int foo() {
// CHECK:      [[CAUGHTEXNVAR:%.*]] = alloca ptr
// CHECK-NEXT: [[SELECTORVAR:%.*]] = alloca i32
// CHECK-NEXT: [[INTCATCHVAR:%.*]] = alloca i32
    try {
      try {
// CHECK-NEXT: [[EXNALLOC:%.*]] = call ptr @__cxa_allocate_exception
// CHECK-NEXT: store i32 1, ptr
// CHECK-NEXT: invoke void @__cxa_throw(ptr [[EXNALLOC]], ptr @_ZTIi, ptr null
        throw 1;
      }

// CHECK:      [[CAUGHTVAL:%.*]] = landingpad { ptr, i32 }
// CHECK-NEXT:   catch ptr @_ZTIi
// CHECK-NEXT:   catch ptr null
// CHECK-NEXT: [[CAUGHTEXN:%.*]] = extractvalue { ptr, i32 } [[CAUGHTVAL]], 0
// CHECK-NEXT: store ptr [[CAUGHTEXN]], ptr [[CAUGHTEXNVAR]]
// CHECK-NEXT: [[SELECTOR:%.*]] = extractvalue { ptr, i32 } [[CAUGHTVAL]], 1
// CHECK-NEXT: store i32 [[SELECTOR]], ptr [[SELECTORVAR]]
// CHECK-NEXT: br label
// CHECK:      [[SELECTOR:%.*]] = load i32, ptr [[SELECTORVAR]]
// CHECK-NEXT: [[T0:%.*]] = call i32 @llvm.eh.typeid.for(ptr @_ZTIi)
// CHECK-NEXT: icmp eq i32 [[SELECTOR]], [[T0]]
// CHECK-NEXT: br i1
// CHECK:      [[T0:%.*]] = load ptr, ptr [[CAUGHTEXNVAR]]
// CHECK-NEXT: [[T1:%.*]] = call ptr @__cxa_begin_catch(ptr [[T0]])
// CHECK-NEXT: [[T3:%.*]] = load i32, ptr [[T1]]
// CHECK-NEXT: store i32 [[T3]], ptr {{%.*}}, align 4
// CHECK-NEXT: invoke void @__cxa_rethrow
      catch (int) {
        throw;
      }
    }
// CHECK:      [[CAUGHTVAL:%.*]] = landingpad { ptr, i32 }
// CHECK-NEXT:   catch ptr null
// CHECK-NEXT: [[CAUGHTEXN:%.*]] = extractvalue { ptr, i32 } [[CAUGHTVAL]], 0
// CHECK-NEXT: store ptr [[CAUGHTEXN]], ptr [[CAUGHTEXNVAR]]
// CHECK-NEXT: [[SELECTOR:%.*]] = extractvalue { ptr, i32 } [[CAUGHTVAL]], 1
// CHECK-NEXT: store i32 [[SELECTOR]], ptr [[SELECTORVAR]]
// CHECK-NEXT: call void @__cxa_end_catch()
// CHECK-NEXT: br label
// CHECK:      load ptr, ptr [[CAUGHTEXNVAR]]
// CHECK-NEXT: call ptr @__cxa_begin_catch
// CHECK-NEXT: call void @__cxa_end_catch
    catch (...) {
    }
// CHECK:      ret i32 0
    return 0;
  }
}

// Ordering of destructors in a catch handler.
namespace test8 {
  struct A { A(const A&); ~A(); };
  void bar();

  // CHECK-LABEL: define{{.*}} void @_ZN5test83fooEv()
  void foo() {
    try {
      // CHECK:      invoke void @_ZN5test83barEv()
      bar();
    } catch (A a) {
      // CHECK:      call ptr @__cxa_get_exception_ptr
      // CHECK-NEXT: invoke void @_ZN5test81AC1ERKS0_(
      // CHECK:      call ptr @__cxa_begin_catch
      // CHECK-NEXT: call void @_ZN5test81AD1Ev(
      // CHECK:      call void @__cxa_end_catch()
      // CHECK:      ret void
    }
  }
}

// Constructor function-try-block must rethrow on fallthrough.
namespace test9 {
  void opaque();

  struct A { A(); };


  // CHECK-LABEL: define{{.*}} void @_ZN5test91AC2Ev(ptr {{[^,]*}} %this) unnamed_addr
  // CHECK-SAME:  personality ptr @__gxx_personality_v0
  A::A() try {
  // CHECK:      invoke void @_ZN5test96opaqueEv()
    opaque();
  } catch (int x) {
  // CHECK:      landingpad { ptr, i32 }
  // CHECK-NEXT:   catch ptr @_ZTIi

  // CHECK:      call ptr @__cxa_begin_catch
  // CHECK:      invoke void @_ZN5test96opaqueEv()
  // CHECK:      invoke void @__cxa_rethrow()

  // CHECK-LABEL:      define{{.*}} void @_ZN5test91AC1Ev(ptr {{[^,]*}} %this) unnamed_addr
  // CHECK:      call void @_ZN5test91AC2Ev
  // CHECK-NEXT: ret void
    opaque();
  }
}

// __cxa_end_catch can throw for some kinds of caught exceptions.
namespace test10 {
  void opaque();

  struct A { ~A(); };
  struct B { int x; };

  // CHECK-LABEL: define{{.*}} void @_ZN6test103fooEv()
  void foo() {
    A a; // force a cleanup context

    try {
    // CHECK:      invoke void @_ZN6test106opaqueEv()
      opaque();
    } catch (int i) {
    // CHECK:      call ptr @__cxa_begin_catch
    // CHECK-NEXT: load i32, ptr
    // CHECK-NEXT: store i32
    // CHECK-NEXT: call void @__cxa_end_catch() [[NUW:#[0-9]+]]
    } catch (B a) {
    // CHECK:      call ptr @__cxa_begin_catch
    // CHECK-NEXT: call void @llvm.memcpy
    // CHECK-NEXT: invoke void @__cxa_end_catch()
    } catch (...) {
    // CHECK:      call ptr @__cxa_begin_catch
    // CHECK-NEXT: invoke void @__cxa_end_catch()
    }

    // CHECK: call void @_ZN6test101AD1Ev(
  }
}

// __cxa_begin_catch returns pointers by value, even when catching by reference
namespace test11 {
  void opaque();

  // CHECK-LABEL: define{{.*}} void @_ZN6test113fooEv()
  void foo() {
    try {
      // CHECK:      invoke void @_ZN6test116opaqueEv()
      opaque();
    } catch (int**&p) {
      // CHECK:      [[EXN:%.*]] = load ptr, ptr
      // CHECK-NEXT: call ptr @__cxa_begin_catch(ptr [[EXN]]) [[NUW]]
      // CHECK-NEXT: [[ADJ1:%.*]] = getelementptr i8, ptr [[EXN]], i32 32
      // CHECK-NEXT: store ptr [[ADJ1]], ptr [[P:%.*]]
      // CHECK-NEXT: call void @__cxa_end_catch() [[NUW]]
    }
  }

  struct A {};

  // CHECK-LABEL: define{{.*}} void @_ZN6test113barEv()
  void bar() {
    try {
      // CHECK:      [[EXNSLOT:%.*]] = alloca ptr
      // CHECK-NEXT: [[SELECTORSLOT:%.*]] = alloca i32
      // CHECK-NEXT: [[P:%.*]] = alloca ptr,
      // CHECK-NEXT: [[TMP:%.*]] = alloca ptr
      // CHECK-NEXT: invoke void @_ZN6test116opaqueEv()
      opaque();
    } catch (A*&p) {
      // CHECK:      [[EXN:%.*]] = load ptr, ptr [[EXNSLOT]]
      // CHECK-NEXT: [[ADJ1:%.*]] = call ptr @__cxa_begin_catch(ptr [[EXN]]) [[NUW]]
      // CHECK-NEXT: store ptr [[ADJ1]], ptr [[TMP]]
      // CHECK-NEXT: store ptr [[TMP]], ptr [[P]]
      // CHECK-NEXT: call void @__cxa_end_catch() [[NUW]]
    }
  }
}

// PR7686
namespace test12 {
  struct A { ~A() noexcept(false); };
  bool opaque(const A&);

  // CHECK-LABEL: define{{.*}} void @_ZN6test124testEv()
  void test() {
    // CHECK: [[X:%.*]] = alloca [[A:%.*]],
    // CHECK: [[EHCLEANUPDEST:%.*]] = alloca i32
    // CHECK: [[Y:%.*]] = alloca [[A:%.*]]
    // CHECK: [[Z:%.*]] = alloca [[A]]
    // CHECK: [[CLEANUPDEST:%.*]] = alloca i32

    A x;
    // CHECK: invoke noundef zeroext i1 @_ZN6test126opaqueERKNS_1AE(
    if (opaque(x)) {
      A y;
      A z;

      // CHECK: invoke void @_ZN6test121AD1Ev(ptr {{[^,]*}} [[Z]])
      // CHECK: invoke void @_ZN6test121AD1Ev(ptr {{[^,]*}} [[Y]])
      // CHECK-NOT: switch
      goto success;
    }

  success:
    bool _ = true;

    // CHECK: call void @_ZN6test121AD1Ev(ptr {{[^,]*}} [[X]])
    // CHECK-NEXT: ret void
  }
}

// Reduced from some TableGen code that was causing a self-host crash.
namespace test13 {
  struct A { ~A(); };

  void test0(int x) {
    try {
      switch (x) {
      case 0:
        break;
      case 1:{
        A a;
        break;
      }
      default:
        return;
      }
      return;
    } catch (int x) {
    }
    return;
  }

  void test1(int x) {
    A y;
    try {
      switch (x) {
      default: break;
      }
    } catch (int x) {}
  }
}

namespace test14 {
  struct A { ~A(); };
  struct B { ~B(); };

  B b();
  void opaque();

  void foo() {
    A a;
    try {
      B str = b();
      opaque();
    } catch (int x) {
    }
  }
}

// JumpDests shouldn't get confused by scopes that aren't normal cleanups.
namespace test15 {
  struct A { ~A(); };

  bool opaque(int);

  // CHECK-LABEL: define{{.*}} void @_ZN6test153fooEv()
  void foo() {
    A a;

    try {
      // CHECK:      [[X:%.*]] = alloca i32
      // CHECK:      store i32 10, ptr [[X]]
      // CHECK-NEXT: br label
      //   -> while.cond
      int x = 10;

      while (true) {
        // CHECK:      load i32, ptr [[X]]
        // CHECK-NEXT: [[COND:%.*]] = invoke noundef zeroext i1 @_ZN6test156opaqueEi
        // CHECK:      br i1 [[COND]]
        if (opaque(x))
        // CHECK:      br label
          break;

        // CHECK:      br label
      }
      // CHECK:      br label
    } catch (int x) { }

    // CHECK: call void @_ZN6test151AD1Ev
  }
}

namespace test16 {
  struct A { A(); ~A() noexcept(false); };
  struct B { int x; B(const A &); ~B() noexcept(false); };
  void foo();
  bool cond();

  // CHECK-LABEL: define{{.*}} void @_ZN6test163barEv()
  void bar() {
    // CHECK:      [[EXN_SAVE:%.*]] = alloca ptr
    // CHECK-NEXT: [[EXN_ACTIVE:%.*]] = alloca i1
    // CHECK-NEXT: [[TEMP:%.*]] = alloca [[A:%.*]],
    // CHECK-NEXT: [[EXNSLOT:%.*]] = alloca ptr
    // CHECK-NEXT: [[SELECTORSLOT:%.*]] = alloca i32
    // CHECK-NEXT: [[TEMP_ACTIVE:%.*]] = alloca i1

    cond() ? throw B(A()) : foo();

    // CHECK-NEXT: [[COND:%.*]] = call noundef zeroext i1 @_ZN6test164condEv()
    // CHECK-NEXT: store i1 false, ptr [[EXN_ACTIVE]]
    // CHECK-NEXT: store i1 false, ptr [[TEMP_ACTIVE]]
    // CHECK-NEXT: br i1 [[COND]],

    // CHECK:      [[EXN:%.*]] = call ptr @__cxa_allocate_exception(i64 4)
    // CHECK-NEXT: store ptr [[EXN]], ptr [[EXN_SAVE]]
    // CHECK-NEXT: store i1 true, ptr [[EXN_ACTIVE]]
    // CHECK-NEXT: invoke void @_ZN6test161AC1Ev(ptr {{[^,]*}} [[TEMP]])
    // CHECK:      store i1 true, ptr [[TEMP_ACTIVE]]
    // CHECK-NEXT: invoke void @_ZN6test161BC1ERKNS_1AE(ptr {{[^,]*}} [[EXN]], ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[TEMP]])
    // CHECK:      store i1 false, ptr [[EXN_ACTIVE]]
    // CHECK-NEXT: invoke void @__cxa_throw(ptr [[EXN]],

    // CHECK:      invoke void @_ZN6test163fooEv()
    // CHECK:      br label

    // CHECK:      invoke void @_ZN6test161AD1Ev(ptr {{[^,]*}} [[TEMP]])
    // CHECK:      ret void

    // CHECK:      [[T0:%.*]] = load i1, ptr [[EXN_ACTIVE]]
    // CHECK-NEXT: br i1 [[T0]]
    // CHECK:      [[T1:%.*]] = load ptr, ptr [[EXN_SAVE]]
    // CHECK-NEXT: call void @__cxa_free_exception(ptr [[T1]])
    // CHECK-NEXT: br label
  }
}

namespace test17 {
class BaseException {
private:
  int a[4];
public:
  BaseException() {};
};

class DerivedException: public BaseException {
};

int foo() {
  throw DerivedException();
  // The alignment passed to memset is 16 on Darwin.

  // CHECK: [[T0:%.*]] = call ptr @__cxa_allocate_exception(i64 16)
  // UNALIGNED-NEXT: call void @llvm.memset.p0.i64(ptr align 8 [[T0]], i8 0, i64 16, i1 false)
  // ALIGNED-NEXT: call void @llvm.memset.p0.i64(ptr align 16 [[T0]], i8 0, i64 16, i1 false)
}
}

// CHECK: attributes [[NUW]] = { nounwind }
// CHECK: attributes [[NR]] = { noreturn }
