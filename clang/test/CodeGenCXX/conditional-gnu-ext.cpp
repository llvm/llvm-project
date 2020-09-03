// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// rdar: // 8353567
// pr7726

extern "C" int printf(...);

void test0() {
// CHECK: call i32 (...) @printf({{.*}}, i8* inttoptr (i64 3735928559 to i8*))
    printf("%p\n", (void *)0xdeadbeef ? : (void *)0xaaaaaa);
}

// rdar://8446940
namespace radar8446940 {
extern "C" void abort();

int main () {
  char x[1];
  char *y = x ? : 0;

  if (x != y)
    abort();
}
}

namespace radar8453812 {
extern "C" void abort();
_Complex int getComplex(_Complex int val) {
  static int count;
  if (count++)
    abort();
  return val;
}

_Complex int cmplx() {
    _Complex int cond;
    _Complex int rhs;

    return getComplex(1+2i) ? : rhs;
}

// lvalue test
void foo (int& lv) {
  ++lv;
}

int global = 1;

int &cond() {
  static int count;
  if (count++)
    abort();
  return global;
}


int main() {
  cmplx();
  int rhs = 10;
  foo (cond()? : rhs);
  return  global-2;
}
}

namespace test3 {
  struct A {
    A();
    A(const A&);
    ~A();
  };

  struct B {
    B();
    B(const B&);
    ~B();
    operator bool();
    operator A();
  };

  B test0(B &x) {
    // CHECK-LABEL:    define void @_ZN5test35test0ERNS_1BE(
    // CHECK:      [[X:%.*]] = alloca [[B:%.*]]*,
    // CHECK:      store [[B]]* {{%.*}}, [[B]]** [[X]]
    // CHECK-NEXT: [[T0:%.*]] = load [[B]]*, [[B]]** [[X]]
    // CHECK-NEXT: [[BOOL:%.*]] = call zeroext i1 @_ZN5test31BcvbEv([[B]]* [[T0]])
    // CHECK-NEXT: br i1 [[BOOL]]
    // CHECK:      call void @_ZN5test31BC1ERKS0_([[B]]* [[RESULT:%.*]], [[B]]* nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[T0]])
    // CHECK-NEXT: br label
    // CHECK:      call void @_ZN5test31BC1Ev([[B]]* [[RESULT]])
    // CHECK-NEXT: br label
    // CHECK:      ret void
    return x ?: B();
  }

  B test1() {
    // CHECK-LABEL:    define void @_ZN5test35test1Ev(
    // CHECK:      [[TEMP:%.*]] = alloca [[B]],
    // CHECK:      call  void @_ZN5test312test1_helperEv([[B]]* sret align 1 [[TEMP]])
    // CHECK-NEXT: [[BOOL:%.*]] = call zeroext i1 @_ZN5test31BcvbEv([[B]]* [[TEMP]])
    // CHECK-NEXT: br i1 [[BOOL]]
    // CHECK:      call void @_ZN5test31BC1ERKS0_([[B]]* [[RESULT:%.*]], [[B]]* nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[TEMP]])
    // CHECK-NEXT: br label
    // CHECK:      call void @_ZN5test31BC1Ev([[B]]* [[RESULT]])
    // CHECK-NEXT: br label
    // CHECK:      call void @_ZN5test31BD1Ev([[B]]* [[TEMP]])
    // CHECK-NEXT: ret void
    extern B test1_helper();
    return test1_helper() ?: B();
  }


  A test2(B &x) {
    // CHECK-LABEL:    define void @_ZN5test35test2ERNS_1BE(
    // CHECK:      [[X:%.*]] = alloca [[B]]*,
    // CHECK:      store [[B]]* {{%.*}}, [[B]]** [[X]]
    // CHECK-NEXT: [[T0:%.*]] = load [[B]]*, [[B]]** [[X]]
    // CHECK-NEXT: [[BOOL:%.*]] = call zeroext i1 @_ZN5test31BcvbEv([[B]]* [[T0]])
    // CHECK-NEXT: br i1 [[BOOL]]
    // CHECK:      call void @_ZN5test31BcvNS_1AEEv([[A:%.*]]* sret align 1 [[RESULT:%.*]], [[B]]* [[T0]])
    // CHECK-NEXT: br label
    // CHECK:      call void @_ZN5test31AC1Ev([[A]]* [[RESULT]])
    // CHECK-NEXT: br label
    // CHECK:      ret void
    return x ?: A();
  }

  A test3() {
    // CHECK-LABEL:    define void @_ZN5test35test3Ev(
    // CHECK:      [[TEMP:%.*]] = alloca [[B]],
    // CHECK:      call  void @_ZN5test312test3_helperEv([[B]]* sret align 1 [[TEMP]])
    // CHECK-NEXT: [[BOOL:%.*]] = call zeroext i1 @_ZN5test31BcvbEv([[B]]* [[TEMP]])
    // CHECK-NEXT: br i1 [[BOOL]]
    // CHECK:      call void @_ZN5test31BcvNS_1AEEv([[A]]* sret align 1 [[RESULT:%.*]], [[B]]* [[TEMP]])
    // CHECK-NEXT: br label
    // CHECK:      call void @_ZN5test31AC1Ev([[A]]* [[RESULT]])
    // CHECK-NEXT: br label
    // CHECK:      call void @_ZN5test31BD1Ev([[B]]* [[TEMP]])
    // CHECK-NEXT: ret void
    extern B test3_helper();
    return test3_helper() ?: A();
  }

}

namespace test4 {
  // Make sure this doesn't crash.
  void f() {
    const int a = 10, b = 20;
    const int *c = &(a ?: b);
  }
}
