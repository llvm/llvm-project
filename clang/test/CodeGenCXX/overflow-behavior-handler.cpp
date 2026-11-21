// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -emit-llvm -o - %s | FileCheck %s

// A lambda is its own function: it cannot reach the enclosing function's
// handler, but may define its own.
int f(int __ob_trap a) {
  auto no_handler = [=]() -> int { return a + 1; };
  auto own_handler = [=]() -> int {
    return a * 2;
  __overflow:
    return -9;
  };
  return no_handler() + own_handler();
__overflow:
  return -1;
}

// CHECK-LABEL: define {{.*}} @_Z1fU8ObtTrap_i(
// CHECK-NOT: br label %__overflow
// CHECK: ret i32

// CHECK-LABEL: define internal {{.*}} @"_ZZ1fU8ObtTrap_iENK3$_0clEv"(
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK: call void @llvm.ubsantrap

// CHECK-LABEL: define internal {{.*}} @"_ZZ1fU8ObtTrap_iENK3$_1clEv"(
// CHECK: @llvm.smul.with.overflow.i32
// CHECK: br label %__overflow

template <class T> T twice(T x) {
  return x + x;
__overflow:
  return T(0);
}

int g(int __ob_trap a) { return twice(a); }

// CHECK-LABEL: define linkonce_odr {{.*}} @_Z5twiceIU8ObtTrap_iET_S1_(
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK: br label %__overflow

struct D { D(); ~D(); int get(); };

// Leaving a scope destroys its objects before the handler; objects still in
// scope at the handler are destroyed when it returns.
// CHECK-LABEL: define {{.*}} @_Z10dtor_scopeU8ObtTrap_i(
// CHECK: call void @_ZN1DC1Ev(ptr {{.*}} %outer)
// CHECK: call void @_ZN1DC1Ev(ptr {{.*}} %inner)
// CHECK: @llvm.smul.with.overflow.i32
// CHECK: br i1 %{{.*}}, label %[[H:[a-z.0-9]+]], label %{{.*}}
// CHECK: [[H]]:
// CHECK-NEXT: store i32 [[DEST:[0-9]+]], ptr %cleanup.dest.slot
// CHECK: call void @_ZN1DD1Ev(ptr {{.*}} %inner)
// CHECK-NOT: call void @_ZN1DD1Ev
// CHECK: i32 [[DEST]], label %__overflow
// CHECK: __overflow:
// CHECK-NOT: call void @_ZN1DD1Ev
// CHECK: call void @_ZN1DD1Ev(ptr {{.*}} %outer)
// CHECK: ret i32
int dtor_scope(int __ob_trap n) {
  D outer;
  {
    D inner;
    n *= 2;
  }
  return n;
__overflow:
  return -1;
}

// A temporary in the overflowing full-expression is destroyed before the
// handler.
// CHECK-LABEL: define {{.*}} @_Z14dtor_temporaryU8ObtTrap_i(
// CHECK: call void @_ZN1DC1Ev(ptr {{.*}} %[[TMP:ref.tmp[0-9]*]])
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK: br i1 %{{.*}}, label %[[H:[a-z.0-9]+]], label %{{.*}}
// CHECK: [[H]]:
// CHECK-NEXT: store i32 [[DEST:[0-9]+]], ptr %cleanup.dest.slot
// CHECK: call void @_ZN1DD1Ev(ptr {{.*}} %[[TMP]])
// CHECK: i32 [[DEST]], label %__overflow
int dtor_temporary(int __ob_trap n) {
  n = D().get() + n;
  return n;
__overflow:
  return -1;
}

// A member function of a local class is its own function: it uses its own
// handler and traps without one, even when the enclosing function has one.
int local_class(int __ob_trap x) {
  struct S {
    int none(int __ob_trap a) { return a + 1; }
    int own(int __ob_trap a) {
      return a * 2;
    __overflow:
      return -1;
    }
  };
  S s;
  return s.none(x) + s.own(x);
__overflow:
  return -1;
}

// CHECK-LABEL: define internal {{.*}} @_ZZ11local_classU8ObtTrap_iEN1S4noneES_(
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK-NOT: br label %__overflow
// CHECK: call void @llvm.ubsantrap

// CHECK-LABEL: define internal {{.*}} @_ZZ11local_classU8ObtTrap_iEN1S3ownES_(
// CHECK: @llvm.smul.with.overflow.i32
// CHECK: br label %__overflow
