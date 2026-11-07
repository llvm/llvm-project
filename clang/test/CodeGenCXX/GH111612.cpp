// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm-only %s

// A memcpy batch of trivially copyable members that follows a noreturn member
// or base operator= is unreachable and must be dropped, not emitted.

struct S {
  __attribute__((noreturn)) void operator=(S);
};

struct T {
  void operator=(T);
};

struct C {
  S b;
  float c;
  _Complex float e;
};
void f1(C &x, C &y) { x = y; }
// CHECK-LABEL: define {{.*}} ptr @_ZN1CaSERKS_(
// CHECK:         call void @_ZN1SaSES_(
// CHECK-NEXT:    unreachable
// CHECK-NEXT:  }

// The batch before the call is still emitted.
struct C2 {
  float c;
  _Complex float e;
  S b;
};
void f2(C2 &x, C2 &y) { x = y; }
// CHECK-LABEL: define {{.*}} ptr @_ZN2C2aSERKS_(
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %{{.*}}, i64 12, i1 false)
// CHECK:         call void @_ZN1SaSES_(
// CHECK-NEXT:    unreachable
// CHECK-NEXT:  }

// The batch is flushed by a later member copy instead of the return.
struct C3 {
  S b;
  float c;
  float d;
  T t;
};
void f3(C3 &x, C3 &y) { x = y; }
// CHECK-LABEL: define {{.*}} ptr @_ZN2C3aSERKS_(
// CHECK:         call void @_ZN1SaSES_(
// CHECK-NEXT:    unreachable
// CHECK-NEXT:  }

struct D : S {
  float c;
  _Complex float e;
};
void f4(D &x, D &y) { x = y; }
// CHECK-LABEL: define {{.*}} ptr @_ZN1DaSERKS_(
// CHECK:         call void @_ZN1SaSES_(
// CHECK-NEXT:    unreachable
// CHECK-NEXT:  }

struct M {
  M &operator=(const M &);
  __attribute__((noreturn)) M &operator=(M &&);
};
struct CM {
  M b;
  float c;
  _Complex float e;
};
void f5(CM &x, CM &y) { x = static_cast<CM &&>(y); }
// CHECK-LABEL: define {{.*}} ptr @_ZN2CMaSEOS_(
// CHECK:         call {{.*}} @_ZN1MaSEOS_(
// CHECK-NEXT:    unreachable
// CHECK-NEXT:  }

// The loop exit is reachable, so the batch after an array member is emitted.
struct CA {
  S arr[2];
  float c;
  _Complex float e;
};
void f6(CA &x, CA &y) { x = y; }
// CHECK-LABEL: define {{.*}} ptr @_ZN2CAaSERKS_(
// CHECK:         call void @_ZN1SaSES_(
// CHECK-NEXT:    unreachable
// CHECK:       for.end:
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %{{.*}}, i64 12, i1 false)
// CHECK-NEXT:    ret ptr %this1

struct N {
  T b;
  float c;
  _Complex float e;
};
void f7(N &x, N &y) { x = y; }
// CHECK-LABEL: define {{.*}} ptr @_ZN1NaSERKS_(
// CHECK:         call void @_ZN1TaSES_(
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %{{.*}}, i64 12, i1 false)
// CHECK-NEXT:    ret ptr %this1

// Constructors keep an insertion point after a noreturn call.
struct SC {
  SC();
  __attribute__((noreturn)) SC(const SC &);
};
struct CC {
  SC b;
  float c;
  _Complex float e;
};
void f8(CC &x) { CC y(x); }
// CHECK-LABEL: define {{.*}} void @_ZN2CCC2ERKS_(
// CHECK:         call void @_ZN2SCC1ERKS_(
// CHECK-NEXT:    unreachable
