// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 -O2 -disable-llvm-passes | FileCheck %s --check-prefixes=CHECK,CHECK-NOOPT
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 -O2 | FileCheck %s --check-prefixes=CHECK,CHECK-OPT
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=amdgcn-amd-amdhsa -O2 | FileCheck %s --check-prefixes=CHECK,CHECK-OPT

namespace {

static int ctorcalls;
static int dtorcalls;
  
struct A {
  A() : i(0) { ctorcalls++; }
  ~A() { dtorcalls++; }
  int i;
  
  friend const A& operator<<(const A& a, int n) {
    return a;
  }
};

void g(int) { }
void g(const A&) { }

void f1(bool b) {
  g(b ? A().i : 0);
  g(b || A().i);
  g(b && A().i);
  g(b ? A() << 1 : A() << 2);
}

struct Checker {
  Checker() {
    f1(true);
    f1(false);
  }
};

Checker c;

}

// CHECK-OPT-LABEL: define{{.*}} i32 @_Z12getCtorCallsv()
int getCtorCalls() {
  // CHECK-OPT: ret i32 5
  return ctorcalls;
}

// CHECK-OPT-LABEL: define{{.*}} i32 @_Z12getDtorCallsv()
int getDtorCalls() {
  // CHECK-OPT: ret i32 5
  return dtorcalls;
}

// CHECK-OPT-LABEL: define{{.*}} zeroext i1 @_Z7successv()
bool success() {
  // CHECK-OPT: ret i1 true
  return ctorcalls == dtorcalls;
}

struct X { ~X(); int f(); };
int g(int, int, int);
// CHECK-LABEL: @_Z16lifetime_nontriv
int lifetime_nontriv(bool cond) {
  // CHECK-NOOPT: store i1 false,
  // CHECK-NOOPT: store i1 false,
  // CHECK-NOOPT: store i1 false,
  // CHECK-NOOPT: store i1 false,
  // CHECK-NOOPT: store i1 false,
  // CHECK-NOOPT: store i1 false,
  // CHECK-NOOPT: br i1
  //
  // CHECK-NOOPT: call void @llvm.lifetime.start
  // CHECK-NOOPT: store i1 true,
  // CHECK-NOOPT: store i1 true,
  // CHECK-NOOPT: call i32 @_ZN1X1fEv(
  // CHECK-NOOPT: call void @llvm.lifetime.start
  // CHECK-NOOPT: store i1 true,
  // CHECK-NOOPT: store i1 true,
  // CHECK-NOOPT: call i32 @_ZN1X1fEv(
  // CHECK-NOOPT: call void @llvm.lifetime.start
  // CHECK-NOOPT: store i1 true,
  // CHECK-NOOPT: store i1 true,
  // CHECK-NOOPT: call i32 @_ZN1X1fEv(
  // CHECK-NOOPT: call i32 @_Z1giii(
  // CHECK-NOOPT: br label
  //
  // CHECK-NOOPT: call i32 @_Z1giii(i32 1, i32 2, i32 3)
  // CHECK-NOOPT: br label
  //
  // CHECK-NOOPT: load i1,
  // CHECK-NOOPT: br i1
  // CHECK-NOOPT: call void @_ZN1XD1Ev(
  // CHECK-NOOPT: br label
  //
  // CHECK-NOOPT: load i1,
  // CHECK-NOOPT: br i1
  // CHECK-NOOPT: call void @llvm.lifetime.end
  // CHECK-NOOPT: br label
  //
  // CHECK-NOOPT: load i1,
  // CHECK-NOOPT: br i1
  // CHECK-NOOPT: call void @_ZN1XD1Ev(
  // CHECK-NOOPT: br label
  //
  // CHECK-NOOPT: load i1,
  // CHECK-NOOPT: br i1
  // CHECK-NOOPT: call void @llvm.lifetime.end
  // CHECK-NOOPT: br label
  //
  // CHECK-NOOPT: load i1,
  // CHECK-NOOPT: br i1
  // CHECK-NOOPT: call void @_ZN1XD1Ev(
  // CHECK-NOOPT: br label
  //
  // CHECK-NOOPT: load i1,
  // CHECK-NOOPT: br i1
  // CHECK-NOOPT: call void @llvm.lifetime.end
  // CHECK-NOOPT: br label
  //
  // CHECK-NOOPT: ret

  // CHECK-OPT: br i1
  //
  // CHECK-OPT: call void @llvm.lifetime.start
  // CHECK-OPT: call i32 @_ZN1X1fEv(
  // CHECK-OPT: call void @llvm.lifetime.start
  // CHECK-OPT: call i32 @_ZN1X1fEv(
  // CHECK-OPT: call void @llvm.lifetime.start
  // CHECK-OPT: call i32 @_ZN1X1fEv(
  // CHECK-OPT: call i32 @_Z1giii(
  // CHECK-OPT: call void @_ZN1XD1Ev(
  // CHECK-OPT: call void @llvm.lifetime.end
  // CHECK-OPT: call void @_ZN1XD1Ev(
  // CHECK-OPT: call void @llvm.lifetime.end
  // CHECK-OPT: call void @_ZN1XD1Ev(
  // CHECK-OPT: call void @llvm.lifetime.end
  // CHECK-OPT: br label
  return cond ? g(X().f(), X().f(), X().f()) : g(1, 2, 3);
}

struct Y { int f(); };
int g(int, int, int);
// CHECK-LABEL: @_Z13lifetime_triv
int lifetime_triv(bool cond) {
  // CHECK-NOOPT: call void @llvm.lifetime.start
  // CHECK-NOOPT: call void @llvm.lifetime.start
  // CHECK-NOOPT: call void @llvm.lifetime.start
  // CHECK-NOOPT: br i1
  //
  // CHECK-NOOPT: call i32 @_ZN1Y1fEv(
  // CHECK-NOOPT: call i32 @_ZN1Y1fEv(
  // CHECK-NOOPT: call i32 @_ZN1Y1fEv(
  // CHECK-NOOPT: call i32 @_Z1giii(
  // CHECK-NOOPT: br label
  //
  // CHECK-NOOPT: call i32 @_Z1giii(i32 1, i32 2, i32 3)
  // CHECK-NOOPT: br label
  //
  // CHECK-NOOPT: call void @llvm.lifetime.end
  // CHECK-NOOPT-NOT: br
  // CHECK-NOOPT: call void @llvm.lifetime.end
  // CHECK-NOOPT-NOT: br
  // CHECK-NOOPT: call void @llvm.lifetime.end
  //
  // CHECK-NOOPT: ret

  // FIXME: LLVM isn't smart enough to remove the lifetime markers from the
  // g(1, 2, 3) path here.

  // CHECK-OPT: call void @llvm.lifetime.start
  // CHECK-OPT: call void @llvm.lifetime.start
  // CHECK-OPT: call void @llvm.lifetime.start
  // CHECK-OPT: br i1
  //
  // CHECK-OPT: call i32 @_ZN1Y1fEv(
  // CHECK-OPT: call i32 @_ZN1Y1fEv(
  // CHECK-OPT: call i32 @_ZN1Y1fEv(
  // CHECK-OPT: call i32 @_Z1giii(
  // CHECK-OPT: br label
  //
  // CHECK-OPT: call void @llvm.lifetime.end
  // CHECK-OPT: call void @llvm.lifetime.end
  // CHECK-OPT: call void @llvm.lifetime.end
  return cond ? g(Y().f(), Y().f(), Y().f()) : g(1, 2, 3);
}

struct Z { ~Z() {} int f(); };
int g(int, int, int);
// CHECK-LABEL: @_Z22lifetime_nontriv_empty
int lifetime_nontriv_empty(bool cond) {
  // CHECK-OPT: br i1
  //
  // CHECK-OPT: call void @llvm.lifetime.start
  // CHECK-OPT: call i32 @_ZN1Z1fEv(
  // CHECK-OPT: call void @llvm.lifetime.start
  // CHECK-OPT: call i32 @_ZN1Z1fEv(
  // CHECK-OPT: call void @llvm.lifetime.start
  // CHECK-OPT: call i32 @_ZN1Z1fEv(
  // CHECK-OPT: call i32 @_Z1giii(
  // CHECK-OPT: call void @llvm.lifetime.end
  // CHECK-OPT: call void @llvm.lifetime.end
  // CHECK-OPT: call void @llvm.lifetime.end
  // CHECK-OPT: br label
  return cond ? g(Z().f(), Z().f(), Z().f()) : g(1, 2, 3);
}
