// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 -O3 | FileCheck %s

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

// CHECK-LABEL: define i32 @_Z12getCtorCallsv()
int getCtorCalls() {
  // CHECK: ret i32 5
  return ctorcalls;
}

// CHECK-LABEL: define i32 @_Z12getDtorCallsv()
int getDtorCalls() {
  // CHECK: ret i32 5
  return dtorcalls;
}

// CHECK-LABEL: define zeroext i1 @_Z7successv()
bool success() {
  // CHECK: ret i1 true
  return ctorcalls == dtorcalls;
}
