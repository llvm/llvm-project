// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s

// Context: GH63818
struct Printy {
  ~Printy() { }
};

struct Printies {
  const Printy &a;
  const Printy &b;
  ~Printies() {}
};

bool foo();

void bar() {
  Printies p2{
    // CHECK: store ptr %ref.tmp
    Printy(), 
    ({
      if(foo()) {
        // CHECK-LABEL: if.then:
        // CHECK-NEXT: call void @_ZN6PrintyD1Ev
        // CHECK-NEXT: br label %return
        return;
      }
      // CHECK-LABEL: if.end:
      // CHECK-NEXT: store ptr %ref.tmp1
      Printy();
    })};
  // CHECK-NEXT: call void @_ZN8PrintiesD1Ev
  // CHECK-NEXT: call void @_ZN6PrintyD1Ev
  // CHECK-NEXT: call void @_ZN6PrintyD1Ev
  // CHECK-NEXT: br label %return
  return;
}

