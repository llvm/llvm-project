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
// CHECK: define dso_local void @_Z3barv()
// CHECK-NOT: call void @_ZN6PrintyD1Ev
  Printies p2{
    Printy(), 
    ({
      if(foo()) {
        // CHECK: if.then:
        // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
        // CHECK-NEXT:    br label %return
        // CHECK-NOT: call void @_ZN6PrintyD1Ev
        return;
      }
      Printy();
    })};
  // CHECK:       if.end:
  // CHECK:         call void @_ZN8PrintiesD1Ev
  // CHECK:         call void @_ZN6PrintyD1Ev
  // CHECK:         call void @_ZN6PrintyD1Ev
  // CHECK-NEXT:    br label %return
  return;
  // CHECK-NOT: call void @_ZN6PrintyD1Ev
}


void test_break() {
// CHECK: define dso_local void @_Z10test_breakv()
// CHECK-NOT: call void @_ZN6PrintyD1Ev
  Printies p2{Printy(), ({
                for (;;) {
                  Printies p3{Printy(), ({
                                if(foo()) {
                                  break;
                                  // CHECK:       if.then:
                                  // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
                                  // CHECK-NEXT:    br label %for.end
                                  // CHECK-NOT:   call void @_ZN6PrintyD1Ev
                                }
                                Printy();
                              })};
                  // CHECK:         if.end:
                  // CHECK:           call void @_ZN6PrintyD1Ev
                  // CHECK:           call void @_ZN6PrintyD1Ev
                  // CHECK-NOT:     call void @_ZN6PrintyD1Ev
                }
                Printy();
              })};
  // CHECK:         for.end:
  // CHECK-COUNT-2:   call void @_ZN6PrintyD1Ev
  // CHECK-NEXT:      ret void
  // CHECK-NOT:     call void @_ZN6PrintyD1Ev
}

