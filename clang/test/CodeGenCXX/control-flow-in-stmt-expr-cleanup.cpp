// RUN: %clang_cc1 --std=c++20 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s

// Context: GH63818
struct Printy {
  Printy(const char *);
  Printy();
  ~Printy();
};

struct Printies {
  const Printy &a;
  const Printy &b;
  ~Printies() {}
};

bool foo();

// ====================================
// Init with lifetime extensions
// ====================================
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

// ====================================
// Break in stmt-expr
// ====================================
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

// =============================================
// Initialisation without lifetime-extension
// =============================================
void test_init_with_no_ref_binding() {
  // CHECK: define dso_local void @_Z29test_init_with_no_ref_bindingv()
  struct PrintiesCopy {
    Printy a;
    Printy b;
    Printy c;
  };
  PrintiesCopy ps(Printy(), ({
                    if (foo()) {
                      // CHECK:       if.then:
                      // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
                      // CHECK-NEXT:    br label %return
                      return;
                    }
                    Printy();
                  }),
                  ({
                    if (foo()) {
                      // CHECK:       if.then2:
                      // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
                      // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
                      // CHECK-NEXT:    br label %return
                      return;
                    }
                    Printy();
                  }));
}

// ====================================
// Array init
// ====================================
void test_array_init() {
  // CHECK: define dso_local void @_Z15test_array_initv()
  Printy arr[] = {
    Printy(), 
    ({
      if (foo()) {
        // CHECK:       if.then:
        // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
        // CHECK-NEXT:    br label %return
        return;
      }
      Printy();
    }),
    ({
      if (foo()) {
        return;
        // CHECK:       if.then3:
        // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
        // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
        // CHECK-NEXT:    br label %return
      }
      Printy();
    })};
  return;
}

void new_array_init() {
  // CHECK: define dso_local void @_Z14new_array_initv()
  Printy *a = new Printy[]("1", ({
                             if (foo()) {
                               return;
                               // CHECK: if.then:
                               // CHECK-NEXT: call void @_ZN6PrintyD1Ev
                             }
                             "2";
                           }));
  delete[] a;
}

// ====================================
// Arrays as sub-objects
// ====================================
void arrays_as_subobjects() {
  // CHECK: define dso_local void @_Z20arrays_as_subobjectsv()
  struct S {
    Printy arr1[2];
    Printy arr2[2];
    Printy p;
  };
  S s{{Printy(), Printy()},
      {Printy(), ({
         if (foo()) {
          /** One dtor followed by an array destruction **/
          // CHECK: if.then:
          // CHECK:   call void @_ZN6PrintyD1Ev
          // CHECK:   br label %arraydestroy.body

          // CHECK: arraydestroy.body:
          // CHECK:   call void @_ZN6PrintyD1Ev

          // CHECK: arraydestroy.done{{.*}}:
          // CHECK:   br label %return
           return;
         }
         Printy();
       })},
      ({
        if (foo()) {
          /** Two array destructions **/
          //CHECK: if.then{{.*}}:
          //CHECK:   br label %arraydestroy.body{{.*}}

          //CHECK: arraydestroy.body{{.*}}:
          //CHECK:   call void @_ZN6PrintyD1Ev

          //CHECK: arraydestroy.done{{.*}}:
          //CHECK:   br label %arraydestroy.body{{.*}}

          //CHECK: arraydestroy.body{{.*}}:
          //CHECK:   call void @_ZN6PrintyD1Ev

          //CHECK: arraydestroy.done{{.*}}:
          //CHECK:   br label %return
          return;
        }
        Printy();
      })};
}

// ====================================
// Lambda capture initialisation
// ====================================
void lambda_init() {
  // CHECK: define dso_local void @_Z11lambda_initv()
  auto S = [a = Printy(), b = ({
                            if (foo()) {
                              return;
                              // CHECK: if.then:
                              // CHECK:   call void @_ZN6PrintyD1Ev
                              // CHECK:   br label %return
                            }
                            Printy();
                          })]() {};
}
