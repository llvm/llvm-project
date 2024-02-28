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
                      // CHECK:       if.then{{.*}}:
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
    // CHECK:     entry:
    // CHECK:       %arrayinit.endOfInit = alloca ptr, align 8
    // CHECK-NEXT:  %arrayinit.begin = getelementptr inbounds
    // CHECK-NEXT:  store ptr %arrayinit.begin, ptr %arrayinit.endOfInit, align 8
    // CHECK-NEXT:  call void @_ZN6PrintyC1Ev
    // CHECK-NEXT:  %arrayinit.element = getelementptr inbounds %struct.Printy, ptr %arrayinit.begin, i64 1
    // CHECK-NEXT:  store ptr %arrayinit.element, ptr %arrayinit.endOfInit, align 8
    // CHECK-NEXT:  @_Z3foov()
    // CHECK-NEXT:  br i1 %call, label %if.then, label %if.end
    ({
      if (foo()) {
        // CHECK:       if.then:
        // CHECK-NEXT:    load ptr, ptr %arrayinit.endOfInit,
        // CHECK-NEXT:    %arraydestroy.isempty = icmp eq ptr %arrayinit.begin, {{.*}}
        // CHECK-NEXT:    br i1 %arraydestroy.isempty, label %arraydestroy.done{{.*}}, label %arraydestroy.body
        
        // CHECK:       arraydestroy.body:
        // CHECK-NEXT:    %arraydestroy.elementPast = phi ptr [ %0, %if.then ], [ %arraydestroy.element, %arraydestroy.body ]
        // CHECK-NEXT:    %arraydestroy.element = getelementptr inbounds %struct.Printy, ptr %arraydestroy.elementPast, i64 -1
        // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
        // CHECK-NEXT:    %arraydestroy.done = icmp eq ptr %arraydestroy.element, %arrayinit.begin
        // CHECK-NEXT:    br i1 %arraydestroy.done, label %arraydestroy.done{{.*}}, label %arraydestroy.body
        
        // CHECK:       arraydestroy.done{{.*}}:
        // CHECK-NEXT:    br label %return
        return;
      }
      Printy();
    }),
    ({
      if (foo()) {
        return;
        // CHECK:       if.then{{.*}}:
        // CHECK-NEXT:    load ptr, ptr %arrayinit.endOfInit
        // CHECK-NEXT:    %arraydestroy.isempty{{.*}} = icmp eq ptr %arrayinit.begin, {{.*}}
        // CHECK-NEXT:    br i1 %arraydestroy.isempty{{.*}}, label %arraydestroy.done{{.*}}, label %arraydestroy.body{{.*}}
        
        // CHECK:       arraydestroy.body{{.*}}:
        // CHECK-NEXT:    %arraydestroy.elementPast{{.*}} = phi ptr [ %1, %if.then{{.*}} ], [ %arraydestroy.element{{.*}}, %arraydestroy.body{{.*}} ]
        // CHECK-NEXT:    %arraydestroy.element{{.*}} = getelementptr inbounds %struct.Printy, ptr %arraydestroy.elementPast{{.*}}, i64 -1
        // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
        // CHECK-NEXT:    %arraydestroy.done{{.*}} = icmp eq ptr %arraydestroy.element{{.*}}, %arrayinit.begin
        // CHECK-NEXT:    br i1 %arraydestroy.done{{.*}}, label %arraydestroy.done{{.*}}, label %arraydestroy.body{{.*}}
        
        // CHECK:       arraydestroy.done{{.*}}:
        // CHECK-NEXT:    br label %return
      }
      Printy();
    })};
  return;
}

void new_array_init() {
  // CHECK: define dso_local void @_Z14new_array_initv()
  Printy *a = new Printy[]("1", 
    // CHECK:     entry:
    // CHECK:       %array.init.end = alloca ptr, align 8
    // CHECK:       store ptr %0, ptr %array.init.end, align 8
    // CHECK-NEXT:  store ptr %0, ptr %array.init.end, align 8
    // CHECK:       store ptr %array.exp.next, ptr %array.init.end, align 8
    ({
        if (foo()) {
          return;
          // CHECK:     if.then{{.*}}:
          // CHECK-NEXT:  load ptr, ptr %array.init.end, align 8
          // CHECK-NEXT:  %arraydestroy.isempty = icmp
          // CHECK-NEXT:  br i1 %arraydestroy.isempty, label %arraydestroy.done{{.*}}, label %arraydestroy.body
          
          // CHECK:       arraydestroy.body:
          // CHECK-NEXT:    %arraydestroy.elementPast = phi ptr [ %{{.*}}, %if.then ], [ %arraydestroy.element, %arraydestroy.body ]
          // CHECK-NEXT:    %arraydestroy.element = getelementptr inbounds %struct.Printy, ptr %arraydestroy.elementPast, i64 -1
          // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
          // CHECK-NEXT:    %arraydestroy.done = icmp eq ptr %arraydestroy.element, %0
          // CHECK-NEXT:    br i1 %arraydestroy.done, label %arraydestroy.done2, label %arraydestroy.body

          // CHECK:       arraydestroy.done{{.*}}:
          // CHECK-NEXT:    br label %delete.end
        }
        "2";
      }));                
    // CHECK:       delete.end{{.*}}:
    // CHECK-NEXT:    ret void
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
  };
  S s{{Printy(),
       // CHECK: %arrayinit.endOfInit = alloca ptr, align 8
       // CHECK: %arrayinit.endOfInit{{.*}} = alloca ptr, align 8
       // CHECK: store ptr %arrayinit.begin, ptr %arrayinit.endOfInit
       // CHECK: call void @_ZN6PrintyC1Ev
       // CHECK: store ptr %arrayinit.element, ptr %arrayinit.endOfInit
       ({
         if (foo()) {
           return;
           // CHECK:         if.then{{.*}}:
           // CHECK-NEXT:      load ptr, ptr %arrayinit.endOfInit
           // CHECK:      arraydestroy.body:
           // CHECK:      arraydestroy.done1:
           // CHECK-NEXT:   br label %return 
         }
         // CHECK:    if.end:
         // CHECK:      call void @_ZN6PrintyC1Ev
         // CHECK:      store ptr %arrayinit.element4, ptr %arrayinit.endOfInit3, align 8
         Printy();
       })},
      {Printy(), ({
         if (foo()) {
           /** One dtor followed by an array destruction **/
           // CHECK:         if.then{{.*}}:
           // CHECK-NEXT:      load ptr, ptr %arrayinit.endOfInit3
           // CHECK:       arraydestroy.body{{.*}}:
           // CHECK:       arraydestroy.done{{.*}}:
           // CHECK:       arraydestroy.body{{.*}}:
           // CHECK:       arraydestroy.done{{.*}}:
           // CHECK-NEXT:    br label %return
           return;
         }
         Printy();
       })}};
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
