// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-version=45 -triple x86_64-apple-darwin13.4.0 -emit-llvm -o - %s | FileCheck %s

//CHECK: call i32 @__kmpc_cancel
//CHECK: br {{.*}}label %[[EXIT:[^,].+]], label %[[CONTINUE:.+]]
//CHECK: [[EXIT]]:
//CHECK: store i32 [[EXIT_SLOT:[0-9]+]]
//CHECK: br label %[[CLEANUP:.+]]
//CHECK: [[CONTINUE]]:
//CHECK: store i32 [[CONT_SLOT:[0-9]+]],
//CHECK: br label %[[CLEANUP]]
//CHECK: [[CLEANUP]]:
//CHECK-NEXT: call void @_ZN3ObjD1Ev
//CHECK: switch i32{{.*}}, label %[[UNREACHABLE:.+]] [
//CHECK:   i32 [[CONT_SLOT]], label %[[CLEANUPCONT:.+]]
//CHECK:   i32 [[EXIT_SLOT]], label %[[CANCELEXIT:.+]]
//CHECK-NEXT: ]
//CHECK: [[CLEANUPCONT]]:
//CHECK: br label %[[CANCELCONT:.+]]
//CHECK: [[CANCELCONT]]:
//CHECK-NEXT: call void @__kmpc_barrier(
//CHECK-NEXT: ret void
//CHECK: [[UNREACHABLE]]:
//CHECK-NEXT: unreachable
//CHECK-NEXT: }

struct Obj {
  int a; Obj(); Obj(const Obj& r) = delete; Obj &operator=(const Obj& r);
  ~Obj();
};
 
void foo() {
  int i,count = 0;
  Obj obj;

  #pragma omp parallel private(i) num_threads(1)
  {
      #pragma omp for reduction(+:count) lastprivate(obj)
      for (i=0; i<1000; i++) {
            if(i==100) {
                obj.a = 100;
                #pragma omp cancel for
            }
            count++;
        }
    }
}
