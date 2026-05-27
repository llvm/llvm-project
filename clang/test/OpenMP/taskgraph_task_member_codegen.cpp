// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// A taskgraph in a member function.  A replay starts from the taskgraph
// construct's enclosing data environment (OpenMP 6.0 [14.3]), whose `this` is
// the object the member function was called for this time, so when a replay
// starts the task's relocation helper takes `this` from the taskgraph's
// reconstructed record:
//
//  - the task's own capture of `this` is set to it;
//  - the initial value of a firstprivate of a non-static data member, whose
//    list item is `this->m`, is read from that member of it.

struct S {
  int m;
  int out;
  void run() {
#pragma omp taskgraph
    {
#pragma omp task firstprivate(m)
      out = m;
    }
  }
};

void f(S &s) { s.run(); }

// The record holds `this` in its first field, by value.
// CHECK: define internal void @.omp.taskgraph.relocate.(ptr noundef %0, ptr noundef %1)
// CHECK: [[REC:%.+]] = load ptr, ptr %.addr1,
// CHECK-NEXT: [[TASK:%.+]] = load ptr, ptr %.addr,
// CHECK-NEXT: [[SHAREDS:%.+]] = load ptr, ptr [[TASK]],
// CHECK-NEXT: [[THISF:%.+]] = getelementptr inbounds nuw %struct.anon, ptr [[REC]], i32 0, i32 0
// CHECK-NEXT: [[THIS:%.+]] = load ptr, ptr [[THISF]],
// CHECK-NEXT: [[SLOT:%.+]] = getelementptr ptr, ptr [[SHAREDS]], i64 0
// CHECK-NEXT: store ptr [[THIS]], ptr [[SLOT]],
// CHECK: [[PRIV:%.+]] = getelementptr inbounds nuw %struct.kmp_task_t_with_privates, ptr %{{.+}}, i32 0, i32 1
// CHECK-NEXT: [[THISF2:%.+]] = getelementptr inbounds nuw %struct.anon, ptr [[REC]], i32 0, i32 0
// CHECK-NEXT: [[THIS2:%.+]] = load ptr, ptr [[THISF2]],
// CHECK-NEXT: [[MADDR:%.+]] = getelementptr inbounds nuw %struct.S, ptr [[THIS2]], i32 0, i32 0
// CHECK-NEXT: [[MDST:%.+]] = getelementptr inbounds nuw %struct..kmp_privates.t, ptr [[PRIV]], i32 0, i32 0
// CHECK-NEXT: [[MVAL:%.+]] = load i32, ptr [[MADDR]],
// CHECK-NEXT: store i32 [[MVAL]], ptr [[MDST]],
// CHECK-NEXT: ret void

#endif
