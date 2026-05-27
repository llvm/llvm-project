// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// A replay generates its tasks afresh (OpenMP 6.0 [14.3]), and the body of a
// task that may be recorded constructs its private copies from the initial
// values its task descriptor holds.  libomp reuses the recorded descriptor, so
// the task's relocation helper reads those initial values again at the start
// of each replay.  A replay starts from the taskgraph construct's enclosing
// data environment, so a firstprivate of a variable that environment has takes
// the value the variable has at replay: through the taskgraph's record for an
// automatic variable, and from its fixed address for one with static storage
// duration.
//
// The saved data environment supplies the rest: a list item with the `saved`
// modifier, and a variable declared inside the region, which has no value at
// replay.  Their initial values stay as the task was recorded with them.

struct Pt {
  int X, Y;
};
int G;
int Out;

void run(int n, struct Pt p, int s) {
#pragma omp taskgraph
  {
    int in = n;
#pragma omp task firstprivate(n, p, G, in) firstprivate(saved : s) shared(Out)
    Out = n + p.Y + G + in + s;
  }
}

// The initial values of n, p, G, in and s, in that order, then the body's
// copies of them.
// CHECK: %struct..kmp_privates.t = type { i32, %struct.Pt, i32, i32, i32, i32, %struct.Pt, i32, i32, i32 }

// CHECK: call i32 @__kmpc_taskgraph_task({{.*}}, ptr @.omp.taskgraph.relocate., ptr null)

// The helper reads n, p and G; in and s keep the values they were recorded
// with.
// CHECK: define internal void @.omp.taskgraph.relocate.(ptr noundef %0, ptr noundef %1)
// CHECK: [[REC:%.+]] = load ptr, ptr %.addr1,
// CHECK-NEXT: [[TASK:%.+]] = load ptr, ptr %.addr,
// CHECK-NEXT: load ptr, ptr [[TASK]],
// CHECK-NEXT: [[TASK2:%.+]] = load ptr, ptr %.addr,
// CHECK-NEXT: [[PRIV:%.+]] = getelementptr inbounds nuw %struct.kmp_task_t_with_privates, ptr [[TASK2]], i32 0, i32 1
// n: a scalar, through the record.
// CHECK-NEXT: [[NF:%.+]] = getelementptr inbounds nuw %struct.anon, ptr [[REC]], i32 0, i32 0
// CHECK-NEXT: [[NADDR:%.+]] = load ptr, ptr [[NF]],
// CHECK-NEXT: [[NDST:%.+]] = getelementptr inbounds nuw %struct..kmp_privates.t, ptr [[PRIV]], i32 0, i32 0
// CHECK-NEXT: [[NVAL:%.+]] = load i32, ptr [[NADDR]],
// CHECK-NEXT: store i32 [[NVAL]], ptr [[NDST]],
// p: an aggregate, through the record.
// CHECK-NEXT: [[PF:%.+]] = getelementptr inbounds nuw %struct.anon, ptr [[REC]], i32 0, i32 1
// CHECK-NEXT: [[PADDR:%.+]] = load ptr, ptr [[PF]],
// CHECK-NEXT: [[PDST:%.+]] = getelementptr inbounds nuw %struct..kmp_privates.t, ptr [[PRIV]], i32 0, i32 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[PDST]], ptr align 4 [[PADDR]], i64 8, i1 false)
// G: from its own address.
// CHECK-NEXT: [[GDST:%.+]] = getelementptr inbounds nuw %struct..kmp_privates.t, ptr [[PRIV]], i32 0, i32 2
// CHECK-NEXT: [[GVAL:%.+]] = load i32, ptr @G,
// CHECK-NEXT: store i32 [[GVAL]], ptr [[GDST]],
// CHECK-NEXT: ret void

#endif
