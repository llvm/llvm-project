// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// A replay generates its tasks afresh (OpenMP 6.0 [14.3]), each with private
// copies of its own, constructed as the clauses say and destroyed when the
// task completes.  The body of a task that may be recorded constructs its
// copies itself, from the initial values of its firstprivates that the task
// descriptor holds, and libomp reuses the recorded descriptor, so:
//
//  - the initial value of a firstprivate of a variable in the taskgraph's
//    enclosing data environment is copy-constructed from that variable, read
//    through the taskgraph's record, by the task's relocation helper when a
//    replay starts, and destroyed by the task's destructors thunk when it
//    ends;
//  - that of a `saved` firstprivate is the saved data environment: the copy
//    the task was recorded with, which the thunk destroys when the record is
//    freed;
//  - a private copy needs nothing: the body default-constructs it.
//
// So the thunk takes which initial values to destroy, and the task's flags
// say so (0x200, here with tiedness and the thunk's own 0x8: 521).
//
// libomp keeps a copy of a recorded task, which a clone function completes by
// copy-constructing the saved initial values; the relocation helper constructs
// the others when a replay starts, so the clone leaves them alone.  The copy
// of the taskloop's tasks is the one task_dup makes.

struct NonTrivial {
  int v;
  int *self;
  NonTrivial();
  NonTrivial(const NonTrivial &other);
  ~NonTrivial();
};

void run() {
  NonTrivial a, s, p;
#pragma omp taskgraph
  {
#pragma omp task firstprivate(a) firstprivate(saved : s) private(p)
    {
      (void)(a.v + s.v + p.v);
    }
#pragma omp taskloop firstprivate(a) firstprivate(saved : s) num_tasks(2)
    for (int i = 0; i < 2; ++i)
      (void)(a.v + s.v);
  }
}

// CHECK: call i32 @__kmpc_taskgraph_task({{.*}}, i32 521, {{.*}}, ptr @.omp.taskgraph.relocate., ptr @.omp_taskgraph_clone.)
// CHECK: call i32 @__kmpc_taskgraph_taskloop({{.*}}, i32 521, {{.*}}, ptr @.omp_task_dup., ptr @.omp.taskgraph.relocate.[[LOOP:[.0-9]+]])

// The task's privates are p, the initial values of a and s, and the body's
// copies of a and s, in that order.

// The clone copy-constructs the initial value of s, the saved one, and not
// that of a.
// CHECK: define internal void @.omp_taskgraph_clone.(ptr noundef %0, ptr noundef %1, i32 noundef %2)
// CHECK: [[DPRIV:%.+]] = getelementptr inbounds nuw %struct.kmp_task_t_with_privates, ptr %{{.+}}, i32 0, i32 1
// CHECK: [[SPRIV:%.+]] = getelementptr inbounds nuw %struct.kmp_task_t_with_privates, ptr %{{.+}}, i32 0, i32 1
// CHECK-NEXT: [[SSRC:%.+]] = getelementptr inbounds nuw %struct..kmp_privates.t, ptr [[SPRIV]], i32 0, i32 2
// CHECK-NEXT: [[SDST:%.+]] = getelementptr inbounds nuw %struct..kmp_privates.t, ptr [[DPRIV]], i32 0, i32 2
// CHECK-NEXT: call void @_ZN10NonTrivialC1ERKS_(ptr {{.*}} [[SDST]], ptr {{.*}} [[SSRC]])
// CHECK-NEXT: ret void

// The thunk destroys both initial values for a task that runs outside a
// record (0), a when a replay ends (1), and s when the record is freed (2).
// CHECK: define internal noundef i32 @.omp_task_destructor.(i32 noundef %0, ptr noalias noundef %1, i32 noundef %2)
// CHECK: [[PRIV:%.+]] = getelementptr inbounds nuw %struct.kmp_task_t_with_privates, ptr %{{.+}}, i32 0, i32 1
// CHECK: switch i32 %{{.+}}, label %omp.destroy.done [
// CHECK-NEXT: i32 0, label %omp.destroy.all
// CHECK-NEXT: i32 1, label %omp.destroy.replay
// CHECK-NEXT: i32 2, label %omp.destroy.record
// CHECK-NEXT: ]
// CHECK: omp.destroy.all:
// CHECK-NEXT: [[A0:%.+]] = getelementptr inbounds nuw %struct..kmp_privates.t, ptr [[PRIV]], i32 0, i32 1
// CHECK-NEXT: call void @_ZN10NonTrivialD1Ev(ptr {{.*}} [[A0]])
// CHECK-NEXT: [[S0:%.+]] = getelementptr inbounds nuw %struct..kmp_privates.t, ptr [[PRIV]], i32 0, i32 2
// CHECK-NEXT: call void @_ZN10NonTrivialD1Ev(ptr {{.*}} [[S0]])
// CHECK-NEXT: br label %omp.destroy.done
// CHECK: omp.destroy.replay:
// CHECK-NEXT: [[A1:%.+]] = getelementptr inbounds nuw %struct..kmp_privates.t, ptr [[PRIV]], i32 0, i32 1
// CHECK-NEXT: call void @_ZN10NonTrivialD1Ev(ptr {{.*}} [[A1]])
// CHECK-NEXT: br label %omp.destroy.done
// CHECK: omp.destroy.record:
// CHECK-NEXT: [[S2:%.+]] = getelementptr inbounds nuw %struct..kmp_privates.t, ptr [[PRIV]], i32 0, i32 2
// CHECK-NEXT: call void @_ZN10NonTrivialD1Ev(ptr {{.*}} [[S2]])
// CHECK-NEXT: br label %omp.destroy.done

// The relocation helper reads a through the record when a replay starts.
// CHECK: define internal void @.omp.taskgraph.relocate.(ptr noundef %0, ptr noundef %1)
// CHECK: [[REC:%.+]] = load ptr, ptr %.addr1,
// CHECK: [[RPRIV:%.+]] = getelementptr inbounds nuw %struct.kmp_task_t_with_privates, ptr %{{.+}}, i32 0, i32 1
// CHECK-NEXT: [[AF:%.+]] = getelementptr inbounds nuw %struct.anon, ptr [[REC]], i32 0, i32 0
// CHECK-NEXT: [[AADDR:%.+]] = load ptr, ptr [[AF]],
// CHECK-NEXT: [[ADST2:%.+]] = getelementptr inbounds nuw %struct..kmp_privates.t, ptr [[RPRIV]], i32 0, i32 1
// CHECK-NEXT: call void @_ZN10NonTrivialC1ERKS_(ptr {{.*}} [[ADST2]], ptr {{.*}} [[AADDR]])
// CHECK-NEXT: ret void

// The taskloop's thunk and helper treat each of its tasks' initial values the
// same way.
// CHECK: define internal noundef i32 @.omp_task_destructor.{{[.0-9]+}}(i32 noundef %0, ptr noalias noundef %1, i32 noundef %2)
// CHECK: omp.destroy.replay:
// CHECK: call void @_ZN10NonTrivialD1Ev(
// CHECK: omp.destroy.record:
// CHECK: call void @_ZN10NonTrivialD1Ev(
// CHECK: define internal void @.omp.taskgraph.relocate.[[LOOP]](ptr noundef %0, ptr noundef %1)
// CHECK: call void @_ZN10NonTrivialC1ERKS_(
// CHECK: ret void

#endif
