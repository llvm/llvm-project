// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-linux-gnu -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// A task that may be recorded into a taskgraph is replayed from its recorded
// task descriptor, so its body leaves the descriptor as it found it.  The
// descriptor holds only the initial values of the firstprivates, and the body
// constructs a copy of every private list item of its own -- a firstprivate's
// from its initial value, a private one by default construction -- uses
// those, and destroys them when it ends.  The task's destructor thunk destroys
// only the initial values.  An untied task's copies live in the descriptor
// too, beside the initial values, so that they last from its first part to
// its last.  A task that cannot be recorded is unchanged.

struct NT {
  int v;
  NT();
  NT(const NT &);
  ~NT();
};

void use(int);

void run(int n) {
  NT a, p;
#pragma omp taskgraph
  {
#pragma omp task firstprivate(a, n) private(p)
    {
      use(a.v + p.v + n);
      a.v = n;
    }
#pragma omp task untied firstprivate(a)
    {
      use(a.v);
#pragma omp taskyield
      use(a.v);
    }
#pragma omp task replayable(false) firstprivate(a)
    use(a.v);
  }
}

// p, the initial values of a and n, and the body's copies of a and n.
// CHECK: %struct..kmp_privates.t = type { %struct.NT, %struct.NT, i32, %struct.NT, i32 }
// The untied task's initial value of a and its copy.
// CHECK: %struct..kmp_privates.t.{{[0-9]+}} = type { %struct.NT, %struct.NT }
// The task that cannot be recorded has a's copy alone.
// CHECK: %struct..kmp_privates.t.{{[0-9]+}} = type { %struct.NT }

// CHECK-LABEL: define internal noundef i32 @.omp_task_entry.(
// CHECK: call void %{{.+}}(ptr %{{.+}}, ptr [[P:%.priv.ptr.addr[^ ,]*]], ptr [[A:%.firstpriv.ptr.addr[^ ,]*]], ptr [[N:%.firstpriv.ptr.addr[^ ,]*]], ptr [[ACOPY:%.firstpriv.copy.ptr.addr[^ ,]*]], ptr [[NCOPY:%.firstpriv.copy.ptr.addr[^ ,)]*]])
// CHECK-NEXT: [[PADDR:%.+]] = load ptr, ptr [[P]],
// CHECK-NEXT: [[ACOPYADDR:%.+]] = load ptr, ptr [[ACOPY]],
// CHECK-NEXT: [[NCOPYADDR:%.+]] = load ptr, ptr [[NCOPY]],
// CHECK-NEXT: [[AADDR:%.+]] = load ptr, ptr [[A]],
// CHECK-NEXT: [[NADDR:%.+]] = load ptr, ptr [[N]],
// CHECK-NEXT: invoke void @_ZN2NTC1Ev(ptr {{.*}} [[PADDR]])
// CHECK: invoke void @_ZN2NTC1ERKS_(ptr {{.*}} [[ACOPYADDR]], ptr {{.*}} [[AADDR]])
// CHECK: [[NVAL:%.+]] = load i32, ptr [[NADDR]],
// CHECK-NEXT: store i32 [[NVAL]], ptr [[NCOPYADDR]],
// The body works on its copies.
// CHECK-NEXT: load i32, ptr [[ACOPYADDR]],
// CHECK-NEXT: load i32, ptr [[PADDR]],
// CHECK: load i32, ptr [[NCOPYADDR]],
// CHECK: invoke void @_Z3usei(
// CHECK: [[NV:%.+]] = load i32, ptr [[NCOPYADDR]],
// CHECK-NEXT: store i32 [[NV]], ptr [[ACOPYADDR]],
// CHECK-NEXT: call void @_ZN2NTD1Ev(ptr {{.*}} [[ACOPYADDR]])
// CHECK-NEXT: call void @_ZN2NTD1Ev(ptr {{.*}} [[PADDR]])
// CHECK-NEXT: ret i32 0

// The thunk destroys the initial value of a and nothing else.
// CHECK-LABEL: define internal noundef i32 @.omp_task_destructor.(
// CHECK-NOT: call void @_ZN2NTD1Ev(
// CHECK: [[INIT:%.+]] = getelementptr inbounds nuw %struct..kmp_privates.t, ptr %{{.+}}, i32 0, i32 1
// CHECK-NEXT: call void @_ZN2NTD1Ev(ptr {{.*}} [[INIT]])
// CHECK-NOT: getelementptr inbounds nuw %struct..kmp_privates.t, ptr %{{.+}}, i32 0, i32 {{[0234]}}
// CHECK: ret i32

// The untied task makes its copy in its first part and destroys it in its
// last.
// CHECK-LABEL: define internal noundef i32 @.omp_task_entry..{{[0-9]+}}(
// CHECK: call void %{{.+}}(ptr %{{.+}}, ptr [[UA:%.firstpriv.ptr.addr[^ ,]*]], ptr [[UACOPY:%.firstpriv.copy.ptr.addr[^ ,)]*]])
// CHECK-NEXT: [[UACOPYADDR:%.+]] = load ptr, ptr [[UACOPY]],
// CHECK-NEXT: [[UAADDR:%.+]] = load ptr, ptr [[UA]],
// CHECK: switch i32 %{{.+}}, label %.untied.done.
// CHECK: .untied.jmp.1.i:
// CHECK-NEXT: invoke void @_ZN2NTC1ERKS_(ptr {{.*}} [[UACOPYADDR]], ptr {{.*}} [[UAADDR]])
// CHECK: call i32 @__kmpc_omp_taskyield(
// CHECK: .untied.jmp.3.i:
// CHECK: call void @_ZN2NTD1Ev(ptr {{.*}} [[UACOPYADDR]])

// The task that cannot be recorded uses its copy in place, and its thunk
// destroys it.
// CHECK-LABEL: define internal noundef i32 @.omp_task_entry..{{[0-9]+}}(
// CHECK: call void %{{.+}}(ptr %{{.+}}, ptr [[RA:%.firstpriv.ptr.addr[^ ,)]*]])
// CHECK-NEXT: [[RAADDR:%.+]] = load ptr, ptr [[RA]],
// CHECK-NEXT: load i32, ptr [[RAADDR]],
// CHECK-NOT: @_ZN2NT
// CHECK: ret i32 0
// CHECK-LABEL: define internal noundef i32 @.omp_task_destructor..{{[0-9]+}}(
// CHECK: call void @_ZN2NTD1Ev(

#endif
