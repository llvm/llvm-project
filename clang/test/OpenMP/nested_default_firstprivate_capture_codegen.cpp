// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

// A construct with default(firstprivate) or default(private), nested inside a
// 'parallel' region, must not change how the 'parallel' region captures the
// variables it references.  Each variable below is shared in the 'parallel'
// region, so the region must be handed its address: a copy would lose the
// writes made through it.  Sema used to take the default data-sharing
// attribute of the innermost construct when deciding how an enclosing region
// captures a variable, and so captured each of these by copy.
//
// The capture kind is decided by the first reference to a variable, which in
// the cases with no clause on the variable is the one in the nested task; the
// write the 'parallel' region makes afterwards is the one that was lost.

#ifndef HEADER
#define HEADER

int G;
static volatile int Sink;

// A reduction under a nested default(private).
int reduction_private(int seed) {
  int x = seed;
  int res = 0;
#pragma omp parallel
#pragma omp single
  {
#pragma omp taskloop num_tasks(4) default(private) firstprivate(x) reduction(+ : res)
    for (int i = 0; i < 8; ++i)
      res += x + i;
  }
  return res;
}

// An explicitly shared variable, written by the nested task.
int shared_write() {
  int res = -1;
#pragma omp parallel num_threads(2)
#pragma omp single
  {
#pragma omp task default(firstprivate) shared(res)
    res = 42;
  }
  return res;
}

// No clause on 'v': the nested task reads it, then the 'parallel' region
// writes it.
int outer_write_firstprivate() {
  int v = 1;
#pragma omp parallel num_threads(1)
  {
#pragma omp task default(firstprivate) shared(Sink)
    Sink = v;
#pragma omp taskwait
    v = 42;
  }
  return v;
}

int outer_write_private() {
  int v = 1;
#pragma omp parallel num_threads(1)
  {
#pragma omp task default(private) shared(Sink)
    {
      v = 7;
      Sink = v;
    }
#pragma omp taskwait
    v = 42;
  }
  return v;
}

// A global, which a default(firstprivate) construct requires to be listed.  The
// task's copy must see the value the 'parallel' region assigned.
int global_firstprivate() {
  int out = -1;
#pragma omp parallel num_threads(1) shared(out)
  {
    G = 5;
#pragma omp task default(firstprivate) firstprivate(G) shared(out)
    out = G;
  }
  return out;
}

// A function-scope static, which default(firstprivate) does apply to.
int static_outer_write() {
  static int s;
  s = 1;
#pragma omp parallel num_threads(1)
  {
#pragma omp task default(firstprivate) shared(Sink)
    Sink = s;
#pragma omp taskwait
    s = 42;
  }
  return s;
}

#endif

// CHECK-LABEL: define {{.*}}@_Z17reduction_privatei(
// CHECK:         call {{.*}}@__kmpc_fork_call(ptr {{.*}}, i32 2, ptr @_Z17reduction_privatei.omp_outlined, ptr %{{[0-9a-z._]+}}, ptr %[[RES:[0-9a-z._]+]])
// CHECK:         %[[RET:.*]] = load i32, ptr %[[RES]], align 4
// CHECK-NEXT:    ret i32 %[[RET]]

// CHECK-LABEL: define {{.*}}@_Z12shared_writev(
// CHECK:         call {{.*}}@__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @_Z12shared_writev.omp_outlined, ptr %[[RES:[0-9a-z._]+]])
// CHECK:         %[[RET:.*]] = load i32, ptr %[[RES]], align 4
// CHECK-NEXT:    ret i32 %[[RET]]

// CHECK-LABEL: define {{.*}}@_Z24outer_write_firstprivatev(
// CHECK:         call {{.*}}@__kmpc_fork_call(ptr {{.*}}, i32 2, ptr @_Z24outer_write_firstprivatev.omp_outlined, ptr @_ZL4Sink, ptr %[[V:[0-9a-z._]+]])
// CHECK:         %[[RET:.*]] = load i32, ptr %[[V]], align 4
// CHECK-NEXT:    ret i32 %[[RET]]

// CHECK-LABEL: define {{.*}}@_Z19outer_write_privatev(
// CHECK:         call {{.*}}@__kmpc_fork_call(ptr {{.*}}, i32 2, ptr @_Z19outer_write_privatev.omp_outlined, ptr @_ZL4Sink, ptr %[[V:[0-9a-z._]+]])
// CHECK:         %[[RET:.*]] = load i32, ptr %[[V]], align 4
// CHECK-NEXT:    ret i32 %[[RET]]

// CHECK-LABEL: define {{.*}}@_Z19global_firstprivatev(
// CHECK:         call {{.*}}@__kmpc_fork_call(ptr {{.*}}, i32 2, ptr @_Z19global_firstprivatev.omp_outlined, ptr @G, ptr %{{[0-9a-z._]+}})

// CHECK-LABEL: define {{.*}}@_Z18static_outer_writev(
// CHECK:         call {{.*}}@__kmpc_fork_call(ptr {{.*}}, i32 2, ptr @_Z18static_outer_writev.omp_outlined, ptr @_ZL4Sink, ptr @_ZZ18static_outer_writevE1s)
// CHECK:         %[[RET:.*]] = load i32, ptr @_ZZ18static_outer_writevE1s, align 4
// CHECK-NEXT:    ret i32 %[[RET]]
