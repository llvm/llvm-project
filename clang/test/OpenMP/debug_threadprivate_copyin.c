// This testcase checks emission of debug info for threadprivate variables
// present in any clause of OpenMP construct.

// REQUIRES: x86_64-linux

// RUN: %clang_cc1 -debug-info-kind=limited -x c -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

// CHECK: define internal void @.omp_outlined._debug__(
// CHECK: call void @llvm.dbg.declare(metadata ptr %.global_tid..addr,
// CHECK: call void @llvm.dbg.declare(metadata ptr %.bound_tid..addr,
// CHECK: call void @llvm.dbg.declare(metadata ptr %nt.addr
// CHECK: store ptr %gbl_dynamic_int, ptr %gbl_dynamic_int.addr, align 8
// CHECK-NOT: call void @llvm.dbg.declare(metadata ptr %gbl_dynamic_int.addr
// CHECK-NOT: call void @llvm.dbg.declare(metadata ptr %gbl_static_int.addr

extern int printf(const char *, ...);
extern void omp_set_num_threads(int);
extern int omp_get_num_threads(void);
extern int omp_get_thread_num(void);

int gbl_dynamic_int;
__thread int gbl_static_int;

#pragma omp threadprivate(gbl_dynamic_int)

int main() {
  int nt = 0;
  int offset = 10;
  gbl_dynamic_int = 55;
  gbl_static_int = 77;

  omp_set_num_threads(4);
#pragma omp parallel copyin(gbl_dynamic_int, gbl_static_int)
  {
    int data;
    int tid;
    nt = omp_get_num_threads();
    tid = omp_get_thread_num();
    data = gbl_dynamic_int + gbl_static_int;
    gbl_dynamic_int += 10;
    gbl_static_int += 20;
#pragma omp barrier
    if (tid == 0)
      printf("In parallel region total threads = %d, thread id = %d data=%d gbl_dyn_addr = %p, gbl_static_addr = %p\n",
             nt, tid, data, &gbl_dynamic_int, &gbl_static_int);
    if (tid == 1)
      printf("In parallel region total threads = %d, thread id = %d data=%d gbl_dyn_addr = %p, gbl_static_addr = %p\n",
             nt, tid, data, &gbl_dynamic_int, &gbl_static_int);
    if (tid == 2)
      printf("In parallel region total threads = %d, thread id = %d data=%d gbl_dyn_addr = %p, gbl_static_addr = %p\n",
             nt, tid, data, &gbl_dynamic_int, &gbl_static_int);
    if (tid == 3)
      printf("In parallel region total threads = %d, thread id = %d data=%d gbl_dyn_addr = %p, gbl_static_addr = %p\n",
             nt, tid, data, &gbl_dynamic_int, &gbl_static_int);
  }

  return 0;
}
