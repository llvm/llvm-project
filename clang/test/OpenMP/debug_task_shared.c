// This testcase checks emission of debug info for variables
// inside shared clause of task construct.

// REQUIRES: x86_64-linux

// RUN: %clang_cc1 -debug-info-kind=constructor -DSHARED -x c -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -debug-info-kind=constructor -x c -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm %s -o - | FileCheck %s --check-prefix=NEG
// RUN: %clang_cc1 -debug-info-kind=line-directives-only -DSHARED -x c -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm %s -o - | FileCheck %s --check-prefix=NEG
// RUN: %clang_cc1 -debug-info-kind=line-tables-only -DSHARED -x c -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm %s -o - | FileCheck %s --check-prefix=NEG
// RUN: %clang_cc1 -debug-info-kind=limited -DSHARED -x c -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK
// expected-no-diagnostics

// CHECK-LABEL: define internal i32 @.omp_task_entry.

// CHECK-DAG:  [[CONTEXT:%[0-9]+]] = load %struct.anon*, %struct.anon** %__context.addr.i, align 8
// CHECK-DAG:  call void @llvm.dbg.declare(metadata %struct.anon* [[CONTEXT]], metadata [[SHARE2:![0-9]+]], metadata !DIExpression(DW_OP_deref))
// CHECK-DAG:  call void @llvm.dbg.declare(metadata %struct.anon* [[CONTEXT]], metadata [[SHARE3:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 8, DW_OP_deref))
// CHECK-DAG:  call void @llvm.dbg.declare(metadata %struct.anon* [[CONTEXT]], metadata [[SHARE1:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 16, DW_OP_deref))

// CHECK-DAG: [[SHARE2]] = !DILocalVariable(name: "share2"
// CHECK-DAG: [[SHARE3]] = !DILocalVariable(name: "share3"
// CHECK-DAG: [[SHARE1]] = !DILocalVariable(name: "share1"

// NEG-LABEL: define internal i32 @.omp_task_entry.
// NEG:  [[CONTEXT:%[0-9]+]] = load %struct.anon*, %struct.anon** %__context.addr.i, align 8
// NEG-NOT: call void @llvm.dbg.declare(metadata %struct.anon* [[CONTEXT]], metadata {{![0-9]+}}, metadata !DIExpression(DW_OP_deref))

extern int printf(const char *, ...);

int foo(int n) {
  int share1 = 9, share2 = 11, share3 = 13, priv1, priv2, fpriv;
  fpriv = n + 4;

  if (n < 2)
    return n;
  else {
#if SHARED
#pragma omp task shared(share1, share2) private(priv1, priv2) firstprivate(fpriv) shared(share3)
#else
#pragma omp task private(priv1, priv2) firstprivate(fpriv)
#endif
    {
      priv1 = n;
      priv2 = n + 2;
      share2 += share3;
      printf("share1 = %d, share2 = %d, share3 = %d\n", share1, share2, share3);
      share1 = priv1 + priv2 + fpriv + foo(n - 1) + share2 + share3;
    }
#pragma omp taskwait
    return share1 + share2 + share3;
  }
}

int main() {
  int n = 10;
  printf("foo(%d) = %d\n", n, foo(n));
  return 0;
}
