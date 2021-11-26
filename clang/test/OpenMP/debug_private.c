// This testcase checks emission of debug info for variables inside
// private/firstprivate/lastprivate.

// REQUIRES: x86_64-linux

// RUN: %clang_cc1 -debug-info-kind=constructor -x c -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

// CHECK: define internal i32 @.omp_task_entry.

// CHECK:  call void @llvm.dbg.declare(metadata i32** %.priv.ptr.addr.i, metadata [[PRIV1:![0-9]+]], metadata !DIExpression(DW_OP_deref))
// CHECK:  call void @llvm.dbg.declare(metadata i32** %.priv.ptr.addr1.i, metadata [[PRIV2:![0-9]+]], metadata !DIExpression(DW_OP_deref))
// CHECK:  call void @llvm.dbg.declare(metadata i32** %.firstpriv.ptr.addr.i, metadata [[FPRIV:![0-9]+]], metadata !DIExpression(DW_OP_deref))

// CHECK: [[PRIV1]] = !DILocalVariable(name: "priv1"
// CHECK: [[PRIV2]] = !DILocalVariable(name: "priv2"
// CHECK: [[FPRIV]] = !DILocalVariable(name: "fpriv"

extern int printf(const char *, ...);

int foo(int n) {
  int res, priv1, priv2, fpriv;
  fpriv = n + 4;

  if (n < 2)
    return n;
  else {
#pragma omp task shared(res) private(priv1, priv2) firstprivate(fpriv)
    {
      priv1 = n;
      priv2 = n + 2;
      printf("Task n=%d,priv1=%d,priv2=%d,fpriv=%d\n", n, priv1, priv2, fpriv);

      res = priv1 + priv2 + fpriv + foo(n - 1);
    }
#pragma omp taskwait
    return res;
  }
}

int main() {
  int n = 10;
  printf("foo(%d) = %d\n", n, foo(n));
  return 0;
}
