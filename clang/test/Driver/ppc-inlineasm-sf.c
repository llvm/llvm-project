// RUN: not %clang -target powerpc-unknown-linux -O2 -fPIC -m32 -msoft-float %s -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-ERRMSG %s
int foo ()
{
  double x,y;
  int a;
  __asm__ ("fctiw %0,%1" : "=f"(x) : "f"(y));
  // CHECK-ERRMSG:      error: invalid output constraint '=f' in asm
  // CHECK-ERRMSG-NEXT:  __asm__ ("fctiw %0,%1" : "=f"(x) : "f"(y));
  __asm__ ("fctiw %0,%1" : "=d"(x) : "d"(y));
  // CHECK-ERRMSG: error: invalid output constraint '=d' in asm
  // CHECK-ERRMSG-NEXT: __asm__ ("fctiw %0,%1" : "=d"(x) : "d"(y));
  __asm__ ("vec_dss %0" : "=v"(a));
  // CHECK-ERRMSG: error: invalid output constraint '=v' in asm
  // CHECK-ERRMSG-NEXT: __asm__ ("vec_dss %0" : "=v"(a));
}

