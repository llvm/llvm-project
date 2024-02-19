// RUN: %clang_cc1 -O2 -triple=x86_64-unknown-linux-gnu -emit-llvm %s -o -  | FileCheck %s -check-prefix CHECK
struct agg 
{
int a ;
int b ;
} t;
struct agg a;
int vt=10;
_Complex float cf;
int volatile vol =10;
void f0() {
    const_cast<volatile _Complex float &>(cf) = const_cast<volatile _Complex float&>(cf) + 1;
//  CHECK: %cf.real = load volatile float, ptr @cf
//  CHECK: %cf.imag = load volatile float, ptr getelementptr
//  CHECK: %add.r = fadd float %cf.real, 1.000000e+00
//  CHECK: %add.i = fadd float %cf.imag, 0.000000e+00
//  CHECK: store volatile float %add.r
//  CHECK: store volatile float %add.i, ptr getelementptr
      static_cast<volatile _Complex float &>(cf) = static_cast<volatile _Complex float&>(cf) + 1;
//  CHECK: %cf.real1 = load volatile float, ptr @cf
//  CHECK: %cf.imag2 = load volatile float, ptr getelementptr
//  CHECK: %add.r3 = fadd float %cf.real1, 1.000000e+00
//  CHECK: %add.i4 = fadd float %cf.imag2, 0.000000e+00
//  CHECK: store volatile float %add.r3, ptr @cf
//  CHECK: store volatile float %add.i4, ptr getelementptr
    const_cast<volatile  int  &>(a.a) = const_cast<volatile int &>(t.a) ;
//  CHECK: %0 = load volatile i32, ptr @t
//  CHECK: store volatile i32 %0, ptr @a
    static_cast<volatile  int  &>(a.b) = static_cast<volatile int  &>(t.a) ;
//  CHECK: %1 = load volatile i32, ptr @t
//  CHECK: store volatile i32 %1, ptr getelementptr
    const_cast<volatile int&>(vt) = const_cast<volatile int&>(vt) + 1;
//  CHECK: %2 = load volatile i32, ptr @vt
//  CHECK: %add = add nsw i32 %2, 1
//  CHECK: store volatile i32 %add, ptr @vt
     static_cast<volatile int&>(vt) = static_cast<volatile int&>(vt) + 1;
//  CHECK: %3 = load volatile i32, ptr @vt
//  CHECK: %add5 = add nsw i32 %3, 1
//  CHECK: store volatile i32 %add5, ptr @vt
    vt = const_cast<int&>(vol);
//  %4 = load i32, ptr @vol
//  store i32 %4, ptr @vt
}
