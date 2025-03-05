// RUN: %clang_cc1 -O2 -triple=x86_64-unknown-linux-gnu -emit-llvm %s -o -  | FileCheck %s
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
//  CHECK: [[Re1:%.*]] = load volatile float, ptr @cf
//  CHECK: [[Im1:%.*]] = load volatile float, ptr getelementptr
//  CHECK: [[Add1:%.*]] = fadd float [[Re1]], 1.000000e+00
//  CHECK: store volatile float [[Add1]], ptr @cf
//  CHECK: store volatile float [[Im1]], ptr getelementptr
      static_cast<volatile _Complex float &>(cf) = static_cast<volatile _Complex float&>(cf) + 1;
//  CHECK: [[Re2:%.*]] = load volatile float, ptr @cf
//  CHECK: [[Im2:%.*]] = load volatile float, ptr getelementptr
//  CHECK: [[Add2:%.*]] = fadd float [[Re2]], 1.000000e+00
//  CHECK: store volatile float [[Add2]], ptr @cf
//  CHECK: store volatile float [[Im2]], ptr getelementptr
    const_cast<volatile  int  &>(a.a) = const_cast<volatile int &>(t.a) ;
//  CHECK: [[I1:%.*]] = load volatile i32, ptr @t
//  CHECK: store volatile i32 [[I1]], ptr @a
    static_cast<volatile  int  &>(a.b) = static_cast<volatile int  &>(t.a) ;
//  CHECK: [[I2:%.*]] = load volatile i32, ptr @t
//  CHECK: store volatile i32 [[I2]], ptr getelementptr
    const_cast<volatile int&>(vt) = const_cast<volatile int&>(vt) + 1;
//  CHECK: [[I3:%.*]] = load volatile i32, ptr @vt
//  CHECK: [[Add3:%.*]] = add nsw i32 [[I3]], 1
//  CHECK: store volatile i32 [[Add3]], ptr @vt
     static_cast<volatile int&>(vt) = static_cast<volatile int&>(vt) + 1;
//  CHECK: [[I4:%.*]] = load volatile i32, ptr @vt
//  CHECK: [[Add4:%.*]] = add nsw i32 [[I4]], 1
//  CHECK: store volatile i32 [[Add4]], ptr @vt
    vt = const_cast<int&>(vol);
//  [[I5:%.*]] = load i32, ptr @vol
//  store i32 [[I5]], ptr @vt
}
