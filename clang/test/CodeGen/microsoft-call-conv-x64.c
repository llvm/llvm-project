// RUN: %clang_cc1 -triple x86_64-pc-win32 -emit-llvm < %s | FileCheck %s

void __fastcall f1(void);
void __stdcall f2(void);
void __fastcall f4(void) {
// CHECK-LABEL: define void @f4()
  f1();
// CHECK: call void @f1()
}
void __stdcall f5(void) {
// CHECK-LABEL: define void @f5()
  f2();
// CHECK: call void @f2()
}

// PR5280
void (__fastcall *pf1)(void) = f1;
void (__stdcall *pf2)(void) = f2;
void (__fastcall *pf4)(void) = f4;
void (__stdcall *pf5)(void) = f5;

int main(void) {
    f4(); f5();
    // CHECK: call void @f4()
    // CHECK: call void @f5()
    pf1(); pf2(); pf4(); pf5();
    // CHECK: call void %{{.*}}()
    // CHECK: call void %{{.*}}()
    // CHECK: call void %{{.*}}()
    // CHECK: call void %{{.*}}()
    return 0;
}

// PR7117
void __stdcall f7(foo) int foo; {}
void f8(void) {
  f7(0);
  // CHECK: call void @f7(i32 0)
}
