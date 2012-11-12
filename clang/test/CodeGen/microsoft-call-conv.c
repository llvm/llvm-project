// RUN: %clang_cc1 -triple i386-pc-linux -emit-llvm < %s | FileCheck %s

void __fastcall f1(void);
void __stdcall f2(void);
void __thiscall f3(void);
void __fastcall f4(void) {
// CHECK: define x86_fastcallcc void @f4()
  f1();
// CHECK: call x86_fastcallcc void @f1()
}
void __stdcall f5(void) {
// CHECK: define x86_stdcallcc void @f5()
  f2();
// CHECK: call x86_stdcallcc void @f2()
}
void __thiscall f6(void) {
// CHECK: define x86_thiscallcc void @f6()
  f3();
// CHECK: call x86_thiscallcc void @f3()
}

// PR5280
void (__fastcall *pf1)(void) = f1;
void (__stdcall *pf2)(void) = f2;
void (__thiscall *pf3)(void) = f3;
void (__fastcall *pf4)(void) = f4;
void (__stdcall *pf5)(void) = f5;
void (__thiscall *pf6)(void) = f6;

int main(void) {
    f4(); f5(); f6();
    // CHECK: call x86_fastcallcc void @f4()
    // CHECK: call x86_stdcallcc void @f5()
    // CHECK: call x86_thiscallcc void @f6()
    pf1(); pf2(); pf3(); pf4(); pf5(); pf6();
    // CHECK: call x86_fastcallcc void %{{.*}}()
    // CHECK: call x86_stdcallcc void %{{.*}}()
    // CHECK: call x86_thiscallcc void %{{.*}}()
    // CHECK: call x86_fastcallcc void %{{.*}}()
    // CHECK: call x86_stdcallcc void %{{.*}}()
    // CHECK: call x86_thiscallcc void %{{.*}}()
    return 0;
}

// PR7117
void __stdcall f7(foo) int foo; {}
void f8(void) {
  f7(0);
  // CHECK: call x86_stdcallcc void @f7(i32 0)
}
