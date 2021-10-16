// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -fdefault-calling-conv=cdecl -emit-llvm -o - %s | FileCheck %s --check-prefix=CDECL --check-prefix=ALL
// RUN: %clang_cc1 -triple i786-unknown-linux-gnu -target-feature +sse4.2 -fdefault-calling-conv=fastcall -emit-llvm -o - %s | FileCheck %s --check-prefix=FASTCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple i486-unknown-linux-gnu -fdefault-calling-conv=stdcall -emit-llvm -o - %s | FileCheck %s --check-prefix=STDCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple i486-unknown-linux-gnu -mrtd -emit-llvm -o - %s | FileCheck %s --check-prefix=STDCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple i986-unknown-linux-gnu -fdefault-calling-conv=vectorcall -emit-llvm -o - %s | FileCheck %s --check-prefix=VECTORCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple i986-unknown-linux-gnu -fdefault-calling-conv=regcall -emit-llvm -o - %s | FileCheck %s --check-prefix=REGCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple i686-pc-win32 -fdefault-calling-conv=vectorcall -emit-llvm -o - %s -DWINDOWS | FileCheck %s --check-prefix=WIN32
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fdefault-calling-conv=vectorcall -emit-llvm -o - %s -DWINDOWS | FileCheck %s --check-prefix=WIN64
// RUN: %clang_cc1 -triple i686-pc-win32 -emit-llvm -o - %s -DEXPLICITCC | FileCheck %s --check-prefix=EXPLICITCC

// CDECL: define{{.*}} void @_Z5test1v
// FASTCALL: define{{.*}} x86_fastcallcc void @_Z5test1v
// STDCALL: define{{.*}} x86_stdcallcc void @_Z5test1v
// VECTORCALL: define{{.*}} x86_vectorcallcc void @_Z5test1v
// REGCALL: define{{.*}} x86_regcallcc void @_Z17__regcall3__test1v
void test1() {}

// fastcall, stdcall, vectorcall and regcall do not support variadic functions.
// CDECL: define{{.*}} void @_Z12testVariadicz
// FASTCALL: define{{.*}} void @_Z12testVariadicz
// STDCALL: define{{.*}} void @_Z12testVariadicz
// VECTORCALL: define{{.*}} void @_Z12testVariadicz
// REGCALL: define{{.*}} void @_Z12testVariadicz
void testVariadic(...){}

// ALL: define{{.*}} void @_Z5test2v
void __attribute__((cdecl)) test2() {}

// ALL: define{{.*}} x86_fastcallcc void @_Z5test3v
void __attribute__((fastcall)) test3() {}

// ALL: define{{.*}} x86_stdcallcc void @_Z5test4v
void __attribute__((stdcall)) test4() {}

// ALL: define{{.*}} x86_vectorcallcc void @_Z5test5v
void __attribute__((vectorcall)) test5() {}

// ALL: define{{.*}} x86_regcallcc void @_Z17__regcall3__test6v
void __attribute__((regcall)) test6() {}

// ALL: define linkonce_odr void @_ZN1A11test_memberEv
class A {
public:
  void test_member() {}
};

void test() {
  A a;
  a.test_member();
}

// ALL: define{{.*}} i32 @main
int main() {
  return 1;
}

#ifdef WINDOWS
// WIN32: define dso_local noundef i32 @wmain
// WIN64: define dso_local noundef i32 @wmain
int wmain() {
  return 1;
}
// WIN32: define dso_local x86_stdcallcc noundef i32 @WinMain
// WIN64: define dso_local noundef i32 @WinMain
int WinMain() {
  return 1;
}
// WIN32: define dso_local x86_stdcallcc noundef i32 @wWinMain
// WIN64: define dso_local noundef i32 @wWinMain
int wWinMain() {
  return 1;
}
// WIN32: define dso_local x86_stdcallcc noundef i32 @DllMain
// WIN64: define dso_local noundef i32 @DllMain
int DllMain() {
  return 1;
}
#endif // Windows

#ifdef EXPLICITCC
// EXPLICITCC: define dso_local x86_fastcallcc noundef i32 @wmain
int __fastcall wmain() {
  return 1;
}
// EXPLICITCC: define dso_local x86_fastcallcc noundef i32 @WinMain
int __fastcall WinMain() {
  return 1;
}
// EXPLICITCC: define dso_local x86_fastcallcc noundef i32 @wWinMain
int __fastcall wWinMain() {
  return 1;
}
// EXPLICITCC: define dso_local x86_fastcallcc noundef i32 @DllMain
int __fastcall DllMain() {
  return 1;
}
#endif // ExplicitCC
