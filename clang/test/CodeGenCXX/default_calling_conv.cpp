// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -fdefault-calling-conv=cdecl -emit-llvm -o - %s -DMAIN | FileCheck %s --check-prefix=CDECL --check-prefix=ALL
// RUN: %clang_cc1 -triple i786-unknown-linux-gnu -target-feature +sse4.2 -fdefault-calling-conv=fastcall -emit-llvm -o - %s -DMAIN | FileCheck %s --check-prefix=FASTCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple i486-unknown-linux-gnu -fdefault-calling-conv=stdcall -emit-llvm -o - %s -DMAIN | FileCheck %s --check-prefix=STDCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple i486-unknown-linux-gnu -mrtd -emit-llvm -o - %s -DMAIN | FileCheck %s --check-prefix=STDCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple i986-unknown-linux-gnu -fdefault-calling-conv=vectorcall -emit-llvm -o - %s -DMAIN | FileCheck %s --check-prefix=VECTORCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple i986-unknown-linux-gnu -fdefault-calling-conv=regcall -emit-llvm -o - %s -DMAIN | FileCheck %s --check-prefix=REGCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple i386-pc-win32  -target-feature +sse4.2 -fdefault-calling-conv=fastcall -emit-llvm -o - %s -DWMAIN | FileCheck %s  --check-prefix=WMAIN
// RUN: %clang_cc1 -triple i386-pc-win32  -target-feature +sse4.2 -fdefault-calling-conv=fastcall -emit-llvm -o - %s -DWINMAIN | FileCheck %s  --check-prefix=WINMAIN
// RUN: %clang_cc1 -triple i386-pc-win32  -target-feature +sse4.2 -fdefault-calling-conv=fastcall -emit-llvm -o - %s -DWWINMAIN | FileCheck %s  --check-prefix=WWINMAIN
// RUN: %clang_cc1 -triple i386-pc-win32  -target-feature +sse4.2 -fdefault-calling-conv=fastcall -emit-llvm -o - %s -DDLLMAIN | FileCheck %s  --check-prefix=DLLMAIN
//
// CDECL: define void @_Z5test1v
// FASTCALL: define x86_fastcallcc void @_Z5test1v
// STDCALL: define x86_stdcallcc void @_Z5test1v
// VECTORCALL: define x86_vectorcallcc void @_Z5test1v
// REGCALL: define x86_regcallcc void @_Z17__regcall3__test1v
void test1() {}

// fastcall, stdcall, vectorcall and regcall do not support variadic functions.
// CDECL: define void @_Z12testVariadicz
// FASTCALL: define void @_Z12testVariadicz
// STDCALL: define void @_Z12testVariadicz
// VECTORCALL: define void @_Z12testVariadicz
// REGCALL: define void @_Z12testVariadicz
void testVariadic(...){}

// ALL: define void @_Z5test2v
void __attribute__((cdecl)) test2() {}

// ALL: define x86_fastcallcc void @_Z5test3v
void __attribute__((fastcall)) test3() {}

// ALL: define x86_stdcallcc void @_Z5test4v
void __attribute__((stdcall)) test4() {}

// ALL: define  x86_vectorcallcc void @_Z5test5v
void __attribute__((vectorcall)) test5() {}

// ALL: define x86_regcallcc void @_Z17__regcall3__test6v
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

#ifdef MAIN
// ALL: define i32 @main
int main() {
  return 1;
}
#endif // main

#ifdef WMAIN
// WMAIN: define dso_local i32 @wmain
int wmain() {
  return 1;
}
#endif // wmain

#ifdef WINMAIN
// WINMAIN: define dso_local i32 @WinMain
int WinMain() {
  return 1;
}
#endif // WinMain

#ifdef WWINMAIN
// WWINMAIN: define dso_local i32 @wWinMain
int wWinMain() {
  return 1;
}
#endif // wWinMain

#ifdef DLLMAIN
// DLLMAIN: define dso_local i32 @DllMain
int DllMain() {
  return 1;
}
#endif // DllMain
