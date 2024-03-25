// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -fdefault-calling-conv=cdecl -emit-llvm -o - %s | FileCheck %s --check-prefix=CDECL --check-prefix=X86 --check-prefix=ALL
// RUN: %clang_cc1 -triple i786-unknown-linux-gnu -target-feature +sse4.2 -fdefault-calling-conv=fastcall -emit-llvm -o - %s | FileCheck %s --check-prefix=FASTCALL --check-prefix=X86 --check-prefix=ALL
// RUN: %clang_cc1 -triple i486-unknown-linux-gnu -fdefault-calling-conv=stdcall -emit-llvm -o - %s | FileCheck %s --check-prefix=STDCALL --check-prefix=X86 --check-prefix=ALL
// RUN: %clang_cc1 -triple i486-unknown-linux-gnu -mrtd -emit-llvm -o - %s | FileCheck %s --check-prefix=STDCALL --check-prefix=X86 --check-prefix=ALL
// RUN: %clang_cc1 -triple i986-unknown-linux-gnu -fdefault-calling-conv=vectorcall -emit-llvm -o - %s | FileCheck %s --check-prefix=VECTORCALL --check-prefix=X86 --check-prefix=ALL
// RUN: %clang_cc1 -triple i986-unknown-linux-gnu -fdefault-calling-conv=regcall -emit-llvm -o - %s | FileCheck %s --check-prefix=REGCALL --check-prefix=X86 --check-prefix=ALL
// RUN: %clang_cc1 -triple i686-pc-win32 -fdefault-calling-conv=vectorcall -emit-llvm -o - %s -DWINDOWS | FileCheck %s --check-prefix=WIN32
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fdefault-calling-conv=vectorcall -emit-llvm -o - %s -DWINDOWS | FileCheck %s --check-prefix=WIN64
// RUN: %clang_cc1 -triple i686-pc-win32 -emit-llvm -o - %s -DEXPLICITCC | FileCheck %s --check-prefix=EXPLICITCC
// RUN: %clang_cc1 -triple m68k-unknown-linux-gnu -mrtd -emit-llvm -o - %s | FileCheck %s --check-prefix=RTDCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple m68k-unknown-linux-gnu -fdefault-calling-conv=rtdcall -emit-llvm -o - %s | FileCheck %s --check-prefix=RTDCALL --check-prefix=ALL

// CDECL: define{{.*}} void @_Z5test1v
// FASTCALL: define{{.*}} x86_fastcallcc void @_Z5test1v
// STDCALL: define{{.*}} x86_stdcallcc void @_Z5test1v
// VECTORCALL: define{{.*}} x86_vectorcallcc void @_Z5test1v
// REGCALL: define{{.*}} x86_regcallcc void @_Z17__regcall3__test1v
// RTDCALL: define{{.*}} m68k_rtdcc void @_Z5test1v
void test1() {}

// fastcall, stdcall, vectorcall, regcall and m68k_rtd do not support variadic functions.
// CDECL: define{{.*}} void @_Z12testVariadicz
// FASTCALL: define{{.*}} void @_Z12testVariadicz
// STDCALL: define{{.*}} void @_Z12testVariadicz
// VECTORCALL: define{{.*}} void @_Z12testVariadicz
// REGCALL: define{{.*}} void @_Z12testVariadicz
// RTDCALL: define{{.*}} void @_Z12testVariadicz
void testVariadic(...){}

// X86: define{{.*}} void @_Z5test2v
void __attribute__((cdecl)) test2() {}

// X86: define{{.*}} x86_fastcallcc void @_Z5test3v
void __attribute__((fastcall)) test3() {}

// X86: define{{.*}} x86_stdcallcc void @_Z5test4v
void __attribute__((stdcall)) test4() {}

// X86: define{{.*}} x86_vectorcallcc void @_Z5test5v
void __attribute__((vectorcall)) test5() {}

// X86: define{{.*}} x86_regcallcc void @_Z17__regcall3__test6v
void __attribute__((regcall)) test6() {}

// RTDCALL: define{{.*}} m68k_rtdcc void @_Z5test7v
void __attribute__((m68k_rtd)) test7() {}

// ALL: define linkonce_odr void @_ZN1A11test_memberEv
class A {
public:
  void test_member() {}
};

void test() {
  A a;
  a.test_member();

// ALL: define internal void @"_ZZ{{.*}}testvENK3$_0clEi"
  auto f = [](int b) {};
  f(87);
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
