// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -fdefault-calling-conv=cdecl -emit-llvm -o - %s | FileCheck %s --check-prefix=CDECL --check-prefix=ALL
// RUN: %clang_cc1 -triple i786-unknown-linux-gnu -target-feature +sse4.2 -fdefault-calling-conv=fastcall -emit-llvm -o - %s | FileCheck %s --check-prefix=FASTCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple i486-unknown-linux-gnu -fdefault-calling-conv=stdcall -emit-llvm -o - %s | FileCheck %s --check-prefix=STDCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple i486-unknown-linux-gnu -mrtd -emit-llvm -o - %s | FileCheck %s --check-prefix=STDCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple i986-unknown-linux-gnu -fdefault-calling-conv=vectorcall -emit-llvm -o - %s | FileCheck %s --check-prefix=VECTORCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple i986-unknown-linux-gnu -fdefault-calling-conv=regcall -emit-llvm -o - %s | FileCheck %s --check-prefix=REGCALL --check-prefix=ALL

// CDECL: define dso_local void @_Z5test1v
// FASTCALL: define dso_local x86_fastcallcc void @_Z5test1v
// STDCALL: define dso_local x86_stdcallcc void @_Z5test1v
// VECTORCALL: define dso_local x86_vectorcallcc void @_Z5test1v
// REGCALL: define dso_local x86_regcallcc void @_Z17__regcall3__test1v
void test1() {}

// fastcall, stdcall, vectorcall and regcall do not support variadic functions.
// CDECL: define dso_local void @_Z12testVariadicz
// FASTCALL: define dso_local void @_Z12testVariadicz
// STDCALL: define dso_local void @_Z12testVariadicz
// VECTORCALL: define dso_local void @_Z12testVariadicz
// REGCALL: define dso_local void @_Z12testVariadicz
void testVariadic(...){}

// ALL: define dso_local void @_Z5test2v
void __attribute__((cdecl)) test2() {}

// ALL: define dso_local x86_fastcallcc void @_Z5test3v
void __attribute__((fastcall)) test3() {}

// ALL: define dso_local x86_stdcallcc void @_Z5test4v
void __attribute__((stdcall)) test4() {}

// ALL: define dso_local  x86_vectorcallcc void @_Z5test5v
void __attribute__((vectorcall)) test5() {}

// ALL: define dso_local x86_regcallcc void @_Z17__regcall3__test6v
void __attribute__((regcall)) test6() {}

// ALL: define linkonce_odr dso_local void @_ZN1A11test_memberEv
class A {
public:
  void test_member() {}
};

void test() {
  A a;
  a.test_member();
}

// ALL: define dso_local i32 @main
int main() {
  return 1;
}
