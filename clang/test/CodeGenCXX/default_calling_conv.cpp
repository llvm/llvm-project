// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -fdefault-calling-conv=cdecl -emit-llvm -o - %s | FileCheck %s --check-prefix=CDECL --check-prefix=ALL
// RUN: %clang_cc1 -triple i786-unknown-linux-gnu -target-feature +sse4.2 -fdefault-calling-conv=fastcall -emit-llvm -o - %s | FileCheck %s --check-prefix=FASTCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple i486-unknown-linux-gnu -fdefault-calling-conv=stdcall -emit-llvm -o - %s | FileCheck %s --check-prefix=STDCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple i486-unknown-linux-gnu -mrtd -emit-llvm -o - %s | FileCheck %s --check-prefix=STDCALL --check-prefix=ALL
// RUN: %clang_cc1 -triple i986-unknown-linux-gnu -fdefault-calling-conv=vectorcall -emit-llvm -o - %s | FileCheck %s --check-prefix=VECTORCALL --check-prefix=ALL

// CDECL: define void @_Z5test1v
// FASTCALL: define x86_fastcallcc void @_Z5test1v
// STDCALL: define x86_stdcallcc void @_Z5test1v
// VECTORCALL: define x86_vectorcallcc void @_Z5test1v
void test1() {}

// fastcall, stdcall, and vectorcall all do not support variadic functions.
// CDECL: define void @_Z12testVariadicz
// FASTCALL: define void @_Z12testVariadicz
// STDCALL: define void @_Z12testVariadicz
// VECTORCALL: define void @_Z12testVariadicz
void testVariadic(...){}

// ALL: define void @_Z5test2v
void __attribute__((cdecl)) test2() {}

// ALL: define x86_fastcallcc void @_Z5test3v
void __attribute__((fastcall)) test3() {}

// ALL: define x86_stdcallcc void @_Z5test4v
void __attribute__((stdcall)) test4() {}

// ALL: define  x86_vectorcallcc void @_Z5test5v
void __attribute__((vectorcall)) test5() {}

// ALL: define linkonce_odr void @_ZN1A11test_memberEv
class A {
public:
  void test_member() {}
};

void test() {
  A a;
  a.test_member();
}
