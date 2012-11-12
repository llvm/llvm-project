// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck -check-prefix GCABI %s
// RUN: %clang_cc1 -emit-llvm %s -o - -DMS_ABI -cxx-abi microsoft -triple=i386-pc-win32 | FileCheck -check-prefix MSABI %s

#ifdef MS_ABI
# define METHOD_CC __thiscall
#else
# define METHOD_CC __attribute__ ((cdecl))
#endif

// Test that it's OK to have multiple function declarations with the default CC
// both mentioned explicitly and implied.
void foo();
void __cdecl foo();
void __cdecl foo() {}
// GCABI: define void @_Z3foov()
// MSABI: define void @"\01?foo@@YAXXZ"

void __cdecl bar();
void bar();
void bar() {}
// GCABI: define void @_Z3barv()
// MSABI: define void @"\01?bar@@YAXXZ"

// Test that it's OK to mark either the method declaration or method definition
// with a default CC explicitly.
class A {
public:
  void baz();
  void METHOD_CC qux();

  void static_baz();
  void __cdecl static_qux();
};

void METHOD_CC A::baz() {}
// GCABI: define void @_ZN1A3bazEv
// MSABI: define x86_thiscallcc void @"\01?baz@A@@QAEXXZ"
void A::qux() {}
// GCABI: define void @_ZN1A3quxEv
// MSABI: define x86_thiscallcc void @"\01?qux@A@@QAEXXZ"

void __cdecl static_baz() {}
// GCABI: define void @_Z10static_bazv
// MSABI: define void @"\01?static_baz@@YAXXZ"
void static_qux() {}
// GCABI: define void @_Z10static_quxv
// MSABI: define void @"\01?static_qux@@YAXXZ"
