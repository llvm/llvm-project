// RUN: %clang_cc1 -triple i386-pc-linux -emit-llvm %s -o - | FileCheck -check-prefix GCABI %s
// RUN: %clang_cc1 -emit-llvm %s -o - -DMS_ABI -triple=i386-pc-win32 | FileCheck -check-prefix MSABI %s

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
// GCABI-LABEL: define void @_Z3foov()
// MSABI: define dso_local void @"?foo@@YAXXZ"

void __cdecl bar();
void bar();
void bar() {}
// GCABI-LABEL: define void @_Z3barv()
// MSABI: define dso_local void @"?bar@@YAXXZ"

// Test that it's OK to mark either the method declaration or method definition
// with a default CC explicitly.
class A {
public:
  void baz();
  void METHOD_CC qux();

  static void static_baz();
  static void __cdecl static_qux();
};

void METHOD_CC A::baz() {}
// GCABI-LABEL: define void @_ZN1A3bazEv
// MSABI: define dso_local x86_thiscallcc void @"?baz@A@@QAEXXZ"
void A::qux() {}
// GCABI-LABEL: define void @_ZN1A3quxEv
// MSABI: define dso_local x86_thiscallcc void @"?qux@A@@QAEXXZ"

void __cdecl static_baz() {}
// GCABI-LABEL: define void @_Z10static_bazv
// MSABI: define dso_local void @"?static_baz@@YAXXZ"
void static_qux() {}
// GCABI-LABEL: define void @_Z10static_quxv
// MSABI: define dso_local void @"?static_qux@@YAXXZ"

namespace PR31656 {
template <int I>
void __cdecl callee(int args[I]);
// GCABI-LABEL: declare void @_ZN7PR316566calleeILi1EEEvPi(
// MSABI: declare dso_local void @"??$callee@$00@PR31656@@YAXQAH@Z"(

void caller() { callee<1>(0); }
}
