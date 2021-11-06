// RUN: %clang_cc1 -triple i386-pc-win32 -emit-llvm %s -o - | FileCheck %s

// PR15768

// A trivial 20 byte struct is returned indirectly and taken as byval.
struct S {
  S();
  int a, b, c, d, e;
};

struct C {
  S variadic_sret(const char *f, ...);
  S __cdecl cdecl_sret();
  S __cdecl byval_and_sret(S a);
  int c;
};

S C::variadic_sret(const char *f, ...) { return S(); }
S C::cdecl_sret() { return S(); }
S C::byval_and_sret(S a) { return S(); }

// CHECK: define dso_local void @"?variadic_sret@C@@QAA?AUS@@PBDZZ"(%struct.C* {{[^,]*}} %this, %struct.S* noalias sret(%struct.S) align 4 %agg.result, i8* noundef %f, ...)
// CHECK: define dso_local void @"?cdecl_sret@C@@QAA?AUS@@XZ"(%struct.C* {{[^,]*}} %this, %struct.S* noalias sret(%struct.S) align 4 %agg.result)
// CHECK: define dso_local void @"?byval_and_sret@C@@QAA?AUS@@U2@@Z"(%struct.C* {{[^,]*}} %this, %struct.S* noalias sret(%struct.S) align 4 %agg.result, %struct.S* noundef byval(%struct.S) align 4 %a)

int main() {
  C c;
  c.variadic_sret("asdf");
  c.cdecl_sret();
  c.byval_and_sret(S());
}
// CHECK-LABEL: define dso_local noundef i32 @main()
// CHECK: call void {{.*}} @"?variadic_sret@C@@QAA?AUS@@PBDZZ"
// CHECK: call void @"?cdecl_sret@C@@QAA?AUS@@XZ"
// CHECK: call void @"?byval_and_sret@C@@QAA?AUS@@U2@@Z"

// __fastcall has similar issues.
struct A {
  S __fastcall f(int x);
};
S A::f(int x) {
  return S();
}
// CHECK-LABEL: define dso_local x86_fastcallcc void @"?f@A@@QAI?AUS@@H@Z"(%struct.A* inreg noundef %this, %struct.S* inreg noalias sret(%struct.S) align 4 %agg.result, i32 noundef %x)
