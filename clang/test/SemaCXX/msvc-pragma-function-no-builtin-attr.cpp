// RUN: %clang_cl -fms-compatibility -Xclang -ast-dump -fsyntax-only -- %s | FileCheck %s

extern "C" __inline float __cdecl fabsf(  float _X);
// CHECK: FunctionDecl {{.*}} fabsf
#pragma function(fabsf)
  __inline float __cdecl fabsf(  float _X)
{
    return 0;
}
// CHECK: FunctionDecl {{.*}} fabsf
// CHECK: NoBuiltinAttr {{.*}} <<invalid sloc>> Implicit fabsf

int bar() {
  return 0;
}
// CHECK: FunctionDecl {{.*}} bar
// CHECK: NoBuiltinAttr {{.*}} <<invalid sloc>> Implicit fabsf

struct A {
    int foo() = delete;
    // CHECK: CXXMethodDecl {{.*}} foo {{.*}} delete
    // CHECK-NOT: NoBuiltinAttr
    A() = default;
    // CHECK: CXXConstructorDecl {{.*}} A {{.*}} default
    // CHECK-NOT: NoBuiltinAttr
};

int main() {
    return 0;
}
// CHECK: FunctionDecl {{.*}} main
// CHECK: NoBuiltinAttr {{.*}} <<invalid sloc>> Implicit fabsf
