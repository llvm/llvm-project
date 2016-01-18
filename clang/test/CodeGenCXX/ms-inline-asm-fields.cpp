// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple i386-apple-darwin10 -fasm-blocks -emit-llvm -o - | FileCheck %s

struct A {
  int a1;
  int a2;
  struct B {
    int b1;
    int b2;
  } a3;
};

namespace asdf {
A a_global;
}

extern "C" int test_param_field(A p) {
// CHECK: define i32 @test_param_field(%struct.A* byval align 4 %p)
// CHECK: getelementptr inbounds %struct.A, %struct.A* %p, i32 0, i32 0
// CHECK: call i32 asm sideeffect inteldialect "mov eax, dword ptr $1"
// CHECK: ret i32
  __asm mov eax, p.a1
}

extern "C" int test_namespace_global() {
// CHECK: define i32 @test_namespace_global()
// CHECK: call i32 asm sideeffect inteldialect "mov eax, dword ptr $1", "{{.*}}"(i32* getelementptr inbounds (%struct.A, %struct.A* @_ZN4asdf8a_globalE, i32 0, i32 2, i32 1))
// CHECK: ret i32
  __asm mov eax, asdf::a_global.a3.b2
}

