// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -ast-dump -o - %s | FileCheck %s

// CHECK: VarDecl {{.*}} global_scalar 'hlsl_private int' static cinit
static int global_scalar = 0;

// CHECK: VarDecl {{.*}} global_buffer 'RWBuffer<float>':'hlsl::RWBuffer<float>' callinit
RWBuffer<float> global_buffer;

class A {
// CHECK: VarDecl {{.*}} a 'hlsl_private int' static
  static int a;
};

class B {
// CHECK: VarDecl {{.*}} b 'hlsl_private int' static
  static int b;
};

// CHECK: VarDecl {{.*}} b 'hlsl_private int' cinit
int B::b = 0;

export void foo() {
// CHECK: VarDecl {{.*}} local_buffer 'RWBuffer<float>':'hlsl::RWBuffer<float>' cinit
  RWBuffer<float> local_buffer = global_buffer;

// CHECK: VarDecl {{.*}} static_local_buffer 'RWBuffer<float>':'hlsl::RWBuffer<float>' static cinit
  static RWBuffer<float> static_local_buffer = global_buffer;

// CHECK: VarDecl {{.*}} local_scalar 'int' cinit
  int local_scalar = global_scalar;

// CHECK: VarDecl {{.*}} static_scalar 'hlsl_private int' static cinit
  static int static_scalar = 0;
}
