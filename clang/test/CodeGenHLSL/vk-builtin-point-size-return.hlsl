// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry main -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// Same as vk-builtin-point-size.hlsl, but [[vk::builtin("PointSize")]]
// annotates the entry point's return value (the Function subject) rather
// than a struct field.

[[vk::builtin("PointSize")]]
[shader("vertex")]
float main([[vk::location(0)]] float4 a : A) : PSIZE {
  return a.x;
}

// CHECK: @PSIZE = {{.*}} addrspace(8) global float, !spirv.Decorations [[PSIZE_DECO:![0-9]+]]
// CHECK: define void @main()
// CHECK: [[PT_SIZE:%[0-9]+]] = call spir_func float @{{.*}}main{{.*}}
// CHECK-NEXT: store float [[PT_SIZE]], ptr addrspace(8) @PSIZE
// CHECK-DAG: [[PSIZE_DECO]] = !{[[PSIZE_OP:![0-9]+]]}
// CHECK-DAG: [[PSIZE_OP]] = !{i32 11, i32 1}
