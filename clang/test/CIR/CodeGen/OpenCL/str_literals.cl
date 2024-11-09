// RUN: %clang_cc1 %s -fclangir -triple=spirv64-unknown-unknown -cl-opt-disable -emit-cir -o %t.cir -ffake-address-space-map
// RUN: FileCheck -input-file=%t.cir -check-prefix=CIR %s
// RUN: %clang_cc1 %s -fclangir -triple=spirv64-unknown-unknown -cl-opt-disable -emit-llvm -o %t.ll -ffake-address-space-map
// RUN: FileCheck -input-file=%t.ll -check-prefix=LLVM %s

__constant char *__constant x = "hello world";
__constant char *__constant y = "hello world";

// CIR: cir.global{{.*}} constant {{.*}}addrspace(offload_constant) @".str" = #cir.const_array<"hello world\00" : !cir.array<!s8i x 12>> : !cir.array<!s8i x 12>
// CIR: cir.global{{.*}} constant {{.*}}addrspace(offload_constant) @x = #cir.global_view<@".str"> : !cir.ptr<!s8i, addrspace(offload_constant)>
// CIR: cir.global{{.*}} constant {{.*}}addrspace(offload_constant) @y = #cir.global_view<@".str"> : !cir.ptr<!s8i, addrspace(offload_constant)>
// CIR: cir.global{{.*}} constant {{.*}}addrspace(offload_constant) @".str.1" = #cir.const_array<"f\00" : !cir.array<!s8i x 2>> : !cir.array<!s8i x 2>
// LLVM: addrspace(2) constant{{.*}}"hello world\00"
// LLVM-NOT: addrspace(2) constant
// LLVM: @x = {{(dso_local )?}}addrspace(2) constant ptr addrspace(2)
// LLVM: @y = {{(dso_local )?}}addrspace(2) constant ptr addrspace(2)
// LLVM: addrspace(2) constant{{.*}}"f\00"

void f() {
  // CIR: cir.store %{{.*}}, %{{.*}} : !cir.ptr<!s8i, addrspace(offload_constant)>, !cir.ptr<!cir.ptr<!s8i, addrspace(offload_constant)>, addrspace(offload_private)>
  // LLVM: store ptr addrspace(2) {{.*}}, ptr
  constant const char *f3 = __func__;
}
