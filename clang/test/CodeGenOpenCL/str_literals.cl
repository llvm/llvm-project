// RUN: %clang_cc1 %s -cl-opt-disable -emit-llvm -o - -ffake-address-space-map | FileCheck %s

__constant char *__constant x = "hello world";
__constant char *__constant y = "hello world";

// CHECK: unnamed_addr addrspace(2) constant{{.*}}"hello world\00"
// CHECK-NOT: addrspace(2) unnamed_addr constant
// CHECK: @x = {{(dso_local )?}}addrspace(2) constant ptr addrspace(2)
// CHECK: @y = {{(dso_local )?}}addrspace(2) constant ptr addrspace(2)
// CHECK: unnamed_addr addrspace(2) constant{{.*}}"f\00"

void f() {
  //CHECK: store ptr addrspace(2) {{.*}}, ptr
  constant const char *f3 = __func__;
}
