// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=clc++ -emit-llvm -O0 -o - | FileCheck %s

void test_reinterpret_cast(){
__private float x;
__private float& y = x; 
// We don't need bitcast to cast pointer type and
// address space at the same time.
//CHECK: addrspacecast ptr %x to ptr addrspace(4)
//CHECK: [[REG:%[0-9]+]] = load ptr, ptr %y
//CHECK: addrspacecast ptr [[REG]] to ptr addrspace(4)
//CHECK-NOT: bitcast 
__generic int& rc1 = reinterpret_cast<__generic int&>(x);
__generic int& rc2 = reinterpret_cast<__generic int&>(y);
}
