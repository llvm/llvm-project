// REQUIRES: x86-registered-target
// Check the constraint "*m" of operand arr and the definition of arr is not removed by FE
// RUN: %clang_cc1 %s -fasm-blocks -triple i386-apple-darwin10 -emit-llvm -o - | FileCheck %s

static int arr[10];
void t1() {
  // CHECK: @arr = internal global [10 x i32]
  // CHECK: call void asm sideeffect inteldialect "mov dword ptr arr[edx * $$4],edx", "=*m,{{.*}}([10 x i32]* @arr)
  __asm mov  dword ptr arr[edx*4],edx
}
