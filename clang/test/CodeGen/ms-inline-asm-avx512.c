// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple x86_64-pc-windows-msvc -target-cpu knl -fasm-blocks -emit-llvm -o - | FileCheck %s

void t1() {
// CHECK: @t1
// CHECK: call void asm sideeffect inteldialect "vaddpd zmm8, zmm27, zmm6", "~{zmm8},~{dirflag},~{fpsr},~{flags}"()
// CHECK: ret void
  __asm {
	  vaddpd zmm8, zmm27, zmm6
  }
}
