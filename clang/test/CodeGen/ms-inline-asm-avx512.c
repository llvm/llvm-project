// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple x86_64-pc-windows-msvc -target-cpu skylake-avx512 -fasm-blocks -emit-llvm -o - | FileCheck %s

void t1() {
// CHECK: @t1
// CHECK: call void asm sideeffect inteldialect "vaddpd zmm8, zmm27, zmm6", "~{zmm8},~{dirflag},~{fpsr},~{flags}"()
// CHECK: ret void
  __asm {
	  vaddpd zmm8, zmm27, zmm6
  }
}


void t2() {
// CHECK: @t2
// CHECK: call void asm sideeffect inteldialect "vaddpd zmm8 {k1}, zmm27, zmm6", "~{zmm8},~{dirflag},~{fpsr},~{flags}"()
// CHECK: ret void
  __asm {
	  vaddpd zmm8 {k1}, zmm27, zmm6
  }
}

void ignore_fe_size() {
  // CHECK-LABEL: define dso_local void @ignore_fe_size()
  char c;
  // CHECK: vaddps xmm1, xmm2, $1{1to4}
  __asm vaddps xmm1, xmm2, [c]{1to4}
  // CHECK: vaddps xmm1, xmm2, $2
  __asm vaddps xmm1, xmm2, [c]
  // CHECK: mov eax, $3
  __asm mov eax, [c]
  // CHECK: mov $0, rax
  __asm mov [c], rax
}
