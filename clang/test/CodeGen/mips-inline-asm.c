// REQUIRES: mips-registered-target
// RUN: %clang_cc1 -triple mips-linux-gnu -emit-llvm -o - %s | FileCheck %s

int data;

void m () {
  asm("lw $1, %0" :: "m"(data));
  // CHECK: call void asm sideeffect "lw $$1, $0", "*m,~{$1}"(i32* @data)
}

void ZC () {
  asm("ll $1, %0" :: "ZC"(data));
  // CHECK: call void asm sideeffect "ll $$1, $0", "*^ZC,~{$1}"(i32* @data)
}

void R () {
  asm("lw $1, %0" :: "R"(data));
  // CHECK: call void asm sideeffect "lw $$1, $0", "*R,~{$1}"(i32* @data)
}

int additionalClobberedRegisters () {
  int temp0;
  asm volatile(
                "mfhi %[temp0], $ac1 \n\t"
                  : [temp0]"=&r"(temp0)
                  :
                  : "memory", "t0", "t1", "$ac1hi", "$ac1lo", "$ac2hi", "$ac2lo", "$ac3hi", "$ac3lo"
  );
  return 0;
  // CHECK: call i32 asm sideeffect "mfhi $0, $$ac1 \0A\09", "=&r,~{memory},~{$8},~{$9},~{$ac1hi},~{$ac1lo},~{$ac2hi},~{$ac2lo},~{$ac3hi},~{$ac3lo},~{$1}"
}
