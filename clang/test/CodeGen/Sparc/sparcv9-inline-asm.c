// RUN: %clang_cc1 -triple sparcv9-unknown-unknown -emit-llvm %s -o - | FileCheck %s

void test_gcc_registers(void) {
    register unsigned int regO6 asm("o6") = 0;
    register unsigned int regSP asm("sp") = 1;
    register unsigned int reg14 asm("r14") = 2;
    register unsigned int regI6 asm("i6") = 3;
    register unsigned int regFP asm("fp") = 4;
    register unsigned int reg30 asm("r30") = 5;

    register float fF20 asm("f20") = 8.0;
    register double dF40 asm("f40") = 11.0;
    register long double qF40 asm("f40") = 14.0;

    // Test remapping register names in register ... asm("rN") statments.

    // CHECK: call void asm sideeffect "add $0,$1,$2", "{r14},{r14},{r14}"
    asm volatile("add %0,%1,%2" : : "r" (regO6), "r" (regSP), "r" (reg14));

    // CHECK: call void asm sideeffect "add $0,$1,$2", "{r30},{r30},{r30}"
    asm volatile("add %0,%1,%2" : : "r" (regI6), "r" (regFP), "r" (reg30));

    // CHECK: call void asm sideeffect "fadds $0,$1,$2", "{f20},{f20},{f20}"
    asm volatile("fadds %0,%1,%2" : : "f" (fF20), "f" (fF20), "f"(fF20));

    // CHECK: call void asm sideeffect "faddd $0,$1,$2", "{f40},{f40},{f40}"
    asm volatile("faddd %0,%1,%2" : : "f" (dF40), "f" (dF40), "f"(dF40));

    // CHECK: call void asm sideeffect "faddq $0,$1,$2", "{f40},{f40},{f40}"
    asm volatile("faddq %0,%1,%2" : : "f" (qF40), "f" (qF40), "f"(qF40));

}
