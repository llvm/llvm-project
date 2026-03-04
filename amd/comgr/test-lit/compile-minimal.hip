// COM: Run Comgr binary to compile HIP source into LLVM IR Bitcode,
// COM: and then generate an executable
// RUN: compile-hip-minimal %s %t.bin

// COM: Disassemble
// RUN: %llvm-objdump -d %t.bin | %FileCheck %s
// CHECK: <_Z3addPfS_S_>:
// CHECK: s_endpgm

__attribute__((global))
void add(float *A, float *B, float *C) {
    *C = *A + *B;
}
