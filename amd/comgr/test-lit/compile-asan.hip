// COM: Run Comgr binary to compile HIP source with AddressSanitizer enabled
// RUN: compile-hip-asan %s %t.bin

// COM: Verify the kernel exists
// RUN: %llvm-objdump -d %t.bin | %FileCheck %s
// CHECK: <_Z3addPfS_S_>:
// CHECK: s_endpgm

// COM: Verify ASAN runtime symbols are present
// RUN: %llvm-objdump -t %t.bin | %FileCheck %s --check-prefix=ASAN
// ASAN: __asan_memcpy
// ASAN: __asan_memmove
// ASAN: __asan_load1
// ASAN: __asan_store1

__attribute__((global))
void add(float *A, float *B, float *C) {
    *C = *A + *B;
}
