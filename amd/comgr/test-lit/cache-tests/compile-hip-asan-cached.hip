// COM: Compile a HIP source with AddressSanitizer enabled while the Comgr
// COM: command cache is active.

// COM: Start from a clean cache and enable it for this test.
// RUN: rm -fr %t.cache
// RUN: export AMD_COMGR_CACHE=1
// RUN: export AMD_COMGR_CACHE_DIR=%t.cache

// COM: Compile with the cache enabled.
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

// COM: The cache directory should have been populated.
// RUN: [ -d %t.cache ]

__attribute__((global))
void add(float *A, float *B, float *C) {
    *C = *A + *B;
}
