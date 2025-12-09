// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// REQUIRES: system-aix
#include <sys/mman.h>

#define ASAN_AIX_SHADOW_OFFSET 0x0a01000000000000ULL

int main() {
    size_t map_size = 4096;
    void* addr = (void*)ASAN_AIX_SHADOW_OFFSET;
    void* ptr = mmap(addr, map_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (ptr != MAP_FAILED) munmap(ptr, map_size);
    return 0;
}

// CHECK: ERROR: AddressSanitizer: mmap requested memory range
// CHECK: overlaps with ASan shadow memory
