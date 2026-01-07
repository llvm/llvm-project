// RUN: %clangxx_asan -O0 %s -o %t && %run %t

// REQUIRES: system-aix

#include <sys/mman.h>
#include <errno.h>
#include <stdio.h>
#include <assert.h>

#define ASAN_AIX_SHADOW_OFFSET 0x0a01000000000000ULL

int main() {
    size_t map_size = 4096;
    void* addr = (void*)ASAN_AIX_SHADOW_OFFSET;
    
    // Attempt to map memory directly on top of the Shadow Memory
    void* ptr = mmap(addr, map_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);

    // We expect this to FAIL because it overlaps shadow memory
    if (ptr != MAP_FAILED) {
        fprintf(stderr, "TEST FAILED: mmap should have failed but returned %p\n", ptr);
        munmap(ptr, map_size);
        return 1;
    }

    // We expect errno to be EINVAL (Invalid Argument)
    if (errno != EINVAL) {
        fprintf(stderr, "TEST FAILED: Expected errno=EINVAL (%d), got %d\n", EINVAL, errno);
        return 1;
    }

    printf("TEST PASSED: mmap failed as expected.\n");
    return 0;
}