// RUN: %clang %s -pie -fPIE -o %t && %run %t
// REQUIRES: x86_64-target-arch

// FIXME: Fails Asan, as expected, with 5lvl page tables.
// UNSUPPORTED: x86_64-target-arch

#include <assert.h>
#include <stdio.h>
#include <sys/mman.h>

int main() {
    for (int j = 0; j < 1024; j++) {
        // Try 1TB offsets. This attempts to find memory addresses where the
        // shadow mappings - which assume a 47-bit address space - are invalid.
        unsigned long long target = (1ULL << 56) - (2 * 4096) - (j * (1ULL << 40));

        // Since we don't use MAP_FIXED, mmap might return an address that is
        // lower in the address space (due to sanitizer and/or kernel limits).
        // That is fine - if the app is also restricted from making high
        // allocations, then they are safe.
        char* ptr = (char*) mmap ((void*) target, 4096, PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        printf ("Allocated at %p\n", ptr);

        assert (ptr != MAP_FAILED);
        for (int i = 0; i < 100; i++) {
            ptr [i] = 0;
        }
        munmap (ptr, 4096);
    }

    return 0;
}
