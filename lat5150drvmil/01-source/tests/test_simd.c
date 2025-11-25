
#define _GNU_SOURCE
#include <stdio.h>
#include <sched.h>
#include <signal.h>
#include <setjmp.h>
#include <immintrin.h>

static jmp_buf jmpbuf;

void sigill_handler(int sig) {
    longjmp(jmpbuf, 1);
}

int test_avx512(void) {
    signal(SIGILL, sigill_handler);
    if (setjmp(jmpbuf) == 0) {
        // Try AVX-512 - will SIGILL if not supported
        asm volatile("vpxord %%zmm0, %%zmm0, %%zmm0" ::: "zmm0");
        return 1;
    }
    return 0;
}

int test_avx2(void) {
    signal(SIGILL, sigill_handler);
    if (setjmp(jmpbuf) == 0) {
        // Try AVX2
        asm volatile("vpxor %%ymm0, %%ymm0, %%ymm0" ::: "ymm0");
        return 1;
    }
    return 0;
}

int main() {
    cpu_set_t cpuset;
    
    // Test AVX-512 on P-cores
    printf("Testing P-cores (0-11) for AVX-512:\n");
    int avx512_count = 0;
    for (int i = 0; i < 12; i++) {
        CPU_ZERO(&cpuset);
        CPU_SET(i, &cpuset);
        if (sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0) {
            if (test_avx512()) {
                printf("  Core %d: AVX-512 YES\n", i);
                avx512_count++;
            } else {
                printf("  Core %d: AVX-512 NO\n", i);
            }
        }
    }
    
    // Test AVX2
    printf("\nTesting AVX2: ");
    if (test_avx2()) {
        printf("YES\n");
    } else {
        printf("NO\n");
    }
    
    printf("\nSummary: %d cores with AVX-512\n", avx512_count);
    return 0;
}
