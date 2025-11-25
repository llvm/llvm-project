#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

#define TEST_SIZE 1000000
#define ITERATIONS 10

int main() {
    printf("SIMD XOR Performance Test\n");
    printf("========================\n");
    
    // Allocate memory
    uint8_t *data1 = malloc(TEST_SIZE);
    uint8_t *data2 = malloc(TEST_SIZE);
    uint8_t *result = malloc(TEST_SIZE);
    
    if (!data1 || !data2 || !result) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    // Initialize data
    for (int i = 0; i < TEST_SIZE; i++) {
        data1[i] = (uint8_t)(i & 0xFF);
        data2[i] = (uint8_t)((i * 2) & 0xFF);
    }
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Perform XOR operations (simple version)
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < TEST_SIZE; i++) {
            result[i] = data1[i] ^ data2[i];
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + 
                    (end.tv_nsec - start.tv_nsec) / 1e9;
    
    uint64_t total_ops = (uint64_t)TEST_SIZE * ITERATIONS;
    double ops_per_sec = total_ops / elapsed;
    
    printf("Test completed successfully\n");
    printf("Total operations: %lu\n", total_ops);
    printf("Elapsed time: %.3f seconds\n", elapsed);
    printf("Performance: %.0f operations/sec\n", ops_per_sec);
    
    // Verify result (simple check)
    int errors = 0;
    for (int i = 0; i < 100; i++) {
        if (result[i] != (data1[i] ^ data2[i])) {
            errors++;
        }
    }
    
    if (errors == 0) {
        printf("Verification: PASSED\n");
    } else {
        printf("Verification: FAILED (%d errors)\n", errors);
    }
    
    free(data1);
    free(data2);
    free(result);
    
    return 0;
}