// Optimized Virtual FP4/MXFP4 Hardware Implementation
// Optimized for E2M1 and E3M0 formats with reduced overhead

#ifndef OPT_VIRTUAL_FP4_HARDWARE_H
#define OPT_VIRTUAL_FP4_HARDWARE_H

#include <stdint.h>
#include <stdbool.h>

// FP4 E2M1 format: 1 sign, 2 exponent, 1 mantissa
// Bit layout: [sign:1][exp:2][mantissa:1]
typedef union {
    uint8_t data : 4;
    struct {
        uint8_t mantissa : 1;  // 0 or 1
        uint8_t exp : 2;       // 0-3
        uint8_t sign : 1;      // 0 or 1
    } e2m1;
} FP4_E2M1;

// FP4 E3M0 format: 1 sign, 3 exponent, 0 mantissa
// Bit layout: [sign:1][exp:3]
typedef union {
    uint8_t data : 4;
    struct {
        uint8_t unused : 0;    // no mantissa
        uint8_t exp : 3;       // 0-7
        uint8_t sign : 1;      // 0 or 1
    } e3m0;
} FP4_E3M0;

// MXFP4: INT4 with block scaling
typedef struct {
    uint8_t data : 4;     // 4-bit integer value
    uint8_t scale_exp;    // 8-bit scale exponent (UE8M0 format)
} MXFP4;

// Optimized lookup tables for faster conversion
extern float e2m1_lookup_table[16];
extern float e3m0_lookup_table[16];

// Initialization function
bool init_optimized_virtual_fp4_hw();

// Optimized conversion functions
static inline float fp4_e2m1_to_float(FP4_E2M1 val);
static inline FP4_E2M1 float_to_fp4_e2m1(float val);
static inline float fp4_e3m0_to_float(FP4_E3M0 val);
static inline FP4_E3M0 float_to_fp4_e3m0(float val);

// Optimized arithmetic operations
static inline FP4_E2M1 fp4_e2m1_add(FP4_E2M1 a, FP4_E2M1 b);
static inline FP4_E2M1 fp4_e2m1_mul(FP4_E2M1 a, FP4_E2M1 b);
static inline FP4_E3M0 fp4_e3m0_add(FP4_E3M0 a, FP4_E3M0 b);
static inline FP4_E3M0 fp4_e3m0_mul(FP4_E3M0 a, FP4_E3M0 b);

// Optimized MXFP4 operations
MXFP4 opt_vmxfp4_quantize(float input, uint8_t block_scale);
float opt_vmxfp4_dequantize(MXFP4 input);

// Optimized matrix operations
void opt_vmxfp4_matrix_multiply(
    const MXFP4* A, const MXFP4* B, MXFP4* C,
    int M, int N, int K,
    const uint8_t* scale_A, const uint8_t* scale_B);

// Performance counters
void reset_opt_performance_counters();
uint64_t get_opt_fp4_ops();
uint64_t get_opt_mxfp4_ops();

#endif // OPT_VIRTUAL_FP4_HARDWARE_H