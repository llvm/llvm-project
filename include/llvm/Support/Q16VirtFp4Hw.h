// Q16-Based Virtual FP4/MXFP4 Hardware Implementation
// Uses Q15.16 fixed-point math for improved precision
// Implements E2M1 and E3M0 formats with Q16 representation

#ifndef Q16_VIRTUAL_FP4_HARDWARE_H
#define Q16_VIRTUAL_FP4_HARDWARE_H

#include <stdint.h>
#include <stdbool.h>

// Define Q16 fixed-point type (15 bits integer, 16 bits fraction, 1 sign bit)
typedef int32_t q16;

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

// MXFP4: INT4 with Q16 block scaling
typedef struct {
    uint8_t data : 4;     // 4-bit integer value
    q16 scale;            // Q16 scale factor
} MXFP4_Q16;

// Q16 conversion functions
static inline q16 float_to_q16(float val);
static inline float q16_to_float(q16 val);

// Efficient conversion functions using Q16 representation
static inline q16 fp4_e2m1_to_q16(FP4_E2M1 val);
static inline FP4_E2M1 q16_to_fp4_e2m1(q16 val);
static inline q16 fp4_e3m0_to_q16(FP4_E3M0 val);
static inline FP4_E3M0 q16_to_fp4_e3m0(q16 val);

// Arithmetic operations using Q16 math
static inline q16 q16_add(q16 a, q16 b);
static inline q16 q16_mul(q16 a, q16 b);

// Efficient MXFP4 operations with Q16
MXFP4_Q16 q16_vmxfp4_quantize(float input, float scale_factor);
float q16_vmxfp4_dequantize(MXFP4_Q16 input);

// Matrix operations with Q16
void q16_vmxfp4_matrix_multiply(
    const MXFP4_Q16* A, const MXFP4_Q16* B, MXFP4_Q16* C,
    int M, int N, int K,
    const q16* scale_A, const q16* scale_B);

// Performance counters
void reset_q16_performance_counters();
uint64_t get_q16_fp4_ops();
uint64_t get_q16_mxfp4_ops();

#endif // Q16_VIRTUAL_FP4_HARDWARE_H