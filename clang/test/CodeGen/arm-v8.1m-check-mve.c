// RUN: %clang --target=arm-none-eabi -mcpu=cortex-m85 -mfloat-abi=hard -O2 -save-temps=obj -S -o - %s | FileCheck %s
// RUN: %clang --target=arm-none-eabi -mcpu=cortex-m85 -mfloat-abi=hard -O2 -c -mthumb -save-temps=obj %s
// RUN: %clang --target=arm-none-eabi -mcpu=cortex-m55 -mfloat-abi=hard -O2 -c -mthumb -save-temps=obj %s

// REQUIRES: arm-registered-target

// CHECK: .fpu   fpv5-d16
// CHECK-NEXT  .arch_extension mve.fp

#define DUMMY_CONST_1 (0.0012345F)
#define DUMMY_CONST_2 (0.01F)
#define DUMMY_CONST_3 (0.02F)
#define DUMMY_CONST_4 (0.03F)
#define DUMMY_CONST_5 (0.04F)

typedef struct
{
    float a;
    float b;
    float c;
    float d;
} dummy_t;

// CHECK-LABEL: foo
// CHECK: vsub.f32        q0, q0, q1
// CHECK-NEXT: vfma.f32        q1, q0, q2

signed char foo(dummy_t *handle)
{
    handle->a += DUMMY_CONST_2 * (DUMMY_CONST_1 - handle->a);
    handle->b += DUMMY_CONST_3 * (DUMMY_CONST_1 - handle->b);
    handle->c += DUMMY_CONST_4 * (DUMMY_CONST_1 - handle->c);
    handle->d += DUMMY_CONST_5 * (DUMMY_CONST_1 - handle->d);
    return 0;
}
