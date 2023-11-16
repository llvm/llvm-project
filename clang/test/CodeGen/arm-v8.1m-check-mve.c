// RUN: %clang --target=arm-none-eabi -mcpu=cortex-m85 -mfloat-abi=hard -O2 -save-temps=obj -S -o - %s | FileCheck %s
// RUN: %clang --target=arm-none-eabi -mcpu=cortex-m55 -mfloat-abi=hard -O2 -save-temps=obj -S -o - %s | FileCheck %s

// The below tests are to make sure that assembly directives do not lose mve feature so that reassembly works with
// mve floating point instructions.
// RUN: %clang --target=arm-none-eabi -mcpu=cortex-m85 -mfloat-abi=hard -O2 -c -mthumb -save-temps=obj %s
// RUN: %clang --target=arm-none-eabi -mcpu=cortex-m55 -mfloat-abi=hard -O2 -c -mthumb -save-temps=obj %s

// REQUIRES: arm-registered-target

// CHECK: .fpu   fpv5-d16
// CHECK-NEXT  .arch_extension mve.fp

#define DUMMY_CONST_1 (0.0012345F)

typedef struct
{
    float a;
    float b;
    float c;
    float d;
} dummy_t;

// CHECK-LABEL: foo
// CHECK: vsub.f32
// CHECK: vfma.f32

signed char foo(dummy_t *handle)
{
    handle->a += 0.01F * (DUMMY_CONST_1 - handle->a);
    handle->b += 0.02F * (DUMMY_CONST_1 - handle->b);
    handle->c += 0.03F * (DUMMY_CONST_1 - handle->c);
    handle->d += 0.04F * (DUMMY_CONST_1 - handle->d);
    return 0;
}
