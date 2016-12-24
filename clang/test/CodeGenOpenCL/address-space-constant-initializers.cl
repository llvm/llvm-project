// RUN: %clang_cc1 %s -ffake-address-space-map -emit-llvm -o - | FileCheck %s

typedef struct {
    int i;
    float f; // At non-zero offset.
} ArrayStruct;

__constant ArrayStruct constant_array_struct = { 0, 0.0f };

typedef struct {
    __constant float* constant_float_ptr;
} ConstantArrayPointerStruct;

// CHECK: %struct.ConstantArrayPointerStruct = type { float addrspace(2)* }
// CHECK: addrspace(2) constant %struct.ConstantArrayPointerStruct { float addrspace(2)* bitcast (i8 addrspace(2)* getelementptr (i8, i8 addrspace(2)* bitcast (%struct.ArrayStruct addrspace(2)* @constant_array_struct to i8 addrspace(2)*), i64 4) to float addrspace(2)*) }
// Bug  18567
__constant ConstantArrayPointerStruct constant_array_pointer_struct = {
    &constant_array_struct.f
};

