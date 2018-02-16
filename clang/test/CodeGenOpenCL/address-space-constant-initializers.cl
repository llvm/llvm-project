// RUN: %clang_cc1 %s -ffake-address-space-map -emit-llvm -o - | FileCheck -check-prefix=FAKE %s
// RUN: %clang_cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm -o - | FileCheck -check-prefix=AMDGCN %s

typedef struct {
    int i;
    float f; // At non-zero offset.
} ArrayStruct;

__constant ArrayStruct constant_array_struct = { 0, 0.0f };

typedef struct {
    __constant float* constant_float_ptr;
} ConstantArrayPointerStruct;

// FAKE: %struct.ConstantArrayPointerStruct = type { float addrspace(2)* }
// FAKE: addrspace(2) constant %struct.ConstantArrayPointerStruct { float addrspace(2)* bitcast (i8 addrspace(2)* getelementptr (i8, i8 addrspace(2)* bitcast (%struct.ArrayStruct addrspace(2)* @constant_array_struct to i8 addrspace(2)*), i64 4) to float addrspace(2)*) }
// AMDGCN: %struct.ConstantArrayPointerStruct = type { float addrspace(4)* }
// AMDGCN: addrspace(4) constant %struct.ConstantArrayPointerStruct { float addrspace(4)* bitcast (i8 addrspace(4)* getelementptr (i8, i8 addrspace(4)* bitcast (%struct.ArrayStruct addrspace(4)* @constant_array_struct to i8 addrspace(4)*), i64 4) to float addrspace(4)*) }
// Bug  18567
__constant ConstantArrayPointerStruct constant_array_pointer_struct = {
    &constant_array_struct.f
};

__kernel void initializer_cast_is_valid_crash()
{
  unsigned char v512[64] = {
      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x02,0x00
  };

}
