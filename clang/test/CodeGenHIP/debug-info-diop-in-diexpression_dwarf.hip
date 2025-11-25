// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -x hip -triple amdgcn-amd-amdhsa -emit-obj -fcuda-is-device -debug-info-kind=limited -gheterogeneous-dwarf -o - %s | llvm-dwarfdump --diff - | FileCheck --check-prefixes=CHECK %s
// RUN: %clang_cc1 -x hip -triple amdgcn-amd-amdhsa -emit-obj -fcuda-is-device -debug-info-kind=limited -gheterogeneous-dwarf -mllvm -stop-after=amdgpu-isel -o - %s | llc -x mir -verify-machineinstrs -start-after=amdgpu-isel -filetype=obj -o - - | llvm-dwarfdump --diff - | FileCheck --check-prefixes=CHECK %s

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

__shared__ int GlobalSharedA;
__shared__ int GlobalSharedB;
__device__ int GlobalDeviceA;
__device__ int GlobalDeviceB;
__constant__ int GlobalConstantA;
__constant__ int GlobalConstantB;

__global__ void kernel1(int Arg) {
  __shared__ int KernelVarSharedA;
  __shared__ int KernelVarSharedB;
  int KernelVarA;
  int KernelVarB;

  auto *KernelVarSharedAPointer = &KernelVarSharedA;
  auto *KernelVarSharedBPointer = &KernelVarSharedB;
  auto *KernelVarAPointer = &KernelVarA;
  auto *KernelVarBPointer = &KernelVarB;
}

__device__ void func1(int Arg) {
  int FuncVarA;
  int FuncVarB;

  auto *FuncVarAPointer = &FuncVarA;
  auto *FuncVarBPointer = &FuncVarB;
}


// CHECK: .debug_info contents:
// CHECK: DW_TAG_compile_unit

// CHECK: DW_TAG_variable
// CHECK: DW_AT_name ("GlobalSharedA")
// CHECK: DW_AT_type ("int")
// CHECK: DW_AT_external (true)
// CHECK: DW_AT_LLVM_memory_space (DW_MSPACE_LLVM_group)
// CHECK-NOT: DW_AT_location

// CHECK: DW_TAG_base_type
// CHECK: DW_AT_name ("int")
// CHECK: DW_AT_encoding (DW_ATE_signed)
// CHECK: DW_AT_byte_size (0x04)

// CHECK: DW_TAG_variable
// CHECK: DW_AT_name ("GlobalSharedB")
// CHECK: DW_AT_type ("int")
// CHECK: DW_AT_external (true)
// CHECK: DW_AT_LLVM_memory_space (DW_MSPACE_LLVM_group)
// CHECK-NOT: DW_AT_location

// CHECK: DW_TAG_variable
// CHECK: DW_AT_name ("GlobalDeviceA")
// CHECK: DW_AT_type ("int")
// CHECK: DW_AT_external (true)
// CHECK: DW_AT_LLVM_memory_space (DW_MSPACE_LLVM_global)
// CHECK: DW_AT_location (DW_OP_addr 0x0, DW_OP_lit0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)

// CHECK: DW_TAG_variable
// CHECK: DW_AT_name ("GlobalDeviceB")
// CHECK: DW_AT_type ("int")
// CHECK: DW_AT_external (true)
// CHECK: DW_AT_LLVM_memory_space (DW_MSPACE_LLVM_global)
// CHECK: DW_AT_location (DW_OP_addr 0x0, DW_OP_lit0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)

// CHECK: DW_TAG_variable
// CHECK: DW_AT_name ("GlobalConstantA")
// CHECK: DW_AT_type ("int")
// CHECK: DW_AT_external (true)
// CHECK: DW_AT_LLVM_memory_space (DW_MSPACE_LLVM_constant)
// CHECK: DW_AT_location (DW_OP_addr 0x0, DW_OP_lit0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)

// CHECK: DW_TAG_variable
// CHECK: DW_AT_name ("GlobalConstantB")
// CHECK: DW_AT_type ("int")
// CHECK: DW_AT_external (true)
// CHECK: DW_AT_LLVM_memory_space (DW_MSPACE_LLVM_constant)
// CHECK: DW_AT_location (DW_OP_addr 0x0, DW_OP_lit0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)

// CHECK: DW_TAG_subprogram
// CHECK: DW_AT_linkage_name ("_Z7kernel1i")
// CHECK: DW_AT_name ("kernel1")
// CHECK: DW_AT_external (true)

// CHECK: DW_TAG_variable
// CHECK: DW_AT_name ("KernelVarSharedA")
// CHECK: DW_AT_type ("int")
// CHECK: DW_AT_LLVM_memory_space (DW_MSPACE_LLVM_group)
// CHECK: DW_AT_location (DW_OP_lit0, DW_OP_lit0, DW_OP_plus, DW_OP_lit3, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)

// CHECK: DW_TAG_variable
// CHECK: DW_AT_name ("KernelVarSharedB")
// CHECK: DW_AT_type ("int")
// CHECK: DW_AT_LLVM_memory_space (DW_MSPACE_LLVM_group)
// CHECK: DW_AT_location (DW_OP_lit0, DW_OP_lit4, DW_OP_plus, DW_OP_lit3, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)

// CHECK: DW_TAG_formal_parameter
// CHECK: DW_AT_location (DW_OP_lit0, DW_OP_lit6, DW_OP_shr, DW_OP_lit0, DW_OP_plus, DW_OP_lit5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
// CHECK: DW_AT_name ("Arg")
// CHECK: DW_AT_type ("int")

// CHECK: DW_TAG_variable
// CHECK: DW_AT_location (DW_OP_lit0, DW_OP_lit6, DW_OP_shr, DW_OP_lit4, DW_OP_plus, DW_OP_lit5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
// CHECK: DW_AT_name ("KernelVarA")
// CHECK: DW_AT_type ("int")

// CHECK: DW_TAG_variable
// CHECK: DW_AT_location (DW_OP_lit0, DW_OP_lit6, DW_OP_shr, DW_OP_lit8, DW_OP_plus, DW_OP_lit5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
// CHECK: DW_AT_name ("KernelVarB")
// CHECK: DW_AT_type ("int")

// CHECK: DW_TAG_variable
// CHECK: DW_AT_location (DW_OP_lit0, DW_OP_lit6, DW_OP_shr, DW_OP_lit16, DW_OP_plus, DW_OP_lit5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
// CHECK: DW_AT_name ("KernelVarSharedAPointer")
// CHECK: DW_AT_type ("int *")

// CHECK: DW_TAG_variable
// CHECK: DW_AT_location (DW_OP_lit0, DW_OP_lit6, DW_OP_shr, DW_OP_lit24, DW_OP_plus, DW_OP_lit5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
// CHECK: DW_AT_name ("KernelVarSharedBPointer")
// CHECK: DW_AT_type ("int *")

// CHECK: DW_TAG_variable
// CHECK: DW_AT_location (DW_OP_lit0, DW_OP_lit6, DW_OP_shr, DW_OP_constu 0x20, DW_OP_plus, DW_OP_lit5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
// CHECK: DW_AT_name ("KernelVarAPointer")
// CHECK: DW_AT_type ("int *")

// CHECK: DW_TAG_variable
// CHECK: DW_AT_location (DW_OP_lit0, DW_OP_lit6, DW_OP_shr, DW_OP_constu 0x28, DW_OP_plus, DW_OP_lit5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
// CHECK: DW_AT_name ("KernelVarBPointer")
// CHECK: DW_AT_type ("int *")
// CHECK: NULL

// CHECK: DW_TAG_subprogram
// CHECK: DW_AT_linkage_name ("_Z5func1i")
// CHECK: DW_AT_name ("func1")
// CHECK: DW_AT_external (true)

// CHECK: DW_TAG_formal_parameter
// CHECK: DW_AT_location (DW_OP_regx 0x40, DW_OP_deref_size 0x4, DW_OP_lit6, DW_OP_shr, DW_OP_lit0, DW_OP_plus, DW_OP_lit5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
// CHECK: DW_AT_name ("Arg")
// CHECK: DW_AT_type ("int")

// CHECK: DW_TAG_variable
// CHECK: DW_AT_location (DW_OP_regx 0x40, DW_OP_deref_size 0x4, DW_OP_lit6, DW_OP_shr, DW_OP_lit4, DW_OP_plus, DW_OP_lit5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
// CHECK: DW_AT_name ("FuncVarA")
// CHECK: DW_AT_type ("int")

// CHECK: DW_TAG_variable
// CHECK: DW_AT_location (DW_OP_regx 0x40, DW_OP_deref_size 0x4, DW_OP_lit6, DW_OP_shr, DW_OP_lit8, DW_OP_plus, DW_OP_lit5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
// CHECK: DW_AT_name ("FuncVarB")
// CHECK: DW_AT_type ("int")

// CHECK: DW_TAG_variable
// CHECK: DW_AT_location (DW_OP_regx 0x40, DW_OP_deref_size 0x4, DW_OP_lit6, DW_OP_shr, DW_OP_lit16, DW_OP_plus, DW_OP_lit5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
// CHECK: DW_AT_name ("FuncVarAPointer")
// CHECK: DW_AT_type ("int *")

// CHECK: DW_TAG_variable
// CHECK: DW_AT_location (DW_OP_regx 0x40, DW_OP_deref_size 0x4, DW_OP_lit6, DW_OP_shr, DW_OP_lit24, DW_OP_plus, DW_OP_lit5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
// CHECK: DW_AT_name ("FuncVarBPointer")
// CHECK: DW_AT_type ("int *")
// CHECK: NULL

// CHECK: DW_TAG_pointer_type
// CHECK: DW_AT_type ("int")
// CHECK: NULL
