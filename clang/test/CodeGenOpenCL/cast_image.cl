// RUN: %clang_cc1 -emit-llvm -o - -triple amdgcn--amdhsa %s | FileCheck --check-prefix=AMDGCN %s
// RUN: %clang_cc1 -emit-llvm -o - -triple spir-unknown-unknown %s | FileCheck --check-prefix=SPIR %s

#ifdef __AMDGCN__

constant int* convert(image2d_t img) {
  // AMDGCN: bitcast %opencl.image2d_ro_t addrspace(4)* %img to i32 addrspace(4)*
  return __builtin_astype(img, constant int*);
}

#else

global int* convert(image2d_t img) {
  // SPIR: bitcast %opencl.image2d_ro_t addrspace(1)* %img to i32 addrspace(1)*
  return __builtin_astype(img, global int*);
}

#endif
