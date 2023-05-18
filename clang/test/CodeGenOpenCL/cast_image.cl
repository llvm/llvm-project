// RUN: %clang_cc1 -emit-llvm -o - -triple amdgcn--amdhsa %s | FileCheck --check-prefix=AMDGCN %s
// RUN: %clang_cc1 -emit-llvm -o - -triple x86_64-unknown-unknown %s | FileCheck --check-prefix=X86 %s

#ifdef __AMDGCN__

constant int* convert(image2d_t img) {
  // AMDGCN: ret ptr addrspace(4) %img
  return __builtin_astype(img, constant int*);
}

#else

global int* convert(image2d_t img) {
  // X86: ret ptr %img
  return __builtin_astype(img, global int*);
}

#endif
