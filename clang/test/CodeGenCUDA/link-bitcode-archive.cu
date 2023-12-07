// Prepare archive of bitcode file.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm-bc \
// RUN:    -fcuda-is-device \
// RUN:    -disable-llvm-passes -DIS_LIB -o %t.bc -xhip %s

// RUN: rm -f %t.a
// RUN: llvm-ar rcs %t.a %t.bc

// Link archive of bitcode file.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:    -mlink-builtin-bitcode %t.a  -emit-llvm \
// RUN:    -disable-llvm-passes -o - -xhip %s \
// RUN:    | FileCheck %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:    -mlink-bitcode-file %t.a  -emit-llvm \
// RUN:    -disable-llvm-passes -o - -xhip %s \
// RUN:    | FileCheck %s

// Test empty file as arhive.

// RUN: rm -f %t.a
// RUN: touch %t.a

// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:    -mlink-builtin-bitcode %t.a  -emit-llvm \
// RUN:    -disable-llvm-passes -o - -xhip %s 2>&1\
// RUN:    | FileCheck %s -check-prefix=INVLID

// Test invalid arhive.

// RUN: echo "!<arch>\nfake" >%t.a
// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:    -mlink-builtin-bitcode %t.a  -emit-llvm \
// RUN:    -disable-llvm-passes -o - -xhip %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix=INVLID

// Test archive of invalid bitcode file.

// RUN: echo "BC\xC0\xDE" >%t.bc
// RUN: rm -f %t.a
// RUN: llvm-ar rcs %t.a %t.bc
// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:    -mlink-builtin-bitcode %t.a  -emit-llvm \
// RUN:    -disable-llvm-passes -o - -xhip %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix=INVLID-BC

#include "Inputs/cuda.h"

#ifdef IS_LIB
__device__ void libfun() {}
#else
__device__ void libfun();
__global__ void kern() {
 libfun();
}
#endif

// CHECK: define {{.*}}void @_Z6libfunv()
// INVLID: fatal error: cannot open file {{.*}}: The file was not recognized as a valid object file
// INVLID-BC: fatal error: cannot open file {{.*}}: Invalid bitcode signature
