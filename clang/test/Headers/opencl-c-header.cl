// RUN: %clang_cc1 -triple spir-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -cl-std=CL1.1| FileCheck %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -cl-std=CL1.2| FileCheck %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -fblocks -emit-llvm -o - %s -cl-std=CL2.0| FileCheck %s
// RUN: %clang_cc1 -triple spir64-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spir64-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -cl-std=CL1.1| FileCheck %s
// RUN: %clang_cc1 -triple spir64-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -cl-std=CL1.2| FileCheck %s
// RUN: %clang_cc1 -triple spir64-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -fblocks -emit-llvm -o - %s -cl-std=CL2.0| FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-amdhsa -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-amdhsa -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -cl-std=CL1.1| FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-amdhsa -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -cl-std=CL1.2| FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-unknown-amdhsa -internal-isystem ../../lib/Headers -include opencl-c.h -fblocks -emit-llvm -o - %s -cl-std=CL2.0| FileCheck %s
// RUN: %clang_cc1 -triple ppc64-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple ppc64-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -cl-std=CL1.1| FileCheck %s
// RUN: %clang_cc1 -triple ppc64-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -cl-std=CL1.2| FileCheck %s
// RUN: %clang_cc1 -triple ppc64-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -fblocks -emit-llvm -o - %s -cl-std=CL2.0| FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -cl-std=CL1.1| FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -cl-std=CL1.2| FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -fblocks -emit-llvm -o - %s -cl-std=CL2.0| FileCheck %s
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -cl-std=CL1.1| FileCheck %s
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -cl-std=CL1.2| FileCheck %s
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -fblocks -emit-llvm -o - %s -cl-std=CL2.0| FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -cl-std=CL1.1| FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -emit-llvm -o - %s -cl-std=CL1.2| FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -internal-isystem ../../lib/Headers -include opencl-c.h -fblocks -emit-llvm -o - %s -cl-std=CL2.0| FileCheck %s

// CHECK: _Z16convert_char_rtec
char f(char x) {
  return convert_char_rte(x);
}
