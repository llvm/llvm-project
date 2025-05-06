// Tests the clang-sycl-linker tool.
//
// REQUIRES: spirv-registered-target
//
// Test the dry run of a simple case to link two input files.
// RUN: %clangxx -emit-llvm -c -target spirv64 %s -o %t_1.bc
// RUN: %clangxx -emit-llvm -c -target spirv64 %s -o %t_2.bc
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SIMPLE-FO
// SIMPLE-FO: sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
// SIMPLE-FO-NEXT: SPIR-V Backend: input: [[LLVMLINKOUT]].bc, output: a_0.spv
//
// Test the dry run of a simple case with device library files specified.
// RUN: touch %T/lib1.bc
// RUN: touch %T/lib2.bc
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc --library-path=%T --device-libs=lib1.bc,lib2.bc -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEVLIBS
// DEVLIBS: sycl-device-link: inputs: {{.*}}.bc  libfiles: {{.*}}lib1.bc, {{.*}}lib2.bc  output: [[LLVMLINKOUT:.*]].bc
// DEVLIBS-NEXT: SPIR-V Backend: input: [[LLVMLINKOUT]].bc, output: a_0.spv
//
// Test a simple case with a random file (not bitcode) as input.
// RUN: touch %t.o
// RUN: not clang-sycl-linker -triple=spirv64 %t.o -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FILETYPEERROR
// FILETYPEERROR: Unsupported file type
//
// Test to see if device library related errors are emitted.
// RUN: not clang-sycl-linker --dry-run -triple=spirv64 %t_1.bc %t_2.bc --library-path=%T --device-libs= -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEVLIBSERR1
// DEVLIBSERR1: Number of device library files cannot be zero
// RUN: not clang-sycl-linker --dry-run -triple=spirv64 %t_1.bc %t_2.bc --library-path=%T --device-libs=lib1.bc,lib2.bc,lib3.bc -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEVLIBSERR2
// DEVLIBSERR2: '{{.*}}lib3.bc' SYCL device library file is not found
//
// Test AOT compilation for an Intel GPU.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 -arch=bmg_g21 %t_1.bc %t_2.bc -o a.out 2>&1 \
// RUN:     --ocloc-options="-a -b" \
// RUN:   | FileCheck %s --check-prefix=AOT-INTEL-GPU
// AOT-INTEL-GPU:      sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc libfiles: output: [[LLVMLINKOUT:.*]].bc
// AOT-INTEL-GPU-NEXT: SPIR-V Backend: input: [[LLVMLINKOUT]].bc, output: [[SPIRVTRANSLATIONOUT:.*]]_0.spv
// AOT-INTEL-GPU-NEXT: "{{.*}}ocloc{{.*}}" {{.*}}-device bmg_g21 -a -b {{.*}}-output a_0.out -file [[SPIRVTRANSLATIONOUT]]_0.spv
//
// Test AOT compilation for an Intel CPU.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 -arch=graniterapids %t_1.bc %t_2.bc -o a.out 2>&1 \
// RUN:     --opencl-aot-options="-a -b" \
// RUN:   | FileCheck %s --check-prefix=AOT-INTEL-CPU
// AOT-INTEL-CPU:      sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc libfiles: output: [[LLVMLINKOUT:.*]].bc
// AOT-INTEL-CPU-NEXT: SPIR-V Backend: input: [[LLVMLINKOUT]].bc, output: [[SPIRVTRANSLATIONOUT:.*]]_0.spv
// AOT-INTEL-CPU-NEXT: "{{.*}}opencl-aot{{.*}}" {{.*}}--device=cpu -a -b {{.*}}-o a_0.out [[SPIRVTRANSLATIONOUT]]_0.spv
//
// Check that the output file must be specified.
// RUN: not clang-sycl-linker --dry-run %t_1.bc %t_2.bc 2>& 1 \
// RUN: | FileCheck %s --check-prefix=NOOUTPUT
// NOOUTPUT: Output file must be specified
//
// Check that the target triple must be.
// RUN: not clang-sycl-linker --dry-run %t_1.bc %t_2.bc -o a.out 2>& 1 \
// RUN: | FileCheck %s --check-prefix=NOTARGET
// NOTARGET: Target triple must be specified
