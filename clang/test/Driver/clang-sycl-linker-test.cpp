// Tests the clang-sycl-linker tool.
//
// REQUIRES: spirv-registered-target
//
// Test the dry run of a simple case to link two input files.
// Also verifies the default split mode ("none").
// RUN: %clangxx -emit-llvm -c -target spirv64 %s -o %t_1.bc
// RUN: %clangxx -emit-llvm -c -target spirv64 %s -o %t_2.bc
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc -o %t-spirv.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SIMPLE-FO
// SIMPLE-FO:      sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
// SIMPLE-FO-NEXT: sycl-module-split: input: [[LLVMLINKOUT]].bc, output: [[LLVMLINKOUT]].bc, mode: none
// SIMPLE-FO-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: {{.*}}_0.spv
//
// Test that IMG_SPIRV image kind is set for non-AOT compilation.
// RUN: llvm-objdump --offloading %t-spirv.out | FileCheck %s --check-prefix=IMAGE-KIND-SPIRV
// IMAGE-KIND-SPIRV: kind            spir-v
//
// Test the dry run of a simple case with device library files specified.
// RUN: mkdir -p %t.dir
// RUN: touch %t.dir/lib1.bc
// RUN: touch %t.dir/lib2.bc
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t_1.bc %t_2.bc --library-path=%t.dir --device-libs=lib1.bc,lib2.bc -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEVLIBS
// DEVLIBS:      sycl-device-link: inputs: {{.*}}.bc  libfiles: {{.*}}lib1.bc, {{.*}}lib2.bc  output: [[LLVMLINKOUT:.*]].bc
// DEVLIBS-NEXT: sycl-module-split: input: [[LLVMLINKOUT]].bc, output: [[LLVMLINKOUT]].bc, mode: none
// DEVLIBS-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: a_0.spv
//
// Test a simple case with a random file (not bitcode) as input.
// RUN: touch %t.o
// RUN: not clang-sycl-linker -triple=spirv64 %t.o -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FILETYPEERROR
// FILETYPEERROR: Unsupported file type
//
// Test to see if device library related errors are emitted.
// RUN: not clang-sycl-linker --dry-run -triple=spirv64 %t_1.bc %t_2.bc --library-path=%t.dir --device-libs= -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEVLIBSERR1
// DEVLIBSERR1: Number of device library files cannot be zero
// RUN: not clang-sycl-linker --dry-run -triple=spirv64 %t_1.bc %t_2.bc --library-path=%t.dir --device-libs=lib1.bc,lib2.bc,lib3.bc -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEVLIBSERR2
// DEVLIBSERR2: '{{.*}}lib3.bc' SYCL device library file is not found
//
// Test AOT compilation for an Intel GPU.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 -arch=bmg_g21 %t_1.bc %t_2.bc -o %t-aot-gpu.out 2>&1 \
// RUN:     --ocloc-options="-a -b" \
// RUN:   | FileCheck %s --check-prefix=AOT-INTEL-GPU
// AOT-INTEL-GPU:      sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc libfiles: output: [[LLVMLINKOUT:.*]].bc
// AOT-INTEL-GPU-NEXT: sycl-module-split: input: [[LLVMLINKOUT]].bc, output: [[LLVMLINKOUT]].bc, mode: none
// AOT-INTEL-GPU-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: [[SPIRVTRANSLATIONOUT:.*]]_0.spv
// AOT-INTEL-GPU-NEXT: "{{.*}}ocloc{{.*}}" {{.*}}-device bmg_g21 -a -b {{.*}}-output [[SPIRVTRANSLATIONOUT]]_0.out -file [[SPIRVTRANSLATIONOUT]]_0.spv
//
// Test that IMG_Object image kind is set for AOT compilation (Intel GPU).
// RUN: llvm-objdump --offloading %t-aot-gpu.out | FileCheck %s --check-prefix=IMAGE-KIND-OBJECT
// IMAGE-KIND-OBJECT: kind            elf
//
// Test AOT compilation for an Intel CPU.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 -arch=graniterapids %t_1.bc %t_2.bc -o %t-aot-cpu.out 2>&1 \
// RUN:     --opencl-aot-options="-a -b" \
// RUN:   | FileCheck %s --check-prefix=AOT-INTEL-CPU
// AOT-INTEL-CPU:      sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc libfiles: output: [[LLVMLINKOUT:.*]].bc
// AOT-INTEL-CPU-NEXT: sycl-module-split: input: [[LLVMLINKOUT]].bc, output: [[LLVMLINKOUT]].bc, mode: none
// AOT-INTEL-CPU-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: [[SPIRVTRANSLATIONOUT:.*]]_0.spv
// AOT-INTEL-CPU-NEXT: "{{.*}}opencl-aot{{.*}}" {{.*}}--device=cpu -a -b {{.*}}-o [[SPIRVTRANSLATIONOUT]]_0.out [[SPIRVTRANSLATIONOUT]]_0.spv
//
// Test that IMG_Object image kind is set for AOT compilation (Intel CPU).
// RUN: llvm-objdump --offloading %t-aot-cpu.out | FileCheck %s --check-prefix=IMAGE-KIND-OBJECT
//
// Check that the output file must be specified.
// RUN: not clang-sycl-linker --dry-run %t_1.bc %t_2.bc 2>&1 \
// RUN: | FileCheck %s --check-prefix=NOOUTPUT
// NOOUTPUT: Output file must be specified
//
// Check that the target triple must be specified.
// RUN: not clang-sycl-linker --dry-run %t_1.bc %t_2.bc -o a.out 2>&1 \
// RUN: | FileCheck %s --check-prefix=NOTARGET
// NOTARGET: Target triple must be specified
//
// Test the split mode ("none"): no extra splits are produced.
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 --module-split-mode=none %t_1.bc %t_2.bc -o %t-split-none.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SPLIT-NONE
// SPLIT-NONE:      sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
// SPLIT-NONE-NEXT: sycl-module-split: input: [[LLVMLINKOUT]].bc, output: [[LLVMLINKOUT]].bc, mode: none
// SPLIT-NONE-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: {{.*}}_0.spv
// SPLIT-NONE-NOT:  LLVM backend: input: {{.*}}.bc, output: {{.*}}_1.spv
//
// Test per-kernel split: a module with two SPIR_KERNEL functions produces two
// device images.
// RUN: llvm-as %S/Inputs/SYCL/two-kernels.ll -o %t-two.bc
// RUN: clang-sycl-linker --dry-run -v -triple=spirv64 --module-split-mode=kernel %t-two.bc -o %t-split-kernel.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SPLIT-KERNEL
// SPLIT-KERNEL:      sycl-device-link: inputs: {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
// SPLIT-KERNEL-NEXT: sycl-module-split: input: [[LLVMLINKOUT]].bc, output: [[SPLIT0:.*]].bc, [[SPLIT1:.*]].bc, mode: kernel
// SPLIT-KERNEL-NEXT: LLVM backend: input: [[SPLIT0]].bc, output: {{.*}}_0.spv
// SPLIT-KERNEL-NEXT: LLVM backend: input: [[SPLIT1]].bc, output: {{.*}}_1.spv
//
// Test that an invalid split mode is rejected.
// RUN: not clang-sycl-linker --dry-run -triple=spirv64 --module-split-mode=bogus %t_1.bc -o a.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SPLIT-INVALID
// SPLIT-INVALID: module-split-mode value isn't recognized: bogus
