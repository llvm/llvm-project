; Tests the clang-sycl-linker tool.
;
; REQUIRES: spirv-registered-target
;
; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-as %t/input1.ll -o %t/input1.bc
; RUN: llvm-as %t/input2.ll -o %t/input2.bc
;
; Test the dry run of a simple case to link two input files.
; RUN: clang-sycl-linker --dry-run -v -triple=spirv64 --module-split-mode=none %t/input1.bc %t/input2.bc -o %t/spirv.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=SIMPLE-FO
; SIMPLE-FO:      sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc  libfiles:  output: [[LLVMLINKOUT:.*]].bc
; SIMPLE-FO-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: {{.*}}_0.spv
; SIMPLE-FO-NOT:  {{.+}}
;
; Test that IMG_SPIRV image kind is set for non-AOT compilation.
; RUN: llvm-objdump --offloading %t/spirv.out | FileCheck %s --check-prefix=IMAGE-KIND-SPIRV
; IMAGE-KIND-SPIRV: kind            spir-v
;
; Test the dry run of a simple case with device library files specified.
; RUN: mkdir -p %t/libs
; RUN: touch %t/libs/lib1.bc
; RUN: touch %t/libs/lib2.bc
; RUN: clang-sycl-linker --dry-run -v -triple=spirv64 --module-split-mode=none %t/input1.bc %t/input2.bc --library-path=%t/libs --device-libs=lib1.bc,lib2.bc -o a.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=DEVLIBS
; DEVLIBS:      sycl-device-link: inputs: {{.*}}.bc  libfiles: {{.*}}lib1.bc, {{.*}}lib2.bc  output: [[LLVMLINKOUT:.*]].bc
; DEVLIBS-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: a_0.spv
;
; Test a simple case with a random file (not bitcode) as input.
; RUN: touch %t/dummy.o
; RUN: not clang-sycl-linker -triple=spirv64 %t/dummy.o -o a.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=FILETYPEERROR
; FILETYPEERROR: Unsupported file type
;
; Test to see if device library related errors are emitted.
; RUN: not clang-sycl-linker --dry-run -triple=spirv64 %t/input1.bc %t/input2.bc --library-path=%t/libs --device-libs= -o a.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=DEVLIBSERR1
; DEVLIBSERR1: Number of device library files cannot be zero
; RUN: not clang-sycl-linker --dry-run -triple=spirv64 %t/input1.bc %t/input2.bc --library-path=%t/libs --device-libs=lib1.bc,lib2.bc,lib3.bc -o a.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=DEVLIBSERR2
; DEVLIBSERR2: '{{.*}}lib3.bc' SYCL device library file is not found
;
; Test AOT compilation for an Intel GPU.
; RUN: clang-sycl-linker --dry-run -v -triple=spirv64 --module-split-mode=none -arch=bmg_g21 %t/input1.bc %t/input2.bc -o %t/aot-gpu.out 2>&1 \
; RUN:     --ocloc-options="-a -b" \
; RUN:   | FileCheck %s --check-prefix=AOT-INTEL-GPU
; AOT-INTEL-GPU:      sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc libfiles: output: [[LLVMLINKOUT:.*]].bc
; AOT-INTEL-GPU-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: [[SPIRVTRANSLATIONOUT:.*]]_0.spv
; AOT-INTEL-GPU-NEXT: "{{.*}}ocloc{{.*}}" {{.*}}-device bmg_g21 -a -b {{.*}}-output [[SPIRVTRANSLATIONOUT]]_0.out -file [[SPIRVTRANSLATIONOUT]]_0.spv
;
; Test that IMG_Object image kind is set for AOT compilation (Intel GPU).
; RUN: llvm-objdump --offloading %t/aot-gpu.out | FileCheck %s --check-prefix=IMAGE-KIND-OBJECT
; IMAGE-KIND-OBJECT: kind            elf
;
; Test AOT compilation for an Intel CPU.
; RUN: clang-sycl-linker --dry-run -v -triple=spirv64 --module-split-mode=none -arch=graniterapids %t/input1.bc %t/input2.bc -o %t/aot-cpu.out 2>&1 \
; RUN:     --opencl-aot-options="-a -b" \
; RUN:   | FileCheck %s --check-prefix=AOT-INTEL-CPU
; AOT-INTEL-CPU:      sycl-device-link: inputs: {{.*}}.bc, {{.*}}.bc libfiles: output: [[LLVMLINKOUT:.*]].bc
; AOT-INTEL-CPU-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: [[SPIRVTRANSLATIONOUT:.*]]_0.spv
; AOT-INTEL-CPU-NEXT: "{{.*}}opencl-aot{{.*}}" {{.*}}--device=cpu -a -b {{.*}}-o [[SPIRVTRANSLATIONOUT]]_0.out [[SPIRVTRANSLATIONOUT]]_0.spv
;
; Test that IMG_Object image kind is set for AOT compilation (Intel CPU).
; RUN: llvm-objdump --offloading %t/aot-cpu.out | FileCheck %s --check-prefix=IMAGE-KIND-OBJECT
;
; Check that the output file must be specified.
; RUN: not clang-sycl-linker --dry-run %t/input1.bc %t/input2.bc 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOOUTPUT
; NOOUTPUT: Output file must be specified
;
; Check that the target triple must be specified.
; RUN: not clang-sycl-linker --dry-run %t/input1.bc %t/input2.bc -o a.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOTARGET
; NOTARGET: Target triple must be specified
;
; Input with no entry points still produces an offload image.
; RUN: llvm-as %t/no-entry-points.ll -o %t/no-entry-points.bc
; RUN: clang-sycl-linker -triple=spirv64 %t/no-entry-points.bc -o %t/no-entry-points.out
; RUN: llvm-objdump --offloading %t/no-entry-points.out | FileCheck %s --check-prefix=NO-ENTRY-POINTS
; NO-ENTRY-POINTS: OFFLOADING IMAGE [0]:
; NO-ENTRY-POINTS: producer        sycl

;--- input1.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_kernel void @kernel_a() #0 {
  ret void
}

attributes #0 = { "sycl-module-id"="TU1.cpp" }

;--- input2.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_kernel void @kernel_b() #0 {
  ret void
}

attributes #0 = { "sycl-module-id"="TU2.cpp" }

;--- no-entry-points.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @helper() {
  ret i32 0
}
