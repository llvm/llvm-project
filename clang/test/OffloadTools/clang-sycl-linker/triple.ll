; Tests the clang-sycl-linker tool: triple handling.
;
; REQUIRES: spirv-registered-target
;
; RUN: rm -rf %t && split-file %s %t
;
; Test when explicit -triple= is used. Input does not supply a triple.
; RUN: llvm-as %t/no-triple.ll -o %t/no-triple.bc
; RUN: clang-sycl-linker --dry-run -v -triple=spirv64 %t/no-triple.bc -o %t/no-triple-input.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NO-TRIPLE-INPUT
; NO-TRIPLE-INPUT: sycl-bundle: image kind: spv, triple: spirv64, arch: {{$}}
;
; Test that triple was inferred from inputs and recorded in the offload image.
; RUN: llvm-as %t/input1.ll -o %t/input1.bc
; RUN: llvm-as %t/input2.ll -o %t/input2.bc
; RUN: clang-sycl-linker --dry-run -v --module-split-mode=none %t/input1.bc %t/input2.bc -o %t/spirv.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=TRIPLE-INFERENCE
; TRIPLE-INFERENCE: sycl-bundle: image kind: spv, triple: spirv64, arch: {{$}}
;
; Test error on mismatched triple between inputs.
; RUN: llvm-as %t/input-mismatch.ll -o %t/input-mismatch.bc
; RUN: not clang-sycl-linker --dry-run %t/input1.bc %t/input-mismatch.bc -o %t/mismatch.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=TRIPLE-MISMATCH
; TRIPLE-MISMATCH: error: conflicting target triples: 'spirv64' (from {{.*}}input1.bc) vs 'spirv32' (from {{.*}}input-mismatch.bc)
;
; Test error when explicit -triple= conflicts with the input's triple.
; RUN: not clang-sycl-linker --dry-run -triple=spirv32 %t/input1.bc -o %t/explicit.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=EXPL-TRIPLE-MISMATCH
; EXPL-TRIPLE-MISMATCH: error: conflicting target triples: 'spirv32' (from --triple=) vs 'spirv64' (from {{.*}}input1.bc)
;
; Test error when neither -triple= nor any input supplies a triple.
; RUN: not clang-sycl-linker --dry-run %t/no-triple.bc -o a.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NO-TRIPLE
; NO-TRIPLE: target triple must be specified or inferable from inputs

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

;--- input-mismatch.ll
target triple = "spirv32"

define spir_kernel void @kernel_c() #0 {
  ret void
}

attributes #0 = { "sycl-module-id"="TU3.cpp" }

;--- no-triple.ll
define spir_kernel void @kernel_d() #0 {
  ret void
}

attributes #0 = { "sycl-module-id"="TU4.cpp" }
