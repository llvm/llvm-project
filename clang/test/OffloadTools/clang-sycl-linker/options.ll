; Tests non-functional command line options of the clang-sycl-linker tool.
;
; REQUIRES: spirv-registered-target
;
; Test --help
; RUN: clang-sycl-linker --help | FileCheck %s --check-prefix=HELP
; HELP: OVERVIEW: A utility that wraps around several steps required to link SYCL device files.
; HELP: USAGE: clang-sycl-linker
; HELP: OPTIONS:
;
; Test --version
; RUN: clang-sycl-linker --version | FileCheck %s --check-prefix=VERSION
; VERSION: clang-sycl-linker version
;
; Test missing input files
; RUN: not clang-sycl-linker --dry-run -triple=spirv64 -o %t.out 2>&1 | FileCheck %s --check-prefix=NO-INPUT
; NO-INPUT: No input files provided
;
; Create a simple bitcode file for subsequent tests
; RUN: llvm-as %s -o %t.bc
;
; Test --print-linked-module
; RUN: clang-sycl-linker --dry-run -triple=spirv64 %t.bc --print-linked-module -o %t.out > %t.ll
; RUN: FileCheck %s --check-prefix=PRINT-LINKED < %t.ll
; PRINT-LINKED: target triple = "spirv64"
;
; Test --save-temps
; RUN: rm -rf %t.dir && mkdir -p %t.dir
; RUN: cd %t.dir && clang-sycl-linker --dry-run -triple=spirv64 %t.bc --save-temps -o out.spv
; RUN: ls %t.dir/out.spv-*.bc | count 1
;
; Test --spirv-dump-device-code (should parse without error)
; RUN: clang-sycl-linker --dry-run -triple=spirv64 %t.bc --spirv-dump-device-code=%t.dir -o %t.out
;
; Test --spirv-dump-device-code with no value (fallback to ./)
; RUN: clang-sycl-linker --dry-run -triple=spirv64 %t.bc --spirv-dump-device-code= -o %t.out

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

