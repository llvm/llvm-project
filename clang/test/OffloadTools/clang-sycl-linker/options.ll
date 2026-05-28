; Tests command-line options of clang-sycl-linker that exercise the SPIR-V
; backend (--print-linked-module, --spirv-dump-device-code, --save-temps).
; CLI-only tests that do not need the SPIR-V backend live in cli.test.
;
; REQUIRES: spirv-registered-target

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

; Build a bitcode input used by every RUN line below.
; RUN: llvm-as %s -o %t.bc

; Test --print-linked-module
; RUN: clang-sycl-linker %t.bc --print-linked-module -o %t.out > %t.ll
; RUN: FileCheck %s --check-prefix=PRINT-LINKED < %t.ll
; PRINT-LINKED: target triple = "spirv64"

; RUN: rm -rf %t.dir && mkdir -p %t.dir

; Test --spirv-dump-device-code: the linker creates the (nested) directory if
; missing and copies the generated SPIR-V file there. Files are named
; "<output-stem>_<index>.spv" and shared across split modules.
; RUN: clang-sycl-linker %t.bc --spirv-dump-device-code=%t.dir/nested -o out.spv
; RUN: test -f %t.dir/nested/out_0.spv

; Test --spirv-dump-device-code with an empty value: falls back to the current
; working directory, so we cd into a scratch dir first to keep artifacts
; isolated and avoid polluting %t.dir's parent.
; RUN: mkdir -p %t.dir/fallback
; RUN: cd %t.dir/fallback && clang-sycl-linker %t.bc --spirv-dump-device-code= -o out.spv
; RUN: test -f %t.dir/fallback/out_0.spv

; Test --spirv-dump-device-code where the path cannot be created (a regular
; file blocks it). Expect a clear diagnostic rather than a deeper failure.
; RUN: rm -rf %t.blocker && touch %t.blocker
; RUN: not clang-sycl-linker %t.bc --spirv-dump-device-code=%t.blocker/sub -o %t.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=DUMP-DIR-ERR
; DUMP-DIR-ERR: cannot create SPIR-V dump directory '{{.*}}.blocker/sub'

; Test --save-temps: keeps intermediate .bc files in the current directory, so
; cd into a scratch dir to make the resulting paths predictable.
; RUN: mkdir -p %t.dir/save-temps
; RUN: cd %t.dir/save-temps && clang-sycl-linker %t.bc --save-temps -o out.spv
; RUN: ls %t.dir/save-temps/out.spv-*.bc
