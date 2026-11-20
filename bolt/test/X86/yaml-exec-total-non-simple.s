## A valid profile on a non-simple CFG still needs diagnostic block totals.
## The writer must not apply those totals to the optimizer's skipped function.
# REQUIRES: system-linux
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe -data %t.fdata -w %t.yaml -o %t.out -strict=0 \
# RUN:   --print-cfg --print-only=main 2>&1 | FileCheck %s --check-prefix=CFG
# RUN: FileCheck %s --check-prefixes=YAML,CALL < %t.yaml
# RUN: sed 's/ 200$/ 120/' %t.fdata > %t.tail.fdata
# RUN: llvm-bolt %t.exe -data %t.tail.fdata -w %t.tail.yaml -o %t.tail.out -strict=0
# RUN: FileCheck %s --check-prefixes=YAML,TAIL < %t.tail.yaml
# RUN: llvm-bolt %t.exe -data %t.fdata -w %t.nofix.yaml -o %t.nofix.out -strict=0 \
# RUN:   -fix-block-counts=0
# RUN: FileCheck %s --check-prefixes=YAML,NOFIX < %t.nofix.yaml
# RUN: llvm-bolt %t.exe -data %t.fdata -w %t.infer.yaml -o %t.infer.out -strict=0 \
# RUN:   -infer-fall-throughs --print-cfg --print-only=main 2>&1 | FileCheck %s --check-prefix=CFG
# RUN: FileCheck %s --check-prefixes=YAML,CALL < %t.infer.yaml
## With strict CFG handling, BOLT processes this function normally. Its native
## counts must agree with the diagnostic-only calculation above.
# RUN: llvm-bolt %t.exe -data %t.fdata -w %t.simple.yaml -o %t.simple.out -strict=1 \
# RUN:   --print-cfg --print-only=main 2>&1 | FileCheck %s --check-prefix=SIMPLE
# RUN: FileCheck %s --check-prefixes=YAML,CALL < %t.simple.yaml
# RUN: llvm-bolt %t.exe -data %t.tail.fdata -w %t.simple-tail.yaml -o %t.simple-tail.out -strict=1
# RUN: FileCheck %s --check-prefixes=YAML,TAIL < %t.simple-tail.yaml

# CFG: Binary Function "main" after building cfg {
# CFG: IsSimple : 0
# CFG: Exec Count : 90
# CFG: Exec Count : 10
# CFG-NOT: Exec Count : 200

# SIMPLE: Binary Function "main" after building cfg {
# SIMPLE: IsSimple : 1
# SIMPLE: Exec Count : 90
# SIMPLE: Exec Count : 200

# YAML: - name:            main
# YAML:   exec:            90
# YAML:   blocks:
# YAML:     - bid:             0
# YAML:       exec:            90
# YAML:       exec-total:      90
# YAML:       exec:            10
# CALL:       exec-total:      200
# TAIL:       exec-total:      150
# NOFIX:      exec-total:      100

  .text
  .globl main
  .type main,@function
main:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 main 0 0 90
Ljump:
  jmp secondary_entry
# FDATA: 1 main #Ljump# 1 main #secondary_entry# 0 90
## Unreachable, unrecognized indirect control flow makes the CFG non-simple.
  movq %rax, %rbx
  jmp *%rax
  .globl secondary_entry
secondary_entry:
# FDATA: 0 [unknown] 0 1 secondary_entry 0 0 10
Lcall:
  call callee
# FDATA: 1 main #Lcall# 1 callee 0 0 200
  testq %rax, %rax
Ltail:
  jne callee
# FDATA: 1 main #Ltail# 1 callee 0 0 150
  retq
  .cfi_endproc
  .size main, .-main

  .globl callee
  .type callee,@function
callee:
  retq
  .size callee, .-callee
