# This reproduces a bug with not processing interprocedural references from
# ignored functions.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out --enable-bat -funcs=main
# RUN: link_fdata %s %t.out %t.preagg PREAGG
# RUN: perf2bolt %t.out -p %t.preagg --pa -o %t.fdata -w %t.yaml
# RUN: FileCheck %s --input-file=%t.fdata --check-prefix=CHECK-FDATA
# RUN: FileCheck %s --input-file=%t.yaml --check-prefix=CHECK-YAML

# CHECK-FDATA: 1 main 0 1 foo a 1 1
# CHECK-YAML: name: main
# CHECK-YAML: calls: {{.*}} disc: 1

# PREAGG: B #main# #foo_secondary# 1 1
# main calls foo at valid instruction offset past nops that are to be stripped.
  .globl main
main:
  .cfi_startproc
  call foo_secondary
  ret
  .cfi_endproc
.size main,.-main

# Placeholder cold fragment to force main to be ignored in non-relocation mode.
  .globl main.cold
main.cold:
  .cfi_startproc
  ud2
  .cfi_endproc
.size main.cold,.-main.cold

# foo is set up to contain a valid instruction at called offset, and trapping
# instructions past that.
  .globl foo
foo:
  .cfi_startproc
  .nops 10
  .globl foo_secondary
foo_secondary:
  ret
  .rept 20
  int3
  .endr
  .cfi_endproc
.size foo,.-foo
