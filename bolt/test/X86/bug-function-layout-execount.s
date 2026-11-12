## Verifies that llvm-bolt correctly sorts functions by their execution counts.

# REQUIRES: x86_64-linux, asserts

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe --data %t.fdata --lite --reorder-functions=exec-count \
# RUN:   -v=2 --debug-only=hfsort -o %t.null 2>&1 | FileCheck %s

# CHECK: Starting pass: reorder-functions
# CHECK-NEXT: hot func func2 (1500)
# CHECK-NEXT: hot func func1 (500)
# CHECK-NEXT: hot func main (400)
# CHECK-NEXT: hot func func5 (110)
# CHECK-NEXT: hot func func3 (100)
# CHECK-NEXT: hot func func4 (99)

  .text
  .globl main
  .type main, %function
main:
# FDATA: 0 [unknown] 0 1 main 0 1 400
  .cfi_startproc
  call func1
  retq
  .size _start, .-_start
  .cfi_endproc

  .globl  func1
  .type func1,@function
func1:
# FDATA: 0 [unknown] 0 1 func1 0 1 500
  .cfi_startproc
  retq
  .size func1, .-func1
  .cfi_endproc

  .globl  func2
  .type func2,@function
func2:
# FDATA: 0 [unknown] 0 1 func2 0 1 1500
  .cfi_startproc
  retq
  .size func2, .-func2
  .cfi_endproc

  .globl  func3
  .type func3,@function
func3:
# FDATA: 0 [unknown] 0 1 func3 0 1 100
  .cfi_startproc
  retq
  .size func3, .-func3
  .cfi_endproc

  .globl  func4
  .type func4,@function
func4:
# FDATA: 0 [unknown] 0 1 func4 0 1 99
  .cfi_startproc
  retq
  .size func4, .-func4
  .cfi_endproc

  .globl  func5
  .type func5,@function
func5:
# FDATA: 0 [unknown] 0 1 func5 0 1 110
  .cfi_startproc
  retq
  .size func5, .-func5
  .cfi_endproc
