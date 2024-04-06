# This reproduces a bug with BOLT setting incorrect discriminator for
# secondary entry points in YAML profile.

# REQUIRES: system-linux
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe -o %t.out --data %t.fdata -w %t.yaml --print-profile \
# RUN:   --print-only=main | FileCheck %s --check-prefix=CHECK-CFG
# RUN: FileCheck %s -input-file %t.yaml
# CHECK:      - name:    main
# CHECK-NEXT:   fid:     2
# CHECK-NEXT:   hash:    0xADF270D550151185
# CHECK-NEXT:   exec:    0
# CHECK-NEXT:   nblocks: 4
# CHECK-NEXT:   blocks:
# CHECK:          - bid:   1
# CHECK-NEXT:       insns: 1
# CHECK-NEXT:       hash:  0x36A303CBA4360014
# CHECK-NEXT:       calls: [ { off: 0x0, fid: 1, disc: 1, cnt: 1 } ]
# CHECK:          - bid:   2
# CHECK-NEXT:       insns: 5
# CHECK-NEXT:       hash:  0x8B2F5747CD0019
# CHECK-NEXT:       calls: [ { off: 0x0, fid: 1, disc: 1, cnt: 1, mis: 1 } ]

# Make sure that the profile is attached correctly
# RUN: llvm-bolt %t.exe -o %t.out --data %t.yaml --print-profile \
# RUN:   --print-only=main | FileCheck %s --check-prefix=CHECK-CFG

# CHECK-CFG: Binary Function "main" after attaching profile {
# CHECK-CFG:      callq secondary_entry # Offset: [[#]] # Count: 1
# CHECK-CFG:      callq *%rax # Offset: [[#]] # CallProfile: 1 (1 misses) :
# CHECK-CFG-NEXT:     { secondary_entry: 1 (1 misses) }

.globl func
.type	func, @function
func:
# FDATA: 0 [unknown] 0 1 func 0 1 0
  .cfi_startproc
  pushq   %rbp
  movq    %rsp, %rbp
.globl secondary_entry
secondary_entry:
  popq    %rbp
  retq
  nopl    (%rax)
  .cfi_endproc
  .size	func, .-func

.globl main
.type	main, @function
main:
  .cfi_startproc
  pushq   %rbp
  movq    %rsp, %rbp
  subq    $16, %rsp
  movl    $0, -4(%rbp)
  testq   %rax, %rax
  jne     Lindcall
Lcall:
  call    secondary_entry
# FDATA: 1 main #Lcall# 1 secondary_entry 0 1 1
Lindcall:
  callq   *%rax
# FDATA: 1 main #Lindcall# 1 secondary_entry 0 1 1
  xorl    %eax, %eax
  addq    $16, %rsp
  popq    %rbp
  retq
# For relocations against .text
  call exit
  .cfi_endproc
  .size	main, .-main
