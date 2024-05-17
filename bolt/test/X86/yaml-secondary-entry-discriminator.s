## This reproduces a bug with BOLT setting incorrect discriminator for
## secondary entry points in YAML profile.

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
# CHECK-NEXT:   hash:    {{.*}}
# CHECK-NEXT:   exec:    0
# CHECK-NEXT:   nblocks: 4
# CHECK-NEXT:   blocks:
# CHECK:          - bid:   1
# CHECK-NEXT:       insns: 1
# CHECK-NEXT:       hash:  {{.*}}
# CHECK-NEXT:       calls: [ { off: 0x0, fid: 1, disc: 1, cnt: 1 } ]
# CHECK:          - bid:   2
# CHECK-NEXT:       insns: 5
# CHECK-NEXT:       hash:  {{.*}}
# CHECK-NEXT:       calls: [ { off: 0x0, fid: 1, disc: 1, cnt: 1, mis: 1 } ]

## Make sure that the profile is attached correctly
# RUN: llvm-bolt %t.exe -o %t.out --data %t.yaml --print-profile \
# RUN:   --print-only=main | FileCheck %s --check-prefix=CHECK-CFG

# CHECK-CFG: Binary Function "main" after attaching profile {
# CHECK-CFG:      callq secondary_entry # Offset: [[#]] # Count: 1
# CHECK-CFG:      callq *%rax # Offset: [[#]] # CallProfile: 1 (1 misses) :
# CHECK-CFG-NEXT:     { secondary_entry: 1 (1 misses) }

## YAML BAT test of calling BAT secondary entry from non-BAT function
## Now force-split func and skip main (making it call secondary entries)
# RUN: llvm-bolt %t.exe -o %t.bat --data %t.fdata --funcs=func \
# RUN:   --split-functions --split-strategy=all --split-all-cold --enable-bat

## Prepare pre-aggregated profile using %t.bat
# RUN: link_fdata %s %t.bat %t.preagg PREAGG
## Strip labels used for pre-aggregated profile
# RUN: llvm-strip -NLcall -NLindcall %t.bat

## Convert pre-aggregated profile using BAT
# RUN: perf2bolt %t.bat -p %t.preagg --pa -o %t.bat.fdata -w %t.bat.yaml

## Convert BAT fdata into YAML
# RUN: llvm-bolt %t.exe -data %t.bat.fdata -w %t.bat.fdata-yaml -o /dev/null

## Check fdata YAML - make sure that a direct call has discriminator field
# RUN: FileCheck %s --input-file %t.bat.fdata-yaml -check-prefix CHECK-BAT-YAML

## Check BAT YAML - make sure that a direct call has discriminator field
# RUN: FileCheck %s --input-file %t.bat.yaml --check-prefix CHECK-BAT-YAML

## YAML BAT test of calling BAT secondary entry from BAT function
# RUN: llvm-bolt %t.exe -o %t.bat2 --data %t.fdata --funcs=main,func \
# RUN:   --split-functions --split-strategy=all --split-all-cold --enable-bat

## Prepare pre-aggregated profile using %t.bat
# RUN: link_fdata %s %t.bat2 %t.preagg2 PREAGG2

## Strip labels used for pre-aggregated profile
# RUN: llvm-strip -NLcall -NLindcall %t.bat2

## Convert pre-aggregated profile using BAT
# RUN: perf2bolt %t.bat2 -p %t.preagg2 --pa -o %t.bat2.fdata -w %t.bat2.yaml

## Convert BAT fdata into YAML
# RUN: llvm-bolt %t.exe -data %t.bat2.fdata -w %t.bat2.fdata-yaml -o /dev/null

## Check fdata YAML - make sure that a direct call has discriminator field
# RUN: FileCheck %s --input-file %t.bat2.fdata-yaml -check-prefix CHECK-BAT-YAML

## Check BAT YAML - make sure that a direct call has discriminator field
# RUN: FileCheck %s --input-file %t.bat2.yaml --check-prefix CHECK-BAT-YAML

# CHECK-BAT-YAML:      - name:    main
# CHECK-BAT-YAML-NEXT:   fid:     [[#]]
# CHECK-BAT-YAML-NEXT:   hash:    0xADF270D550151185
# CHECK-BAT-YAML-NEXT:   exec:    0
# CHECK-BAT-YAML-NEXT:   nblocks: 4
# CHECK-BAT-YAML-NEXT:   blocks:
# CHECK-BAT-YAML:          - bid:   1
# CHECK-BAT-YAML-NEXT:       insns: [[#]]
# CHECK-BAT-YAML-NEXT:       hash:  0x36A303CBA4360018
# CHECK-BAT-YAML-NEXT:       calls: [ { off: 0x0, fid: [[#]], disc: 1, cnt: 1

.globl func
.type	func, @function
func:
# FDATA: 0 [unknown] 0 1 func 0 1 0
# PREAGG: B X:0 #func# 1 1
# PREAGG2: B X:0 #func# 1 1
  .cfi_startproc
  pushq   %rbp
  movq    %rsp, %rbp
## Placeholder code to make splitting profitable
.rept 5
  testq   %rax, %rax
.endr
.globl secondary_entry
secondary_entry:
## Placeholder code to make splitting profitable
.rept 5
  testq   %rax, %rax
.endr
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
.globl Lcall
Lcall:
  call    secondary_entry
# FDATA: 1 main #Lcall# 1 secondary_entry 0 1 1
# PREAGG: B #Lcall# #secondary_entry# 1 1
# PREAGG2: B #main.cold.0# #func.cold.0# 1 1
.globl Lindcall
Lindcall:
  callq   *%rax
# FDATA: 1 main #Lindcall# 1 secondary_entry 0 1 1
# PREAGG: B #Lindcall# #secondary_entry# 1 1
# PREAGG2: B #main.cold.1# #func.cold.0# 1 1
  xorl    %eax, %eax
  addq    $16, %rsp
  popq    %rbp
  retq
## For relocations against .text
  call exit
  .cfi_endproc
  .size	main, .-main
