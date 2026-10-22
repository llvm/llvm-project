## Check that TSTBR14 branches in skipped original code reach relocated foo
## directly when in range and through its patched original entry when out of
## range. Exercise in-range and out-of-range layouts in both code models.

## --lite leaves cold _start at its original address and moves hot foo.
## --no-huge-pages reduces page alignment from 2MB to 64KB.
## --section-start puts the original text 4KB before a 64KB boundary.
## --align-text=0x1000 keeps unpadded foo within TSTBR14's 32KB range.

# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-unknown --defsym PAD=0 %s -o %t.o
# RUN: ld.lld --emit-relocs --section-start=.text=0x3ff000 %t.o -o %t.exe
# RUN: link_fdata %s %t.exe %t.fdata
# RUN: llvm-strip --strip-unneeded %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --lite --no-huge-pages \
# RUN:   --align-text=0x1000
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefixes=COMMON,NEAR
# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --lite --no-huge-pages \
# RUN:   --align-text=0x1000 --compact-code-model
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefixes=COMMON,NEAR

## Add 32KB of padding after _start to put the relocated foo out of range of TSTBR14.
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-unknown --defsym PAD=0x8000 %s -o %t.o
# RUN: ld.lld --emit-relocs --section-start=.text=0x3ff000 %t.o -o %t.exe
# RUN: link_fdata %s %t.exe %t.fdata
# RUN: llvm-strip --strip-unneeded %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --lite --no-huge-pages \
# RUN:   --align-text=0x1000
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefixes=COMMON,FAR
# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --lite --no-huge-pages \
# RUN:   --align-text=0x1000 --compact-code-model
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefixes=COMMON,FAR

# COMMON: Disassembly of section .bolt.org.text:
# COMMON: [[#%x,FOO_OLD:]] <foo.org.0>:
# COMMON-NEXT: {{.*}} adrp x16, 0x[[#%x,FOO:]] <foo>
# COMMON-NEXT: {{.*}} add x16, x16, #0x0
# COMMON-NEXT: {{.*}} br x16
# COMMON: <_start>:

# NEAR-NEXT: {{.*}} tbz w0, #0x0, 0x[[#FOO]] <foo>
# NEAR-NEXT: {{.*}} tbz x10, #0x20, 0x[[#FOO]] <foo>
# NEAR-NEXT: {{.*}} tbnz w21, #0x1f, 0x[[#FOO]] <foo>
# NEAR-NEXT: {{.*}} tbnz xzr, #0x3f, 0x[[#FOO]] <foo>

# FAR-NEXT: {{.*}} tbz w0, #0x0, 0x[[#FOO_OLD]] <foo.org.0>
# FAR-NEXT: {{.*}} tbz x10, #0x20, 0x[[#FOO_OLD]] <foo.org.0>
# FAR-NEXT: {{.*}} tbnz w21, #0x1f, 0x[[#FOO_OLD]] <foo.org.0>
# FAR-NEXT: {{.*}} tbnz xzr, #0x3f, 0x[[#FOO_OLD]] <foo.org.0>

# COMMON-NEXT: {{.*}} ret
# COMMON: Disassembly of section .text:
# COMMON: [[#FOO]] <foo>:
# COMMON-NEXT: {{.*}} ret

    .type foo,@function
    .globl foo
foo:
# FDATA: 0 [unknown] 0 1 foo 0 0 100
  .rept 3
    nop
  .endr
  ret
  .size foo, .-foo

    .type _start,@function
    .globl _start
_start:
  tbz w0, #0, foo
  tbz x10, #32, foo
  tbnz w21, #31, foo
  tbnz xzr, #63, foo
  ret
  .size _start, .-_start

## Keep both input functions adjacent so their original branches stay in range.
.space PAD
