## Check CONDBR19 conditional tail calls from cold _start to hot foo in lite mode.

# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-unknown --defsym PAD=0 %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: ld.lld --emit-relocs %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --lite --no-huge-pages
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefixes=COMMON,NEAR
# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --lite --no-huge-pages \
# RUN:   --compact-code-model
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefixes=COMMON,NEAR

## Add 1MB of padding after _start to put the relocated foo out of range of CONDBR19.
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-unknown --defsym PAD=0x100000 %s -o %t.o
# RUN: ld.lld --emit-relocs %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --lite --no-huge-pages
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefixes=COMMON,FAR
# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --lite --no-huge-pages \
# RUN:   --compact-code-model
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefixes=COMMON,FAR

# FDATA: 0 [unknown] 0 1 foo 0 0 100

# COMMON: Disassembly of section .bolt.org.text:
# COMMON: [[#%x,FOO_OLD:]] <foo.org.0>:
# COMMON-NEXT: {{.*}} adrp x16, 0x[[#%x,FOO:]] <foo>
# COMMON-NEXT: {{.*}} add x16, x16, #0x0
# COMMON-NEXT: {{.*}} br x16
# COMMON: <_start>:

# NEAR-NEXT: {{.*}} b.ge 0x[[#FOO]] <foo>
# NEAR-NEXT: {{.*}} b.pl 0x[[#FOO]] <foo>
# NEAR-NEXT: {{.*}} cbz w0, 0x[[#FOO]] <foo>
# NEAR-NEXT: {{.*}} cbz x21, 0x[[#FOO]] <foo>
# NEAR-NEXT: {{.*}} cbnz x10, 0x[[#FOO]] <foo>
# NEAR-NEXT: {{.*}} cbnz wzr, 0x[[#FOO]] <foo>

# FAR-NEXT: {{.*}} b.ge 0x[[#FOO_OLD]] <foo.org.0>
# FAR-NEXT: {{.*}} b.pl 0x[[#FOO_OLD]] <foo.org.0>
# FAR-NEXT: {{.*}} cbz w0, 0x[[#FOO_OLD]] <foo.org.0>
# FAR-NEXT: {{.*}} cbz x21, 0x[[#FOO_OLD]] <foo.org.0>
# FAR-NEXT: {{.*}} cbnz x10, 0x[[#FOO_OLD]] <foo.org.0>
# FAR-NEXT: {{.*}} cbnz wzr, 0x[[#FOO_OLD]] <foo.org.0>

# COMMON-NEXT: {{.*}} ret
# COMMON: Disassembly of section .text:
# COMMON: [[#FOO]] <foo>:
# COMMON-NEXT: {{.*}} ret

    .type foo,@function
    .globl foo
foo:
  .rept 3
    nop
  .endr
  ret
  .size foo, .-foo

    .type _start,@function
    .globl _start
_start:
  b.ge foo
  b.pl foo
  cbz w0, foo
  cbz x21, foo
  cbnz x10, foo
  cbnz wzr, foo
  ret
  .size _start, .-_start

## Keep both input functions adjacent so their original branches stay in range.
.space PAD
