## Verify that BOLT assigns target addresses using the same block alignment as
## ExecutableFileMemoryManager uses when laying out the section contents.

# REQUIRES: system-linux

## Build a regular executable for BOLT. Keep a text relocation so that BOLT
## runs in relocation mode.
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux \
# RUN:   -defsym=MAIN=1 %s -o %t.main.o
# RUN: ld.lld --emit-relocs -e _start %t.main.o -o %t.exe

## Build a minimal hugify runtime with two input sections, then give the
## sections the same name so that ELFLinkGraphBuilder represents them as two
## blocks in one JITLink section. Their input addresses make the block order
## deterministic.
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux %s \
# RUN:   -o %t.runtime.pre.o
# RUN: llvm-objcopy --remove-section .text \
# RUN:   --change-section-address .text.first=0x1000 \
# RUN:   --change-section-address .text.second=0x2000 \
# RUN:   --rename-section .text.first=.text \
# RUN:   --rename-section .text.second=.text \
# RUN:   %t.runtime.pre.o %t.runtime.o

## The first block is four bytes long and the second is 16-byte aligned. The
## memory manager therefore places the second block at section offset 16. The
## branch displacement must be 16 bytes. Assigning block addresses without the
## alignment padding would incorrectly encode a four-byte displacement.
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=0 --hugify \
# RUN:   --runtime-hugify-lib=%t.runtime.o
# RUN: llvm-objdump -d --section=.text.bolt.extra.1 %t.bolt | FileCheck %s

# CHECK: 14000004 {{.*}}b
# CHECK: d65f03c0 {{.*}}ret

.ifdef MAIN
  .text
  .globl _start
  .type _start, %function
_start:
  bl target
  ret
  .size _start, .-_start

  .globl target
  .type target, %function
target:
  ret
  .size target, .-target
.else
  .section .text.first,"ax",@progbits
  b __bolt_hugify_self

  .section .text.second,"ax",@progbits
  .p2align 4
  .globl __bolt_hugify_self
  .type __bolt_hugify_self, %function
__bolt_hugify_self:
  ret
  .size __bolt_hugify_self, .-__bolt_hugify_self
.endif
