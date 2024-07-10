# REQUIRES: aarch64
## Prior to Android V, there was a bug that caused RELR relocations to be
## applied after packed relocations. This meant that resolvers referenced by
## IRELATIVE relocations in the packed relocation section would read unrelocated
## globals when --pack-relative-relocs=android+relr is enabled. Work around this
## by placing IRELATIVE in .rela.plt.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-android a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-android b.s -o b.o
# RUN: ld.lld -shared b.o -o b.so
# RUN: ld.lld -pie --pack-dyn-relocs=android+relr -z separate-loadable-segments a.o b.so -o a
# RUN: llvm-readobj -r a | FileCheck %s
# RUN: llvm-objdump -d a | FileCheck %s --check-prefix=ASM

# CHECK:      .relr.dyn {
# CHECK-NEXT:   0x30000 R_AARCH64_RELATIVE -
# CHECK-NEXT: }
# CHECK:      .rela.plt {
# CHECK-NEXT:   0x30020 R_AARCH64_JUMP_SLOT bar 0x0
# CHECK-NEXT:   0x30028 R_AARCH64_IRELATIVE - 0x10000
# CHECK-NEXT: }

# ASM:      <.iplt>:
# ASM-NEXT:   adrp    x16, 0x30000
# ASM-NEXT:   ldr     x17, [x16, #0x28]
# ASM-NEXT:   add     x16, x16, #0x28
# ASM-NEXT:   br      x17

#--- a.s
.text
.type foo, %gnu_indirect_function
.globl foo
foo:
  ret

.globl _start
_start:
  bl foo
  bl bar

.data
.balign 8
.quad .data

#--- b.s
.globl bar
bar:
  ret
