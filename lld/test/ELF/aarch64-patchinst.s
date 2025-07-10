# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/use.s -o %t/use-le.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/def.s -o %t/def-le.o

## Deactivation symbol used without being defined: instruction emitted as usual.
# RUN: ld.lld -o %t/undef-le %t/use-le.o
# RUN: llvm-objdump -d %t/undef-le | FileCheck --check-prefix=UNDEF %s

## Deactivation symbol defined: instructions overwritten with NOPs.
# RUN: ld.lld -o %t/def-le %t/use-le.o %t/def-le.o
# RUN: llvm-objdump -d %t/def-le | FileCheck --check-prefix=DEF %s

## Behavior unchanged by endianness: relocation always written as little endian.
# RUN: llvm-mc -filetype=obj -triple=aarch64_be %t/use.s -o %t/use-be.o
# RUN: llvm-mc -filetype=obj -triple=aarch64_be %t/def.s -o %t/def-be.o
# RUN: ld.lld -o %t/undef-be %t/use-be.o
# RUN: llvm-objdump -d %t/undef-be | FileCheck --check-prefix=UNDEF %s
# RUN: ld.lld -o %t/def-be %t/use-be.o %t/def-be.o
# RUN: llvm-objdump -d %t/def-be | FileCheck --check-prefix=DEF %s

#--- use.s
.weak ds
# This instruction has a single relocation: the DS relocation.
# UNDEF: add x0, x1, x2
# DEF: nop
.reloc ., R_AARCH64_PATCHINST, ds
add x0, x1, x2
# This instruction has two relocations: the DS relocation and the JUMP26 to f1.
# Make sure that the DS relocation takes precedence.
.reloc ., R_AARCH64_PATCHINST, ds
# UNDEF: b {{.*}} <f1>
# DEF: nop
b f1

.section .text.f1,"ax",@progbits
f1:
ret

#--- def.s
.globl ds
ds = 0xd503201f
