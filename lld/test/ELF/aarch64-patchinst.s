# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/use.s -o %t/use-le.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/def.s -o %t/def-le.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/rel.s -o %t/rel-le.o

## Deactivation symbol used without being defined: instruction emitted as usual.
# RUN: ld.lld -o %t/undef-le %t/use-le.o --emit-relocs
# RUN: llvm-objdump -r %t/undef-le | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump -d %t/undef-le | FileCheck --check-prefix=UNDEF %s
# RUN: ld.lld -pie -o %t/undef-le %t/use-le.o --emit-relocs
# RUN: llvm-objdump -r %t/undef-le | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump -d %t/undef-le | FileCheck --check-prefix=UNDEF %s

## Deactivation symbol defined: instructions overwritten with NOPs.
# RUN: ld.lld -o %t/def-le %t/use-le.o %t/def-le.o --emit-relocs
# RUN: llvm-objdump -r %t/def-le | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump -d %t/def-le | FileCheck --check-prefix=DEF %s
# RUN: ld.lld -pie -o %t/def-le %t/use-le.o %t/def-le.o --emit-relocs
# RUN: llvm-objdump -r %t/def-le | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump -d %t/def-le | FileCheck --check-prefix=DEF %s

## Relocation pointing to a non-SHN_UNDEF non-SHN_ABS symbol is an error.
# RUN: not ld.lld -o %t/rel-le %t/use-le.o %t/rel-le.o 2>&1 | FileCheck --check-prefix=ERROR %s
# RUN: not ld.lld -pie -o %t/rel-le %t/use-le.o %t/rel-le.o 2>&1 | FileCheck --check-prefix=ERROR %s

## Behavior unchanged by endianness: relocation always written as little endian.
# RUN: llvm-mc -filetype=obj -triple=aarch64_be %t/use.s -o %t/use-be.o
# RUN: llvm-mc -filetype=obj -triple=aarch64_be %t/def.s -o %t/def-be.o
# RUN: llvm-mc -filetype=obj -triple=aarch64_be %t/rel.s -o %t/rel-be.o
# RUN: ld.lld -o %t/undef-be %t/use-be.o --emit-relocs
# RUN: llvm-objdump -r %t/undef-be | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump -d %t/undef-be | FileCheck --check-prefix=UNDEF %s
# RUN: ld.lld -pie -o %t/undef-be %t/use-be.o --emit-relocs
# RUN: llvm-objdump -r %t/undef-be | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump -d %t/undef-be | FileCheck --check-prefix=UNDEF %s
# RUN: ld.lld -o %t/def-be %t/use-be.o %t/def-be.o --emit-relocs
# RUN: llvm-objdump -r %t/def-be | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump -d %t/def-be | FileCheck --check-prefix=DEF %s
# RUN: ld.lld -pie -o %t/def-be %t/use-be.o %t/def-be.o --emit-relocs
# RUN: llvm-objdump -r %t/def-be | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump -d %t/def-be | FileCheck --check-prefix=DEF %s
# RUN: not ld.lld -o %t/rel-be %t/use-be.o %t/rel-be.o 2>&1 | FileCheck --check-prefix=ERROR %s
# RUN: not ld.lld -pie -o %t/rel-be %t/use-be.o %t/rel-be.o 2>&1 | FileCheck --check-prefix=ERROR %s

# RELOC:      R_AARCH64_JUMP26
# RELOC-NEXT: R_AARCH64_PATCHINST ds
# RELOC-NEXT: R_AARCH64_PATCHINST ds
# RELOC-NEXT: R_AARCH64_PATCHINST ds0+0xd503201f

#--- use.s
.weak ds
.weak ds0
# This instruction has a single relocation: the DS relocation.
# UNDEF: add x0, x1, x2
# DEF: nop
# ERROR: R_AARCH64_PATCHINST relocation against non-absolute symbol ds
.reloc ., R_AARCH64_PATCHINST, ds
add x0, x1, x2
# This instruction has two relocations: the DS relocation and the JUMP26 to f1.
# Make sure that the DS relocation takes precedence.
.reloc ., R_AARCH64_PATCHINST, ds
# UNDEF: b {{.*}} <f1>
# DEF: nop
# ERROR: R_AARCH64_PATCHINST relocation against non-absolute symbol ds
b f1
# Alternative representation: instruction opcode stored in addend.
# UNDEF: add x3, x4, x5
# DEF: nop
# ERROR: R_AARCH64_PATCHINST relocation against non-absolute symbol ds0
.reloc ., R_AARCH64_PATCHINST, ds0 + 0xd503201f
add x3, x4, x5

.section .text.f1,"ax",@progbits
f1:
ret

#--- def.s
.globl ds
ds = 0xd503201f
.globl ds0
ds0 = 0

#--- rel.s
.globl ds
ds:
.globl ds0
ds0:
