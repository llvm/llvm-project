# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o

# RUN: echo "SECTIONS { /DISCARD/ : { *(.shstrtab) } }" > %t.script
# RUN: not ld.lld -o /dev/null --script %t.script %t.o 2>&1 | \
# RUN:   FileCheck -check-prefix=SHSTRTAB %s
# SHSTRTAB: discarding .shstrtab section is not allowed

## We allow discarding .dynamic, check we don't crash.
# RUN: echo "SECTIONS { /DISCARD/ : { *(.dynamic) } }" > %t.script
# RUN: ld.lld -pie -o %t --script %t.script %t.o

## We allow discarding .dynsym, check we don't crash.
# RUN: echo "SECTIONS { /DISCARD/ : { *(.dynsym) } }" > %t.script
# RUN: ld.lld -pie -o %t --script %t.script %t.o

## We allow discarding .dynstr, check we don't crash.
# RUN: echo "SECTIONS { /DISCARD/ : { *(.dynstr) } }" > %t.script
# RUN: ld.lld -pie -o %t --script %t.script %t.o

# RUN: echo "SECTIONS { /DISCARD/ : { *(.rela.dyn) } }" > %t.script
# RUN: ld.lld -pie -o %t %t.o
# RUN: llvm-readobj -S %t | FileCheck --check-prefix=RELADYN %s
# RELADYN: Name: .rela.dyn
# RUN: ld.lld -pie -o %t --script %t.script %t.o
# RUN: llvm-readobj -S %t | FileCheck /dev/null --implicit-check-not='Name: .rela.dyn'

# RUN: echo "SECTIONS { /DISCARD/ : { *(.relr.dyn) } }" > %t.script
# RUN: ld.lld -pie --pack-dyn-relocs=relr -T %t.script %t.o -o %t
# RUN: llvm-readobj -S -r %t | FileCheck /dev/null --implicit-check-not='Name: .relr.dyn' --implicit-check-not=R_X86_64_RELATIVE

.data
.align 8
foo:
## Emits an R_X86_64_RELATIVE in -pie mode.
.quad foo
