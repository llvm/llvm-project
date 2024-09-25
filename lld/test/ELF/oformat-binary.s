# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

# RUN: ld.lld -o %t.out %t --oformat=binary
# RUN: od -t x1 -v %t.out | FileCheck %s
# CHECK:      0000000 90 11 22
# CHECK-NEXT: 0000003

## OUTPUT_FORMAT(binary) selects the binary format as well.
# RUN: echo "OUTPUT_FORMAT(binary)" > %t.script
# RUN: ld.lld -o %t2.out -T %t.script %t
# RUN: od -t x1 -v %t2.out | FileCheck %s
## More OUTPUT_FORMAT commands are ignored.
# RUN: echo "OUTPUT_FORMAT("binary")OUTPUT_FORMAT(elf64-x86-64)" > %t.script
# RUN: ld.lld -o %t2.out -T %t.script %t
# RUN: od -t x1 -v %t2.out | FileCheck %s

## --oformat=binary overrides an ELF OUTPUT_FORMAT.
# RUN: echo "OUTPUT_FORMAT(elf64-x86-64) SECTIONS { . = 0x1000; }" > %t.script
# RUN: ld.lld -o %t2.out --script %t.script %t --oformat binary
# RUN: od -t x1 -v %t2.out | FileCheck %s

# RUN: echo "SECTIONS { }" > %t.script
# RUN: ld.lld -o %t2.out --script %t.script %t --oformat binary
# RUN: od -t x1 -v %t2.out | FileCheck %s

## LMA(.text)=0x100, LMA(.mysec)=0x108. The minimum LMA of all non-empty sections is 0x100.
## We place an output section at its LMA minus 0x100.
# RUN: echo 'SECTIONS { .text 0x100 : {*(.text)} .mysec ALIGN(8) : {*(.mysec*)} }' > %talign.lds
# RUN: ld.lld -T %talign.lds %t --oformat binary -o %talign
# RUN: od -Ax -t x1 %talign | FileCheck %s --check-prefix=ALIGN --ignore-case

# ALIGN:      000000 90 00 00 00 00 00 00 00 11 22
# ALIGN-NEXT: 00000a

## The empty section .data is ignored when computing the file size.
# RUN: echo 'SECTIONS { .text : {*(.text .mysec*)} .data 0x100 : {keep = .;}}' > %tempty.lds
# RUN: ld.lld -T %tempty.lds %t --oformat binary -o %tempty
# RUN: od -t x1 %tempty | FileCheck %s

## NOBITS sections are ignored as well.
## Also test that SIZEOF_HEADERS evaluates to 0.
# RUN: echo 'SECTIONS { .text : {. += SIZEOF_HEADERS; *(.text .mysec*)} .data 0x100 (NOLOAD) : {BYTE(0)}}' > %tnobits.lds
# RUN: ld.lld -T %tnobits.lds %t --oformat binary -o %tnobits
# RUN: od -t x1 %tnobits | FileCheck %s

## FIXME .mysec should be placed at file offset 1.
## This does not work because for a section without PT_LOAD, we consider LMA = VMA.
# RUN: echo 'SECTIONS { .text : {*(.text)} .mysec 0x8 : AT(1) {*(.mysec*)} }' > %tlma.lds
# RUN: ld.lld -T %tlma.lds %t --oformat binary -o %tlma
# RUN: od -Ax -t x1 %tlma | FileCheck %s --check-prefix=ALIGN --ignore-case

# RUN: not ld.lld -o /dev/null %t --oformat foo 2>&1 \
# RUN:   | FileCheck %s --check-prefix ERR
# ERR: unknown --oformat value: foo

# RUN: echo "OUTPUT_FORMAT(binary-freebsd)" > %t.script
# RUN: not ld.lld -T %t.script %t -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR2
# ERR2: error: {{.*}}.script:1: unknown output format name: binary-freebsd

# RUN: ld.lld -o /dev/null %t --oformat elf
# RUN: ld.lld -o /dev/null %t --oformat=elf-foo

.text
.align 4
.globl _start
_start:
 nop

.section        .mysec.1,"ax"
.byte   0x11

.section        .mysec.2,"ax"
.byte   0x22
