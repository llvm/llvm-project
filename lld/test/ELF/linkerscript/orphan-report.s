# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { .text : { *(.text.1) } }" > %t.script
# RUN: ld.lld --hash-style=sysv -shared -o %t.out --script %t.script %t.o --verbose | FileCheck %s

# CHECK:      {{.*}}.o:(.text) is being placed in '.text'
# CHECK-NEXT: {{.*}}.o:(.text.2) is being placed in '.text'
# CHECK-NEXT: <internal>:(.comment) is being placed in '.comment'
# CHECK-NEXT: <internal>:(.bss) is being placed in '.bss'
# CHECK-NEXT: <internal>:(.bss.rel.ro) is being placed in '.bss.rel.ro'
# CHECK-NEXT: <internal>:(.dynsym) is being placed in '.dynsym'
# CHECK-NEXT: <internal>:(.gnu.version) is being placed in '.gnu.version'
# CHECK-NEXT: <internal>:(.gnu.version_r) is being placed in '.gnu.version_r'
# CHECK-NEXT: <internal>:(.hash) is being placed in '.hash'
# CHECK-NEXT: <internal>:(.dynamic) is being placed in '.dynamic'
# CHECK-NEXT: <internal>:(.dynstr) is being placed in '.dynstr'
# CHECK-NEXT: <internal>:(.rela.dyn) is being placed in '.rela.dyn'
# CHECK-NEXT: <internal>:(.got) is being placed in '.got'
# CHECK-NEXT: <internal>:(.got.plt) is being placed in '.got.plt'
# CHECK-NEXT: <internal>:(.got.plt) is being placed in '.got.plt'
# CHECK-NEXT: <internal>:(.rela.plt) is being placed in '.rela.plt'
# CHECK-NEXT: <internal>:(.rela.plt) is being placed in '.rela.plt'
# CHECK-NEXT: <internal>:(.plt) is being placed in '.plt'
# CHECK-NEXT: <internal>:(.plt) is being placed in '.plt'
# CHECK-NEXT: <internal>:(.eh_frame) is being placed in '.eh_frame'
# CHECK-NEXT: <internal>:(.symtab) is being placed in '.symtab'
# CHECK-NEXT: <internal>:(.shstrtab) is being placed in '.shstrtab'
# CHECK-NEXT: <internal>:(.strtab) is being placed in '.strtab'

.section .text.1,"a"
 nop

.section .text.2,"a"
 nop
