# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck %s --check-prefix=OBJ
# RUN: ld.lld %t.o -o %t -pie
# RUN: llvm-objdump -s --section=.foobar --section=.got -r -d -t \
# RUN:   --dynamic-reloc %t | FileCheck %s --check-prefixes=CHECK,REL
# RUN: ld.lld %t.o -o %t-rela -pie -z rela
# RUN: llvm-objdump -s --section=.foobar --section=.got -r -d -t \
# RUN:   --dynamic-reloc %t-rela | FileCheck %s --check-prefixes=CHECK,RELA

# Unlike bfd and gold we accept this.

# OBJ:      Relocations [
# OBJ-NEXT:   Section (4) .rel.foobar {
# OBJ-NEXT:     0x2 R_386_GOT32 foo
# OBJ-NEXT:   }
# OBJ-NEXT: ]

# CHECK-LABEL: SYMBOL TABLE:
# REL:  00001180 l       .text	00000000 foo
# REL:  00002180 g       .foobar	00000000 _start
# RELA: 00001188 l       .text	00000000 foo
# RELA: 00002188 g       .foobar	00000000 _start

# CHECK-LABEL: DYNAMIC RELOCATION RECORDS
# REL-NEXT:  00002182 R_386_RELATIVE *ABS*{{$}}
# REL-NEXT:  000031f0 R_386_RELATIVE *ABS*{{$}}
# RELA-NEXT: 0000218a R_386_RELATIVE *ABS*+0x31f8{{$}}
# RELA-NEXT: 000031f8 R_386_RELATIVE *ABS*+0x1188{{$}}
# CHECK-NEXT: Contents of section .foobar:
# REL-NEXT:   2180 8b1df031 0000
##                     ^--- VA of GOT entry (0x31f0)
# RELA-NEXT:  2188 8b1d0000 0000
##                     ^--- VA of GOT entry in Elf_Rela addend
# CHECK-NEXT: Contents of section .got:
# REL-NEXT:   31f0 80110000
##                 ^--- VA of foo (0x1180)
# RELA-NEXT:  31f8 00000000
##                 ^--- VA of foo in Elf_Rela addend

# CHECK-LABEL: Disassembly of section .foobar:
# CHECK:     <_start>:
# REL-NEXT:  2180: 8b 1d f0 31 00 00 movl 12784, %ebx
##                       ^--- VA of GOT entry (0x31f0)
# RELA-NEXT: 2188: 8b 1d 00 00 00 00 movl 0, %ebx
##                       ^--- VA of GOT entry in in Elf_Rela addend

foo:

.section .foobar, "awx"
.global _start
_start:
 movl foo@GOT, %ebx
