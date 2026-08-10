# REQUIRES: x86
## Test st_value of the STT_SECTION symbol equals the output section address,
## instead of the first input section address.

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/asm -o %t.o

# RUN: ld.lld --emit-relocs -T %t/lds %t.o -o %t.out
# RUN: llvm-readelf -S -r -s %t.out | FileCheck %s --check-prefix=EXE

## In -r mode, section addresses are zeros, hence the st_value fields of
## STT_SECTION are zeros.
# RUN: ld.lld -r -T %t/lds %t.o -o %t.ro
# RUN: llvm-readelf -S -r -s %t.ro | FileCheck %s --check-prefix=RO

# EXE:      [Nr] Name  Type      Address
# EXE-NEXT: [ 0]
# EXE-NEXT: [ 1] .text PROGBITS  0000000000000000
# EXE-NEXT: [ 2] .bss  NOBITS    000000000000000a

# EXE:      R_X86_64_64 {{.*}} .bss + 1

# EXE:      Symbol table '.symtab' contains 4 entries:
# EXE-NEXT: Num:    Value          Size Type    Bind   Vis       Ndx Name
# EXE-NEXT:   0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
# EXE-NEXT:   1: 000000000000000a     0 SECTION LOCAL  DEFAULT     2 .bss
# EXE-NEXT:   2: 0000000000000000     0 SECTION LOCAL  DEFAULT     1 .text
# EXE-NEXT:   3: 0000000000000000     0 SECTION LOCAL  DEFAULT     4 .comment

# RO:       [Nr] Name  Type      Address
# RO-NEXT:  [ 0]
# RO-NEXT:  [ 1] .bss  NOBITS    0000000000000000

# RO:       R_X86_64_64 {{.*}} .bss + 1

# RO:      Symbol table '.symtab' contains 3 entries:
# RO-NEXT: Num:    Value          Size Type    Bind   Vis       Ndx Name
# RO-NEXT:   0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
# RO-NEXT:   1: 0000000000000000     0 SECTION LOCAL  DEFAULT     1 .bss
# RO-NEXT:   2: 0000000000000000     0 SECTION LOCAL  DEFAULT     2 .text

#--- asm
movabsq .bss, %rax

.bss
.byte 0

#--- lds
SECTIONS {
  .bss : { BYTE(0) *(.bss) }
}
