# REQUIRES: sparc
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=sparc a.s -o v8.o
# RUN: llvm-mc -filetype=obj -triple=sparc -mattr=+v8plus b.s -o v8plus.o
# RUN: llvm-mc -filetype=obj -triple=sparcv9 b.s -o v9.o
# RUN: yaml2obj us1.yaml -o us1.o
# RUN: yaml2obj us3.yaml -o us3.o
# RUN: yaml2obj hal.yaml -o hal.o

## An EM_SPARC32PLUS object holds V9 instructions in the 32-bit ABI, so it links
## with plain EM_SPARC objects. A link needing them is tagged EM_SPARC32PLUS
## with EF_SPARC_32PLUS, whatever the order the objects appear in.
# RUN: ld.lld -m elf32_sparc v8.o v8plus.o -o mixed1
# RUN: llvm-readelf -h mixed1 | FileCheck %s --check-prefix=PLUS
# RUN: ld.lld -m elf32_sparc v8plus.o v8.o -o mixed2
# RUN: llvm-readelf -h mixed2 | FileCheck %s --check-prefix=PLUS
# RUN: ld.lld -m elf32_sparc v8plus.o -o pureplus
# RUN: llvm-readelf -h pureplus | FileCheck %s --check-prefix=PLUS

# PLUS: Machine: Sparc v8+
# PLUS: Flags: 0x100, V8+ ABI{{$}}

## A link whose objects are all plain V8 keeps the base machine and no flags.
# RUN: ld.lld -m elf32_sparc v8.o -o purev8
# RUN: llvm-readelf -h purev8 | FileCheck %s --check-prefix=V8

# V8: Machine: Sparc{{$}}
# V8: Flags: 0x0{{$}}

## The extension bits name nested instruction sets, so the output takes the
## widest one any object asks for. EF_SPARC_SUN_US3 implies EF_SPARC_SUN_US1.
# RUN: ld.lld -m elf32_sparc v8.o us1.o -o us1a
# RUN: ld.lld -m elf32_sparc us1.o v8plus.o -o us1b
# RUN: llvm-readelf -h us1a us1b | FileCheck %s --check-prefix=US1

# US1: Machine: Sparc v8+
# US1: Flags: 0x300, V8+ ABI, Sun UltraSPARC I extensions{{$}}
# US1: Machine: Sparc v8+
# US1: Flags: 0x300, V8+ ABI, Sun UltraSPARC I extensions{{$}}

# RUN: ld.lld -m elf32_sparc us1.o us3.o -o us3a
# RUN: ld.lld -m elf32_sparc us3.o us1.o -o us3b
# RUN: llvm-readelf -h us3a us3b | FileCheck %s --check-prefix=US3

# US3: Flags: 0xb00, V8+ ABI, Sun UltraSPARC I extensions, Sun UltraSPARC III extensions{{$}}
# US3: Flags: 0xb00, V8+ ABI, Sun UltraSPARC I extensions, Sun UltraSPARC III extensions{{$}}

## EF_SPARC_HAL_R1 names no instruction set here, so it selects nothing and is
## not written out.
# RUN: ld.lld -m elf32_sparc v8.o hal.o -o hala
# RUN: llvm-readelf -h hala | FileCheck %s --check-prefix=PLUS

## EM_SPARCV9 is the 64-bit ABI and does not belong in a 32-bit link.
# RUN: not ld.lld -m elf32_sparc v8.o v9.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# ERR: error: v9.o is incompatible with elf32_sparc

## EM_SPARC32PLUS names the 32-bit big-endian ABI, so another class or byte
## order is malformed, as is leaving out the instruction set flag.
# RUN: yaml2obj be64.yaml -o be64.o
# RUN: yaml2obj le32.yaml -o le32.o
# RUN: yaml2obj bare.yaml -o bare.o
# RUN: not ld.lld be64.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=BE64
# RUN: not ld.lld le32.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=LE32
# RUN: not ld.lld bare.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=BARE

# BE64: error: be64.o is incompatible
# LE32: error: le32.o is incompatible
# BARE: error: bare.o: EM_SPARC32PLUS object without EF_SPARC_32PLUS

#--- a.s
.globl _start
_start:
  nop

#--- b.s
.globl g
g:
  nop

#--- us1.yaml
--- !ELF
FileHeader:
  Class:   ELFCLASS32
  Data:    ELFDATA2MSB
  Type:    ET_REL
  Machine: EM_SPARC32PLUS
  Flags:   [ EF_SPARC_32PLUS, EF_SPARC_SUN_US1 ]

#--- us3.yaml
--- !ELF
FileHeader:
  Class:   ELFCLASS32
  Data:    ELFDATA2MSB
  Type:    ET_REL
  Machine: EM_SPARC32PLUS
  Flags:   [ EF_SPARC_32PLUS, EF_SPARC_SUN_US3 ]

#--- hal.yaml
--- !ELF
FileHeader:
  Class:   ELFCLASS32
  Data:    ELFDATA2MSB
  Type:    ET_REL
  Machine: EM_SPARC32PLUS
  Flags:   [ EF_SPARC_32PLUS, EF_SPARC_HAL_R1 ]

#--- be64.yaml
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2MSB
  Type:    ET_REL
  Machine: EM_SPARC32PLUS
  Flags:   [ EF_SPARC_32PLUS, EF_SPARC_SUN_US1 ]

#--- le32.yaml
--- !ELF
FileHeader:
  Class:   ELFCLASS32
  Data:    ELFDATA2LSB
  Type:    ET_REL
  Machine: EM_SPARC32PLUS
  Flags:   [ EF_SPARC_32PLUS, EF_SPARC_SUN_US1 ]

#--- bare.yaml
--- !ELF
FileHeader:
  Class:   ELFCLASS32
  Data:    ELFDATA2MSB
  Type:    ET_REL
  Machine: EM_SPARC32PLUS
