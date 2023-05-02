# REQUIRES: avr
# RUN: llvm-mc -filetype=obj -triple=avr-unknown-linux -mcpu=atmega328p %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -Ttext=0
# RUN: llvm-objdump -d %t.exe --mcpu=atmega328 | FileCheck %s --check-prefix=ASM
# RUN: llvm-readelf --file-headers --sections -l --symbols %t.exe \
# RUN:     | FileCheck %s --check-prefix=ELF

main:
  call  foo
foo:
  jmp   foo
  rcall foo
  rjmp  foo

# ASM:      <main>:
# ASM-NEXT:   0: 0e 94 02 00  call  0x4
# ASM:      <foo>:
# ASM-NEXT:   4: 0c 94 02 00  jmp   0x4
# ASM-NEXT:   8: fd df        rcall .-6
# ASM-NEXT:   a: fc cf        rjmp  .-8

# ELF:      ELF Header:
# ELF-NEXT:   Magic:   7f 45 4c 46 01 01 01 00 00 00 00 00 00 00 00 00
# ELF-NEXT:   Class:                             ELF32
# ELF-NEXT:   Data:                              2's complement, little endian
# ELF-NEXT:   Version:                           1 (current)
# ELF-NEXT:   OS/ABI:                            UNIX - System V
# ELF-NEXT:   ABI Version:                       0
# ELF-NEXT:   Type:                              EXEC (Executable file)
# ELF-NEXT:   Machine:                           Atmel AVR 8-bit microcontroller
# ELF-NEXT:   Version:                           0x1
# ELF-NEXT:   Entry point address:               0x0
# ELF-NEXT:   Start of program headers:          52 (bytes into file)
# ELF-NEXT:   Start of section headers:          4216 (bytes into file)
# ELF-NEXT:   Flags:                             0x85, EF_AVR_ARCH_AVR5, relaxable
# ELF-NEXT:   Size of this header:               52 (bytes)
# ELF-NEXT:   Size of program headers:           32 (bytes)
# ELF-NEXT:   Number of program headers:         4
# ELF-NEXT:   Size of section headers:           40 (bytes)
# ELF-NEXT:   Number of section headers:         6
# ELF-NEXT:   Section header string table index: 4
# ELF-NEXT: There are 6 section headers, starting at offset 0x1078:

# ELF:      Section Headers:
# ELF-NEXT:   [Nr] Name              Type            Address  Off    Size   ES Flg Lk Inf Al
# ELF-NEXT:   [ 0]                   NULL            00000000 000000 000000 00      0   0  0
# ELF-NEXT:   [ 1] .text             PROGBITS        00000000 001000 00000c 00  AX  0   0  4
# ELF-NEXT:   [ 2] .comment          PROGBITS        00000000 00100c 000008 01  MS  0   0  1
# ELF-NEXT:   [ 3] .symtab           SYMTAB          00000000 001014 000030 10      5   3  4
# ELF-NEXT:   [ 4] .shstrtab         STRTAB          00000000 001044 00002a 00      0   0  1
# ELF-NEXT:   [ 5] .strtab           STRTAB          00000000 00106e 00000a 00      0   0  1
# ELF-NEXT: Key to Flags:
# ELF-NEXT:   W (write), A (alloc), X (execute), M (merge), S (strings), I (info),
# ELF-NEXT:   L (link order), O (extra OS processing required), G (group), T (TLS),
# ELF-NEXT:   C (compressed), x (unknown), o (OS specific), E (exclude),
# ELF-NEXT:   R (retain), p (processor specific)

# ELF:      Elf file type is EXEC (Executable file)
# ELF-NEXT: Entry point 0x0
# ELF-NEXT: There are 4 program headers, starting at offset 52

# ELF:      Program Headers:
# ELF-NEXT:   Type           Offset   VirtAddr   PhysAddr   FileSiz MemSiz  Flg Align
# ELF-NEXT:   PHDR           0x000034 0x00010034 0x00010034 0x00080 0x00080 R   0x4
# ELF-NEXT:   LOAD           0x000000 0x00010000 0x00010000 0x000b4 0x000b4 R   0x1000
# ELF-NEXT:   LOAD           0x001000 0x00000000 0x00000000 0x0000c 0x0000c R E 0x1000
# ELF-NEXT:   GNU_STACK      0x000000 0x00000000 0x00000000 0x00000 0x00000 RW  0x0

# ELF:      Section to Segment mapping:
# ELF-NEXT:   Segment Sections...
# ELF-NEXT:    00
# ELF-NEXT:    01
# ELF-NEXT:    02     .text
# ELF-NEXT:    03
# ELF-NEXT:    None   .comment .symtab .shstrtab .strtab

# ELF:      Symbol table '.symtab' contains 3 entries:
# ELF-NEXT:    Num:    Value  Size Type    Bind   Vis       Ndx Name
# ELF-NEXT:      0: 00000000     0 NOTYPE  LOCAL  DEFAULT   UND
# ELF-NEXT:      1: 00000000     0 NOTYPE  LOCAL  DEFAULT     1 main
# ELF-NEXT:      2: 00000004     0 NOTYPE  LOCAL  DEFAULT     1 foo
