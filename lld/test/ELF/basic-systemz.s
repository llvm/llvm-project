# REQUIRES: systemz
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %s -o %t.o
# RUN: ld.lld --hash-style=sysv -discard-all -shared %t.o -o %t.so
# RUN: llvm-readelf --file-header --program-headers --section-headers --dynamic-table %t.so | FileCheck %s

# Exits with return code 55 on linux.
.text
  lghi 2,55
  svc 1

# CHECK:       ELF Header:
# CHECK-NEXT:  Magic:   7f 45 4c 46 02 02 01 00 00 00 00 00 00 00 00 00
# CHECK-NEXT:  Class:                             ELF64
# CHECK-NEXT:  Data:                              2's complement, big endian
# CHECK-NEXT:  Version:                           1 (current)
# CHECK-NEXT:  OS/ABI:                            UNIX - System V
# CHECK-NEXT:  ABI Version:                       0
# CHECK-NEXT:  Type:                              DYN (Shared object file)
# CHECK-NEXT:  Machine:                           IBM S/390
# CHECK-NEXT:  Version:                           0x1
# CHECK-NEXT:  Entry point address:               0x0
# CHECK-NEXT:  Start of program headers:          64 (bytes into file)
# CHECK-NEXT:  Start of section headers:          768 (bytes into file)
# CHECK-NEXT:  Flags:                             0x0
# CHECK-NEXT:  Size of this header:               64 (bytes)
# CHECK-NEXT:  Size of program headers:           56 (bytes)
# CHECK-NEXT:  Number of program headers:         7
# CHECK-NEXT:  Size of section headers:           64 (bytes)
# CHECK-NEXT:  Number of section headers:         11
# CHECK-NEXT:  Section header string table index: 9

# CHECK:       Section Headers:
# CHECK-NEXT:  [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK-NEXT:  [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
# CHECK-NEXT:  [ 1] .dynsym           DYNSYM          00000000000001c8 0001c8 000018 18   A  3   1  8
# CHECK-NEXT:  [ 2] .hash             HASH            00000000000001e0 0001e0 000010 04   A  1   0  4
# CHECK-NEXT:  [ 3] .dynstr           STRTAB          00000000000001f0 0001f0 000001 00   A  0   0  1
# CHECK-NEXT:  [ 4] .text             PROGBITS        00000000000011f4 0001f4 000006 00  AX  0   0  4
# CHECK-NEXT:  [ 5] .dynamic          DYNAMIC         0000000000002200 000200 000060 10  WA  3   0  8
# CHECK-NEXT:  [ 6] .relro_padding    NOBITS          0000000000002260 000260 000da0 00  WA  0   0  1
# CHECK-NEXT:  [ 7] .comment          PROGBITS        0000000000000000 000260 000008 01  MS  0   0  1
# CHECK-NEXT:  [ 8] .symtab           SYMTAB          0000000000000000 000268 000030 18     10   2  8
# CHECK-NEXT:  [ 9] .shstrtab         STRTAB          0000000000000000 000298 000058 00      0   0  1
# CHECK-NEXT:  [10] .strtab           STRTAB          0000000000000000 0002f0 00000a 00      0   0  1

# CHECK:       Program Headers:
# CHECK-NEXT:  Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK-NEXT:  PHDR           0x000040 0x0000000000000040 0x0000000000000040 0x000188 0x000188 R   0x8
# CHECK-NEXT:  LOAD           0x000000 0x0000000000000000 0x0000000000000000 0x0001f1 0x0001f1 R   0x1000
# CHECK-NEXT:  LOAD           0x0001f4 0x00000000000011f4 0x00000000000011f4 0x000006 0x000006 R E 0x1000
# CHECK-NEXT:  LOAD           0x000200 0x0000000000002200 0x0000000000002200 0x000060 0x000e00 RW  0x1000
# CHECK-NEXT:  DYNAMIC        0x000200 0x0000000000002200 0x0000000000002200 0x000060 0x000060 RW  0x8
# CHECK-NEXT:  GNU_RELRO      0x000200 0x0000000000002200 0x0000000000002200 0x000060 0x000e00 R   0x1
# CHECK-NEXT:  GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0x0

# CHECK:       Dynamic section at offset 0x200 contains 6 entries:
# CHECK-NEXT:  Tag                Type     Name/Value
# CHECK-NEXT:  0x0000000000000006 (SYMTAB) 0x1c8
# CHECK-NEXT:  0x000000000000000b (SYMENT) 24 (bytes)
# CHECK-NEXT:  0x0000000000000005 (STRTAB) 0x1f0
# CHECK-NEXT:  0x000000000000000a (STRSZ)  1 (bytes)
# CHECK-NEXT:  0x0000000000000004 (HASH)   0x1e0
# CHECK-NEXT:  0x0000000000000000 (NULL)   0x0
