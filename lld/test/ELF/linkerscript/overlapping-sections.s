# TODO: maybe this should be converted to an x86 test to get more buildbot coverage
# REQUIRES: mips
# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux %s -o %t.o

# RUN: echo "SECTIONS { \
# RUN:   .sec1 0x8000 : { sec1_start = .; *(.first_sec) sec1_end = .;} \
# RUN:   .sec2 0x8800 : { sec2_start = .; *(.second_sec) sec2_end = .;} \
# RUN: }" > %t.script
# RUN: ld.lld -o %t.so --script %t.script %t.o -shared
# RUN: llvm-readobj -sections -program-headers %t.so | FileCheck %s -check-prefix GOOD

# GOOD:        Name: .sec1
# GOOD-NEXT:   Type: SHT_PROGBITS (0x1)
# GOOD-NEXT:   Flags [ (0x3)
# GOOD-NEXT:     SHF_ALLOC (0x2)
# GOOD-NEXT:     SHF_WRITE (0x1)
# GOOD-NEXT:   ]
# GOOD-NEXT:   Address: 0x8000
# GOOD-NEXT:   Offset: 0x18000
# GOOD-NEXT:   Size: 256

# GOOD:        Name: .sec2
# GOOD-NEXT:   Type: SHT_PROGBITS (0x1)
# GOOD-NEXT:   Flags [ (0x3)
# GOOD-NEXT:     SHF_ALLOC (0x2)
# GOOD-NEXT:     SHF_WRITE (0x1)
# GOOD-NEXT:   ]
# GOOD-NEXT:   Address: 0x8800
# GOOD-NEXT:   Offset: 0x18800
# GOOD-NEXT:   Size: 256

# GOOD:      ProgramHeaders [
# GOOD-NEXT:  ProgramHeader {
# GOOD-NEXT:    Type: PT_LOAD (0x1)
# GOOD-NEXT:    Offset: 0x10000
# GOOD-NEXT:    VirtualAddress: 0x0
# GOOD-NEXT:    PhysicalAddress: 0x0
# GOOD-NEXT:    FileSize: 481
# GOOD-NEXT:    MemSize: 481
# GOOD-NEXT:    Flags [ (0x5)
# GOOD-NEXT:      PF_R (0x4)
# GOOD-NEXT:      PF_X (0x1)
# GOOD-NEXT:    ]
# GOOD-NEXT:    Alignment: 65536
# GOOD-NEXT:  }
# GOOD-NEXT:  ProgramHeader {
# GOOD-NEXT:    Type: PT_LOAD (0x1)
# GOOD-NEXT:    Offset: 0x18000
# GOOD-NEXT:    VirtualAddress: 0x8000
# GOOD-NEXT:    PhysicalAddress: 0x8000
# GOOD-NEXT:    FileSize: 2320
# GOOD-NEXT:    MemSize: 2320
# GOOD-NEXT:    Flags [ (0x6)
# GOOD-NEXT:      PF_R (0x4)
# GOOD-NEXT:      PF_W (0x2)
# GOOD-NEXT:    ]
# GOOD-NEXT:    Alignment: 65536
# GOOD-NEXT:  }

# RUN: echo "SECTIONS { \
# RUN:   .sec1 0x8000 : AT(0x8000) { sec1_start = .; *(.first_sec) sec1_end = .;} \
# RUN:   .sec2 0x8800 : AT(0x8080) { sec2_start = .; *(.second_sec) sec2_end = .;} \
# RUN: }" > %t-lma.script
# RUN: not ld.lld -o %t.so --script %t-lma.script %t.o -shared 2>&1 | FileCheck %s -check-prefix LMA-OVERLAP-ERR
# LMA-OVERLAP-ERR:      error: section .sec1 load address range overlaps with .sec2
# LMA-OVERLAP-ERR-NEXT: >>> .sec1 range is [0x8000, 0x80FF]
# LMA-OVERLAP-ERR-NEXT: >>> .sec2 range is [0x8080, 0x817F]

# Check that we create the expected binary with --noinhibit-exec or --no-check-sections:
# RUN: ld.lld -o %t.so --script %t-lma.script %t.o -shared --noinhibit-exec
# RUN: ld.lld -o %t.so --script %t-lma.script %t.o -shared --no-check-sections -fatal-warnings
# RUN: ld.lld -o %t.so --script %t-lma.script %t.o -shared --check-sections --no-check-sections -fatal-warnings

# Verify that the .sec2 was indeed placed in a PT_LOAD where the PhysAddr
# overlaps with where .sec1 is loaded:
# RUN: llvm-readobj -sections -program-headers -elf-output-style=GNU %t.so | FileCheck %s -check-prefix BAD-LMA
# BAD-LMA-LABEL: Section Headers:
# BAD-LMA: .sec1             PROGBITS        0000000000008000 018000 000100 00  WA  0   0  1
# BAD-LMA: .sec2             PROGBITS        0000000000008800 018800 000100 00  WA  0   0  1
# BAD-LMA-LABEL: Program Headers:
# BAD-LMA-NEXT:  Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# BAD-LMA-NEXT:  LOAD           0x010000 0x0000000000000000 0x0000000000000000 0x0001e1 0x0001e1 R E 0x10000
# BAD-LMA-NEXT:  LOAD           0x018000 0x0000000000008000 0x0000000000008000 0x000100 0x000100 RW  0x10000
# BAD-LMA-NEXT:  LOAD           0x018800 0x0000000000008800 0x0000000000008080 0x000110 0x000110 RW  0x10000
# BAD-LMA-LABEL: Section to Segment mapping:
# BAD-LMA:  01     .sec1
# BAD-LMA:  02     .sec2 .data .got


# Now try a script where the virtual memory addresses overlap but ensure that the
# load addresses don't:
# RUN: echo "SECTIONS { \
# RUN:   .sec1 0x8000 : AT(0x8000) { sec1_start = .; *(.first_sec) sec1_end = .;} \
# RUN:   .sec2 0x8020 : AT(0x8800) { sec2_start = .; *(.second_sec) sec2_end = .;} \
# RUN: }" > %t-vaddr.script
# RUN: not ld.lld -o %t.so --script %t-vaddr.script %t.o -shared 2>&1 | FileCheck %s -check-prefix VADDR-OVERLAP-ERR
# VADDR-OVERLAP-ERR:      error: section .sec1 virtual address range overlaps with .sec2
# VADDR-OVERLAP-ERR-NEXT: >>> .sec1 range is [0x8000, 0x80FF]
# VADDR-OVERLAP-ERR-NEXT: >>> .sec2 range is [0x8020, 0x811F]

# Check that the expected binary was created with --noinhibit-exec:
# RUN: ld.lld -o %t.so --script %t-vaddr.script %t.o -shared --noinhibit-exec
# RUN: llvm-readobj -sections -program-headers -elf-output-style=GNU %t.so | FileCheck %s -check-prefix BAD-VADDR
# BAD-VADDR-LABEL: Section Headers:
# BAD-VADDR: .sec1             PROGBITS        0000000000008000 018000 000100 00  WA  0   0  1
# BAD-VADDR: .sec2             PROGBITS        0000000000008020 028020 000100 00  WA  0   0  1
# BAD-VADDR-LABEL: Program Headers:
# BAD-VADDR-NEXT:  Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# BAD-VADDR-NEXT:  LOAD           0x010000 0x0000000000000000 0x0000000000000000 0x0001e1 0x0001e1 R E 0x10000
# BAD-VADDR-NEXT:  LOAD           0x018000 0x0000000000008000 0x0000000000008000 0x000100 0x000100 RW  0x10000
# BAD-VADDR-NEXT:  LOAD           0x028020 0x0000000000008020 0x0000000000008800 0x000110 0x000110 RW  0x10000
# BAD-VADDR-LABEL: Section to Segment mapping:
# BAD-VADDR:  01     .sec1
# BAD-VADDR:  02     .sec2 .data .got

# Finally check the case where both LMA and vaddr overlap

# RUN: echo "SECTIONS { \
# RUN:   .sec1 0x8000 : { sec1_start = .; *(.first_sec) sec1_end = .;} \
# RUN:   .sec2 0x8040 : { sec2_start = .; *(.second_sec) sec2_end = .;} \
# RUN: }" > %t-both-overlap.script

# RUN: not ld.lld -o %t.so --script %t-both-overlap.script %t.o -shared 2>&1 | FileCheck %s -check-prefix BOTH-OVERLAP-ERR

# BOTH-OVERLAP-ERR:      error: section .sec1 file range overlaps with .sec2
# BOTH-OVERLAP-ERR-NEXT: >>> .sec1 range is [0x18000, 0x180FF]
# BOTH-OVERLAP-ERR-NEXT: >>> .sec2 range is [0x18040, 0x1813F]
# BOTH-OVERLAP-ERR:      error: section .sec1 virtual address range overlaps with .sec2
# BOTH-OVERLAP-ERR-NEXT: >>> .sec1 range is [0x8000, 0x80FF]
# BOTH-OVERLAP-ERR-NEXT: >>> .sec2 range is [0x8040, 0x813F]
# BOTH-OVERLAP-ERR:      error: section .sec1 load address range overlaps with .sec2
# BOTH-OVERLAP-ERR-NEXT: >>> .sec1 range is [0x8000, 0x80FF]
# BOTH-OVERLAP-ERR-NEXT: >>> .sec2 range is [0x8040, 0x813F]

# RUN: ld.lld -o %t.so --script %t-both-overlap.script %t.o -shared --noinhibit-exec
# Note: I case everything overlaps we create a binary with overlapping file
# offsets. ld.bfd seems to place .sec1 to file offset 18000 and .sec2
# at 18100 so that only virtual addr and LMA overlap
# However, in order to create such a broken binary the user has to ignore a
# fatal error by passing --noinhibit-exec, so this behaviour is fine.

# RUN: llvm-objdump -s %t.so | FileCheck %s -check-prefix BROKEN-OUTPUT-FILE
# BROKEN-OUTPUT-FILE-LABEL: Contents of section .sec1:
# BROKEN-OUTPUT-FILE-NEXT: 8000 01010101 01010101 01010101 01010101  ................
# BROKEN-OUTPUT-FILE-NEXT: 8010 01010101 01010101 01010101 01010101  ................
# BROKEN-OUTPUT-FILE-NEXT: 8020 01010101 01010101 01010101 01010101  ................
# BROKEN-OUTPUT-FILE-NEXT: 8030 01010101 01010101 01010101 01010101  ................
# Starting here the contents of .sec2 overwrites .sec1:
# BROKEN-OUTPUT-FILE-NEXT: 8040 02020202 02020202 02020202 02020202  ................

# RUN: llvm-readobj -sections -program-headers -elf-output-style=GNU %t.so | FileCheck %s -check-prefix BAD-BOTH
# BAD-BOTH-LABEL: Section Headers:
# BAD-BOTH: .sec1             PROGBITS        0000000000008000 018000 000100 00  WA  0   0  1
# BAD-BOTH: .sec2             PROGBITS        0000000000008040 018040 000100 00  WA  0   0  1
# BAD-BOTH-LABEL: Program Headers:
# BAD-BOTH-NEXT:  Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# BAD-BOTH-NEXT:  LOAD           0x010000 0x0000000000000000 0x0000000000000000 0x0001e1 0x0001e1 R E 0x10000
# BAD-BOTH-NEXT:  LOAD           0x018000 0x0000000000008000 0x0000000000008000 0x000150 0x000150 RW  0x10000
# BAD-BOTH-LABEL: Section to Segment mapping:
# BAD-BOTH:   01     .sec1 .sec2 .data .got


.section        .first_sec,"aw",@progbits
.rept 0x100
.byte 1
.endr

.section        .second_sec,"aw",@progbits
.rept 0x100
.byte 2
.endr
