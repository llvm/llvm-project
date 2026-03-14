# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: ld.lld --gdb-index %t -o /dev/null 2>&1 | FileCheck %s

# CHECK: warning: {{.*}}(.debug_gnu_pubnames): name lookup table at offset 0x0 parsing failed: unexpected end of data at offset 0x1 while reading [0x0, 0x4)

    .section .debug_abbrev,"",@progbits
    .byte 1                         # Abbreviation Code
    .byte 17                        # DW_TAG_compile_unit
    .byte 0                         # DW_CHILDREN_no
    .byte 0                         # EOM(1)
    .byte 0                         # EOM(2)
    .byte 0                         # EOM(3)

    .section .debug_info,"",@progbits
.LCUBegin:
    .long .LUnitEnd-.LUnitBegin     # Length of Unit
.LUnitBegin:
    .short 4                        # DWARF version number
    .long .debug_abbrev             # Offset Into Abbrev. Section
    .byte 8                         # Address Size (in bytes)
    .byte 1                         # Abbrev [1] DW_TAG_compile_unit
.LUnitEnd:

    .section .debug_gnu_pubnames,"",@progbits
    .byte 0
