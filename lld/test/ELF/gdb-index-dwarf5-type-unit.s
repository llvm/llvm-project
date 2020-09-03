# REQUIRES: x86, zlib
## -gdwarf-5 -fdebug-types-section may produce multiple .debug_info sections.
## All except one are type units. Test we can locate the compile unit, add it to
## the index, and not erroneously duplicate it (which would happen if we
## consider every .debug_info a compile unit).

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --gdb-index -Ttext=0x1000 %t.o -o %t
# RUN: llvm-dwarfdump --gdb-index %t | FileCheck %s

## Test we don't uncompress a section while another thread is concurrently
## accessing it. This would be detected by tsan as a data race.
# RUN: llvm-objcopy --compress-debug-sections %t.o
# RUN: ld.lld --gdb-index -Ttext=0x1000 %t.o -o %t1
# RUN: llvm-dwarfdump --gdb-index %t1 | FileCheck %s

## In this test, there are actually two compile unit .debug_info (very uncommon;
## -r --unique). Currently we only handle the last compile unit.
# CHECK:      CU list offset = 0x18, has 1 entries:
# CHECK-NEXT:   0: Offset = 0x32, Length = 0x19

# CHECK:      Address area offset = 0x28, has 1 entries:
# CHECK-NEXT:   Low/High address = [0x1001, 0x1002) (Size: 0x1), CU id = 0

.Lfunc_begin0:
  ret
.Lfunc_end0:
.Lfunc_begin1:
  ret
.Lfunc_end1:

.section  .debug_abbrev,"",@progbits
  .byte  1                         # Abbreviation Code
  .byte  65                        # DW_TAG_type_unit
  .byte  0                         # DW_CHILDREN_no
  .byte  0                         # EOM(1)
  .byte  0                         # EOM(2)

  .byte  2                         # Abbreviation Code
  .byte  17                        # DW_TAG_compile_unit
  .byte  0                         # DW_CHILDREN_no
  .byte  17                        # DW_AT_low_pc
  .byte  1                         # DW_FORM_addr
  .byte  18                        # DW_AT_high_pc
  .byte  6                         # DW_FORM_data4
  .byte  0                         # EOM(1)
  .byte  0                         # EOM(2)

  .byte  0                         # EOM(3)

.macro TYPE_UNIT id signature
.section  .debug_info,"G",@progbits,\signature
  .long  .Ldebug_info_end\id-.Ldebug_info_start\id # Length of Unit
.Ldebug_info_start\id:
  .short 5                         # DWARF version number
  .byte  2                         # DWARF Unit Type
  .byte  8                         # Address Size
  .long  .debug_abbrev             # Offset Into Abbrev. Section
  .quad  \signature                # Type Signature
  .long  .Ldebug_info_end\id       # Type DIE Offset
  .byte  1                         # Abbrev [1] DW_TAG_type_unit
.Ldebug_info_end\id:
.endm

## We place compile units between two type units (rare). A naive approach will
## take either the first or the last .debug_info
TYPE_UNIT 0, 123

.section  .debug_info,"",@progbits,unique,0
.Lcu_begin0:
  .long .Lcu_end0-.Lcu_begin0-4    # Length of Unit
  .short 5                         # DWARF version number
  .byte  1                         # DWARF Unit Type
  .byte  8                         # Address Size
  .long  .debug_abbrev             # Offset Into Abbrev. Section
  .byte  2                         # Abbrev [2] DW_TAG_compile_unit
  .quad  .Lfunc_begin0             # DW_AT_low_pc
  .long  .Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
.Lcu_end0:

.section  .debug_info,"",@progbits,unique,1
.Lcu_begin1:
  .long .Lcu_end1-.Lcu_begin1-4    # Length of Unit
  .short 5                         # DWARF version number
  .byte  1                         # DWARF Unit Type
  .byte  8                         # Address Size
  .long  .debug_abbrev             # Offset Into Abbrev. Section
  .byte  2                         # Abbrev [2] DW_TAG_compile_unit
  .quad  .Lfunc_begin1             # DW_AT_low_pc
  .long  .Lfunc_end1-.Lfunc_begin1 # DW_AT_high_pc
.Lcu_end1:

TYPE_UNIT 1, 456
