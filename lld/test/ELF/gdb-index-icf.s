# REQUIRES: x86
## Test that the address range contributed by an ICF folded function is identical
## to that of the folded-in function. Not considering ICF may lead to an address
## range whose low address equals the start address of the output section.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --gdb-index --icf=all -Ttext=0x1000 %t.o -o %t
# RUN: llvm-dwarfdump --gdb-index %t | FileCheck %s

# CHECK:      Address area offset = 0x38, has 2 entries:
# CHECK-NEXT:   Low/High address = [0x1001, 0x1002) (Size: 0x1), CU id = 0
# CHECK-NEXT:   Low/High address = [0x1001, 0x1002) (Size: 0x1), CU id = 1

.text
nop

.section .text.0,"ax"
.Lfunc_begin0:
  ret
.Lfunc_end0:

.section .text.1,"ax"
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

.section  .debug_info,"",@progbits
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
