# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck %s
# RUN: ld.lld %t.o -o %t

# CHECK:      .rela.debug_info {
# CHECK-NEXT:   0x6 R_AARCH64_ABS32 .debug_abbrev 0x0
# CHECK-NEXT:   0xD R_AARCH64_TLS_DTPREL64 var 0x0
# CHECK-NEXT: }

.section .tdata,"awT",@progbits
.globl var
var:
  .word 0

.section .debug_abbrev,"",@progbits
.byte 1                  // Abbreviation Code
.byte 17                 // DW_TAG_compile_unit
.byte 1                  // DW_CHILDREN_yes
.byte 0                  // EOM(1)
.byte 0                  // EOM(2)

.byte 2                  // Abbreviation Code
.byte 52                 // DW_TAG_variable
.byte 0                  // DW_CHILDREN_no
.byte 2;                 // DW_AT_location
.byte 24                 // DW_FORM_exprloc
.byte 0                  // EOM(1)
.byte 0                  // EOM(2)

.section        .debug_info,"",@progbits
.Lcu_begin0:
  .word .Lcu_end - .Lcu_body // Length of Unit
.Lcu_body:
  .hword 4               // DWARF version number
  .word   .debug_abbrev  // Offset Into Abbrev. Section
  .byte   8              // Address Size (in bytes)
  .byte   1              // Abbrev [1] DW_TAG_compile_unit
  .byte   2              // Abbrev [2] DW_TAG_variable
  .xword  %dtprel(var)
.Lcu_end:
