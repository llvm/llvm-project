# Check that llvm-bolt keeps DW_AT_high_pc consistent with its form when it
# relocates a function described by a DWARF 2 CU.
#
# DW_AT_high_pc holding an offset from DW_AT_low_pc was introduced in DWARF 4.
# Before that the attribute is always class address, so a DWARF 2 producer emits
# it with DW_FORM_addr. BOLT used to write the size of the function regardless
# of the form it was reusing, leaving high_pc below low_pc. Consumers derive the
# size as "high_pc - low_pc", so the subtraction underflowed and the function
# appeared to span almost the entire address space.
#
# go_dwarf.test covers the same form, but there the function keeps its input
# address. Here it is relocated, so the address arithmetic is covered too.

# REQUIRES: system-linux

# RUN: llvm-mc --filetype=obj --triple aarch64-unknown-linux %s -o %t.o
# RUN: ld.lld --no-pie %t.o -o %t.exe -q
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections

# The CU must be DWARF 2 with an address-form high_pc, otherwise this test
# would not exercise the address class path.
# RUN: llvm-dwarfdump --show-form --debug-info %t.exe \
# RUN:   | FileCheck --check-prefix=CHECK-PRE %s

# CHECK-PRE: version = 0x0002
# CHECK-PRE: DW_TAG_subprogram
# CHECK-PRE: DW_AT_low_pc [DW_FORM_addr]
# CHECK-PRE-NEXT: DW_AT_high_pc [DW_FORM_addr]

# The form is preserved, so the value has to remain an end address.
# RUN: llvm-dwarfdump --show-form --debug-info %t.bolt \
# RUN:   | FileCheck --check-prefix=CHECK-POST %s

# CHECK-POST: DW_TAG_subprogram
# CHECK-POST: DW_AT_low_pc [DW_FORM_addr]
# CHECK-POST-NEXT: DW_AT_high_pc [DW_FORM_addr]

# An inverted range is reported as "Invalid address range".
# RUN: llvm-dwarfdump --verify %t.bolt \
# RUN:   | FileCheck --check-prefix=CHECK-VERIFY %s

# CHECK-VERIFY: No errors.

# Every high_pc must stay above its low_pc, and the code is expected to have
# moved, so the subprogram addresses must differ from the input.
# RUN: llvm-dwarfdump --debug-info %t.bolt \
# RUN:   | awk '/DW_AT_low_pc/{gsub(/[()]/,"",$2); L=$2} \
# RUN:          /DW_AT_high_pc/{gsub(/[()]/,"",$2); print L, $2}' > %t.pcs
# RUN: bash -c "test -s %t.pcs; while read L H; do \
# RUN:   test \$((H)) -gt \$((L)) || exit 1; done < %t.pcs"
# RUN: llvm-dwarfdump --debug-info %t.exe \
# RUN:   | awk '/DW_AT_low_pc/{print $2}' > %t.pcs-orig
# RUN: llvm-dwarfdump --debug-info %t.bolt \
# RUN:   | awk '/DW_AT_low_pc/{print $2}' > %t.pcs-bolt
# RUN: not diff %t.pcs-orig %t.pcs-bolt

  .text
  .file 1 "dwarf2-highpc-addr-form.s"
  .globl _start
  .type _start, %function
_start:
.L_start_begin:
  .cfi_startproc
  .loc 1 1 0
  bl foo
  .loc 1 2 0
  bl foo
  .loc 1 3 0
  ret
  .cfi_endproc
.L_start_end:
  .size _start, .-_start

  .globl foo
  .type foo, %function
foo:
.Lfoo_begin:
  .cfi_startproc
  .loc 1 5 0
  add x0, x0, #1
  .loc 1 6 0
  ret
  .cfi_endproc
.Lfoo_end:
  .size foo, .-foo

  .section .debug_abbrev,"",@progbits
  .uleb128 1                    // Abbrev code
  .uleb128 0x11                 // DW_TAG_compile_unit
  .byte 1                       // DW_CHILDREN_yes
  .uleb128 0x03                 // DW_AT_name
  .uleb128 0x08                 // DW_FORM_string
  .uleb128 0x10                 // DW_AT_stmt_list
  .uleb128 0x06                 // DW_FORM_data4
  .uleb128 0x11                 // DW_AT_low_pc
  .uleb128 0x01                 // DW_FORM_addr
  .uleb128 0x12                 // DW_AT_high_pc
  .uleb128 0x01                 // DW_FORM_addr
  .uleb128 0
  .uleb128 0

  .uleb128 2                    // Abbrev code
  .uleb128 0x2e                 // DW_TAG_subprogram
  .byte 0                       // DW_CHILDREN_no
  .uleb128 0x03                 // DW_AT_name
  .uleb128 0x08                 // DW_FORM_string
  .uleb128 0x11                 // DW_AT_low_pc
  .uleb128 0x01                 // DW_FORM_addr
  .uleb128 0x12                 // DW_AT_high_pc
  .uleb128 0x01                 // DW_FORM_addr
  .uleb128 0
  .uleb128 0

  .byte 0                       // End of abbrevs

  .section .debug_info,"",@progbits
  .long .Lcu_end - .Lcu_start   // Length
.Lcu_start:
  .short 2                      // DWARF version
  .long 0                       // Abbrev offset
  .byte 8                       // Address size

  .uleb128 1                    // DW_TAG_compile_unit
  .asciz "dwarf2-highpc-addr-form.s"
  .long 0                       // DW_AT_stmt_list
  .quad .L_start_begin
  .quad .Lfoo_end

  .uleb128 2                    // DW_TAG_subprogram
  .asciz "_start"
  .quad .L_start_begin
  .quad .L_start_end

  .uleb128 2                    // DW_TAG_subprogram
  .asciz "foo"
  .quad .Lfoo_begin
  .quad .Lfoo_end

  .byte 0                       // End of children
.Lcu_end:
