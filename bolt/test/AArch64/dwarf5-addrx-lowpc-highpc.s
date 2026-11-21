# Check that llvm-bolt handles DW_AT_low_pc and DW_AT_high_pc encoded with the
# fixed-size indexed address forms.
#
# DW_FORM_addrx is already covered by the DWARF 5 tests; this test uses
# DW_FORM_addrx2, and gives "foo" an indexed DW_AT_high_pc so the read side has
# to resolve an index there too.

# REQUIRES: system-linux

# RUN: llvm-mc --filetype=obj --triple aarch64-unknown-linux %s -o %t.o
# RUN: ld.lld --no-pie %t.o -o %t.exe -q

# Guard the fixture: if these forms ever stop being emitted the test would
# quietly stop covering the indexed path.
# RUN: llvm-dwarfdump --show-form --debug-info %t.exe \
# RUN:   | FileCheck --check-prefix=CHECK-PRE %s

# CHECK-PRE: version = 0x0005
# CHECK-PRE: DW_AT_name [DW_FORM_string] ("_start")
# CHECK-PRE-NEXT: DW_AT_low_pc [DW_FORM_addrx2]
# CHECK-PRE-NEXT: DW_AT_high_pc [DW_FORM_data4]
# CHECK-PRE: DW_AT_name [DW_FORM_string] ("foo")
# CHECK-PRE-NEXT: DW_AT_low_pc [DW_FORM_addrx2]
# CHECK-PRE-NEXT: DW_AT_high_pc [DW_FORM_addrx2]

# The attributes must survive cloning rather than being dropped.
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections 2>&1 \
# RUN:   | FileCheck --check-prefix=CHECK-BOLT %s

# CHECK-BOLT-NOT: Unsupported attribute form

# The values stay .debug_addr indices, but the fixed-width form is normalized
# to DW_FORM_addrx: the index is reassigned against a rebuilt .debug_addr that
# may hold more entries than the input's, and a fixed-width form would silently
# truncate one that no longer fits.
# RUN: llvm-dwarfdump --show-form --debug-info %t.bolt \
# RUN:   | FileCheck --check-prefix=CHECK-POST %s \
# RUN:       --implicit-check-not=DW_FORM_addrx2

# CHECK-POST: DW_AT_name [DW_FORM_string] ("_start")
# CHECK-POST-NEXT: DW_AT_low_pc [DW_FORM_addrx]
# CHECK-POST-NEXT: DW_AT_high_pc [DW_FORM_data4]
# CHECK-POST: DW_AT_name [DW_FORM_string] ("foo")
# CHECK-POST-NEXT: DW_AT_low_pc [DW_FORM_addrx]
# CHECK-POST-NEXT: DW_AT_high_pc [DW_FORM_addrx]

# RUN: llvm-dwarfdump --verify %t.bolt \
# RUN:   | FileCheck --check-prefix=CHECK-VERIFY %s

# CHECK-VERIFY: No errors.

# Each index has to resolve to where the function actually ended up, and foo's
# high_pc to the address one past its end.
# RUN: llvm-dwarfdump --debug-info %t.bolt > %t.dump
# RUN: llvm-nm %t.bolt | awk '$3=="_start"{print $1}' > %t.start-addr
# RUN: llvm-nm %t.bolt | awk '$3=="foo"{print $1}' > %t.foo-addr
# RUN: bash -c "read A < %t.start-addr; \
# RUN:   grep -q \"DW_AT_low_pc.*(0x\$A)\" %t.dump"
# RUN: bash -c "read A < %t.foo-addr; \
# RUN:   grep -q \"DW_AT_low_pc.*(0x\$A)\" %t.dump"
# RUN: bash -c "read A < %t.foo-addr; \
# RUN:   E=\$(printf '%%016x' \$((0x\$A + 8))); \
# RUN:   grep -q \"DW_AT_high_pc.*(0x\$E)\" %t.dump"

  .text
  .file 1 "dwarf5-addrx-lowpc-highpc.s"
  .globl _start
  .type _start, %function
_start:
.L_start_begin:
  .cfi_startproc
  .loc 1 1 0
  bl foo
  .loc 1 2 0
  ret
  .cfi_endproc
.L_start_end:
  .size _start, .-_start

  .globl foo
  .type foo, %function
foo:
.Lfoo_begin:
  .cfi_startproc
  .loc 1 4 0
  add x0, x0, #1
  .loc 1 5 0
  ret
  .cfi_endproc
.Lfoo_end:
  .size foo, .-foo

  .section .debug_addr,"",@progbits
.Ldebug_addr_start:
  .long .Ldebug_addr_end - .Ldebug_addr_version   // unit_length
.Ldebug_addr_version:
  .short 5                                        // version
  .byte 8                                         // address_size
  .byte 0                                         // segment_selector_size
.Laddr_base:
  .quad .L_start_begin                            // index 0
  .quad .Lfoo_begin                               // index 1
  .quad .Lfoo_end                                 // index 2
.Ldebug_addr_end:

  .section .debug_abbrev,"",@progbits
  .uleb128 1              // abbrev code: compile unit
  .uleb128 0x11           // DW_TAG_compile_unit
  .byte 1                 // DW_CHILDREN_yes
  .uleb128 0x03           // DW_AT_name
  .uleb128 0x08           // DW_FORM_string
  .uleb128 0x10           // DW_AT_stmt_list
  .uleb128 0x17           // DW_FORM_sec_offset
  .uleb128 0x73           // DW_AT_addr_base
  .uleb128 0x17           // DW_FORM_sec_offset
  .uleb128 0x11           // DW_AT_low_pc
  .uleb128 0x1b           // DW_FORM_addrx
  .uleb128 0x12           // DW_AT_high_pc
  .uleb128 0x06           // DW_FORM_data4
  .uleb128 0
  .uleb128 0

  .uleb128 2              // abbrev code: low_pc addrx2, high_pc data4
  .uleb128 0x2e           // DW_TAG_subprogram
  .byte 0                 // DW_CHILDREN_no
  .uleb128 0x03           // DW_AT_name
  .uleb128 0x08           // DW_FORM_string
  .uleb128 0x11           // DW_AT_low_pc
  .uleb128 0x2a           // DW_FORM_addrx2
  .uleb128 0x12           // DW_AT_high_pc
  .uleb128 0x06           // DW_FORM_data4
  .uleb128 0
  .uleb128 0

  .uleb128 3              // abbrev code: low_pc addrx2, high_pc addrx2
  .uleb128 0x2e           // DW_TAG_subprogram
  .byte 0                 // DW_CHILDREN_no
  .uleb128 0x03           // DW_AT_name
  .uleb128 0x08           // DW_FORM_string
  .uleb128 0x11           // DW_AT_low_pc
  .uleb128 0x2a           // DW_FORM_addrx2
  .uleb128 0x12           // DW_AT_high_pc
  .uleb128 0x2a           // DW_FORM_addrx2
  .uleb128 0
  .uleb128 0

  .byte 0                 // end of abbrevs

  .section .debug_info,"",@progbits
  .long .Lcu_end - .Lcu_start     // unit_length
.Lcu_start:
  .short 5                        // DWARF version
  .byte 1                         // DW_UT_compile
  .byte 8                         // address size
  .long 0                         // abbrev offset

  .uleb128 1                      // DW_TAG_compile_unit
  .asciz "dwarf5-addrx-lowpc-highpc.s"
  .long 0                                    // DW_AT_stmt_list
  .long .Laddr_base - .Ldebug_addr_start     // DW_AT_addr_base
  .uleb128 0                                 // DW_AT_low_pc  -> addr[0]
  .long .Lfoo_end - .L_start_begin           // DW_AT_high_pc

  .uleb128 2                      // DW_TAG_subprogram
  .asciz "_start"
  .short 0                                   // DW_AT_low_pc  -> addr[0]
  .long .L_start_end - .L_start_begin        // DW_AT_high_pc

  .uleb128 3                      // DW_TAG_subprogram
  .asciz "foo"
  .short 1                                   // DW_AT_low_pc  -> addr[1]
  .short 2                                   // DW_AT_high_pc -> addr[2]

  .byte 0                         // end of children
.Lcu_end:
