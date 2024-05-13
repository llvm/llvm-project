// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: not ld.lld -shared --vs-diagnostics %t.o -o /dev/null 2>&1 | FileCheck %s

// CHECK: dyn.s(15): error: relocation R_X86_64_64 cannot be used against local symbol; recompile with -fPIC
// CHECK-NEXT: >>> defined in {{.*}}.o
// CHECK-NEXT: >>> referenced by dyn.s:15
// CHECK-NEXT: >>>{{.*}}.o:(.text+0x{{.+}})

// CHECK: /tmp{{/|\\}}dyn.s(20): error: relocation R_X86_64_64 cannot be used against local symbol; recompile with -fPIC
// CHECK-NEXT: >>> defined in {{.*}}.o
// CHECK-NEXT: >>> referenced by dyn.s:20 (/tmp{{/|\\}}dyn.s:20)
// CHECK-NEXT: >>>{{.*}}.o:(.text+0x{{.+}})

.file 1 "dyn.s"
.loc 1 15

foo:
.quad foo

.file 2 "/tmp" "dyn.s"
.loc 2 20

bar:
.quad bar

.section .debug_abbrev,"",@progbits
  .byte  1                      # Abbreviation Code
  .byte 17                      # DW_TAG_compile_unit
  .byte  0                      # DW_CHILDREN_no
  .byte 16                      # DW_AT_stmt_list
  .byte 23                      # DW_FORM_sec_offset
  .byte  0                      # EOM(1)
  .byte  0                      # EOM(2)
  .byte  0                      # EOM(3)

.section .debug_info,"",@progbits
  .long .Lend0 - .Lbegin0       # Length of Unit
.Lbegin0:
  .short 4                      # DWARF version number
  .long  .debug_abbrev          # Offset Into Abbrev. Section
  .byte  8                      # Address Size (in bytes)
  .byte  1                      # Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
  .long  .debug_line            # DW_AT_stmt_list
.Lend0:
  .section .debug_line,"",@progbits
