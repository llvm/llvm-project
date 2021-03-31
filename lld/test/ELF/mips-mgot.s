# REQUIRES: mips
# Check MIPS multi-GOT layout.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t0.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         %p/Inputs/mips-mgot-1.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         %p/Inputs/mips-mgot-2.s -o %t2.o
# RUN: echo "SECTIONS { \
# RUN:         . = 0x10000; .text : { *(.text) } \
# RUN:         . = 0x70000; .got  : { *(.got)  } \
# RUN:       }" > %t.script
# RUN: ld.lld -shared -mips-got-size 52 --script %t.script %t0.o %t1.o %t2.o -o %t.so
# RUN: llvm-objdump -s --section=.got -t %t.so | FileCheck %s
# RUN: llvm-readobj -r --dyn-syms -A %t.so | FileCheck -check-prefix=GOT %s

# CHECK: SYMBOL TABLE:
# CHECK:           00000000 l      .tdata          00000000 loc0
# CHECK: [[FOO0:[0-9a-f]+]] g      .text           00000000 foo0
# CHECK:           00000000 g      .tdata          00000000 tls0
# CHECK:           00000004 g      .tdata          00000000 tls1
# CHECK: [[FOO2:[0-9a-f]+]] g      .text           00000000 foo2

# CHECK:      Contents of section .got:
# CHECK-NEXT:  70000 00000000 80000000 [[FOO0]] [[FOO2]]
# CHECK-NEXT:  70010 00000000 00000004 00010000 00020000
# CHECK-NEXT:  70020 00030000 00040000 00050000 00060000
# CHECK-NEXT:  70030 00000000 00000000 00000000 00000000
# CHECK-NEXT:  70040 00000000 00000000 00000000

# GOT:      Relocations [
# GOT-NEXT:   Section (7) .rel.dyn {
# GOT-NEXT:     0x70018 R_MIPS_REL32 -
# GOT-NEXT:     0x7001C R_MIPS_REL32 -
# GOT-NEXT:     0x70020 R_MIPS_REL32 -
# GOT-NEXT:     0x70024 R_MIPS_REL32 -
# GOT-NEXT:     0x70028 R_MIPS_REL32 -
# GOT-NEXT:     0x7002C R_MIPS_REL32 -
# GOT-NEXT:     0x70030 R_MIPS_REL32 foo0
# GOT-NEXT:     0x70034 R_MIPS_REL32 foo2
# GOT-NEXT:     0x70044 R_MIPS_TLS_DTPMOD32 -
# GOT-NEXT:     0x70010 R_MIPS_TLS_TPREL32 tls0
# GOT-NEXT:     0x70038 R_MIPS_TLS_TPREL32 tls0
# GOT-NEXT:     0x7003C R_MIPS_TLS_DTPMOD32 tls0
# GOT-NEXT:     0x70040 R_MIPS_TLS_DTPREL32 tls0
# GOT-NEXT:     0x70014 R_MIPS_TLS_TPREL32 tls1
# GOT-NEXT:   }
# GOT-NEXT: ]

# GOT:      DynamicSymbols [
# GOT:        Symbol {
# GOT:          Name: foo0
# GOT-NEXT:     Value: 0x[[FOO0:[0-9A-F]+]]
# GOT:        }
# GOT-NEXT:   Symbol {
# GOT-NEXT:     Name: foo2
# GOT-NEXT:     Value: 0x[[FOO2:[0-9A-F]+]]
# GOT:        }
# GOT-NEXT: ]

# GOT:      Primary GOT {
# GOT-NEXT:   Canonical gp value: 0x77FF0
# GOT-NEXT:   Reserved entries [
# GOT-NEXT:     Entry {
# GOT-NEXT:       Address:
# GOT-NEXT:       Access: -32752
# GOT-NEXT:       Initial: 0x0
# GOT-NEXT:       Purpose: Lazy resolver
# GOT-NEXT:     }
# GOT-NEXT:     Entry {
# GOT-NEXT:       Address:
# GOT-NEXT:       Access: -32748
# GOT-NEXT:       Initial: 0x80000000
# GOT-NEXT:       Purpose: Module pointer (GNU extension)
# GOT-NEXT:     }
# GOT-NEXT:   ]
# GOT-NEXT:   Local entries [
# GOT-NEXT:   ]
# GOT-NEXT:   Global entries [
# GOT-NEXT:     Entry {
# GOT-NEXT:       Address:
# GOT-NEXT:       Access: -32744
# GOT-NEXT:       Initial: 0x[[FOO0]]
# GOT-NEXT:       Value: 0x[[FOO0]]
# GOT-NEXT:       Type: None
# GOT-NEXT:       Section: .text
# GOT-NEXT:       Name: foo0
# GOT-NEXT:     }
# GOT-NEXT:     Entry {
# GOT-NEXT:       Address:
# GOT-NEXT:       Access: -32740
# GOT-NEXT:       Initial: 0x[[FOO2]]
# GOT-NEXT:       Value: 0x[[FOO2]]
# GOT-NEXT:       Type: None
# GOT-NEXT:       Section: .text
# GOT-NEXT:       Name: foo2
# GOT-NEXT:     }
# GOT-NEXT:   ]
# GOT-NEXT:   Number of TLS and multi-GOT entries: 15
# GOT-NEXT: }

  .text
  .global foo0
foo0:
  lw     $2, %got(.data)($gp)     # page entry
  addi   $2, $2, %lo(.data)
  lw     $2, %call16(foo0)($gp)   # global entry
  addiu  $2, $2, %tlsgd(tls0)     # tls gd entry
  addiu  $2, $2, %gottprel(tls0)  # tls got entry
  addiu  $2, $2, %tlsldm(loc0)    # tls ld entry

  .data
  .space 0x20000

  .section .tdata,"awT",%progbits
  .global tls0
tls0:
loc0:
  .word 0
