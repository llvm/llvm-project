# Check MIPS TLS relocations handling.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         %p/Inputs/mips-tls.s -o %t.so.o
# RUN: ld.lld -shared %t.so.o -o %t.so
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o %t.so -o %t.exe
# RUN: llvm-objdump -d -s -t %t.exe | FileCheck -check-prefix=DIS %s
# RUN: llvm-readobj -r -mips-plt-got %t.exe | FileCheck %s

# REQUIRES: mips

# DIS:      __start:
# DIS-NEXT:    20000:   24 62 80 1c   addiu   $2, $3, -32740
# DIS-NEXT:    20004:   24 62 80 24   addiu   $2, $3, -32732
# DIS-NEXT:    20008:   8f 82 80 18   lw      $2, -32744($gp)
# DIS-NEXT:    2000c:   24 62 80 2c   addiu   $2, $3, -32724

# DIS:      Contents of section .got:
# DIS_NEXT:  30004 00000000 80000000 00020000 00000000
# DIS_NEXT:  30014 00000000 00000000 00000000 00000000

# DIS: 00030000 l       .tdata          00000000 .tdata
# DIS: 00030000 l       .tdata          00000000 loc
# DIS: 00000000 g       *UND*           00000000 foo

# CHECK:      Relocations [
# CHECK-NEXT:   Section (7) .rel.dyn {
# CHECK-NEXT:     0x30018 R_MIPS_TLS_DTPMOD32 - 0x0
# CHECK-NEXT:     0x30010 R_MIPS_TLS_DTPMOD32 foo 0x0
# CHECK-NEXT:     0x30014 R_MIPS_TLS_DTPREL32 foo 0x0
# CHECK-NEXT:     0x30020 R_MIPS_TLS_TPREL32 foo 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]
# CHECK-NEXT: Primary GOT {
# CHECK-NEXT:   Canonical gp value: 0x37FF4
# CHECK-NEXT:   Reserved entries [
# CHECK-NEXT:     Entry {
# CHECK-NEXT:       Address: 0x30004
# CHECK-NEXT:       Access: -32752
# CHECK-NEXT:       Initial: 0x0
# CHECK-NEXT:       Purpose: Lazy resolver
# CHECK-NEXT:     }
# CHECK-NEXT:     Entry {
# CHECK-NEXT:       Address: 0x30008
# CHECK-NEXT:       Access: -32748
# CHECK-NEXT:       Initial: 0x80000000
# CHECK-NEXT:       Purpose: Module pointer (GNU extension)
# CHECK-NEXT:     }
# CHECK-NEXT:   ]
# CHECK-NEXT:   Local entries [
# CHECK-NEXT:     Entry {
# CHECK-NEXT:       Address: 0x3000C
# CHECK-NEXT:       Access: -32744
# CHECK-NEXT:       Initial: 0x20000
# CHECK-NEXT:     }
# CHECK-NEXT:   ]
# CHECK-NEXT:   Global entries [
# CHECK-NEXT:   ]
# CHECK-NEXT:   Number of TLS and multi-GOT entries: 5
#               ^-- 0x30010 / -32740 - R_MIPS_TLS_GD  - R_MIPS_TLS_DTPMOD32 foo
#               ^-- 0x30018 / -32736                  - R_MIPS_TLS_DTPREL32 foo
#               ^-- 0x3001C / -32732 - R_MIPS_TLS_LDM - R_MIPS_TLS_DTPMOD32 loc
#               ^-- 0x30020 / -32728
#               ^-- 0x30024 / -32724 - R_MIPS_TLS_GOTTPREL - R_MIPS_TLS_TPREL32

  .text
  .global  __start
__start:
  addiu $2, $3, %tlsgd(foo)     # R_MIPS_TLS_GD
  addiu $2, $3, %tlsldm(loc)    # R_MIPS_TLS_LDM
  lw    $2, %got(__start)($gp)
  addiu $2, $3, %gottprel(foo)  # R_MIPS_TLS_GOTTPREL

 .section .tdata,"awT",%progbits
loc:
 .word 0
