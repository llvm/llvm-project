# Check MIPS TLS 64-bit relocations handling.

# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux \
# RUN:         %p/Inputs/mips-dynamic.s -o %t.so.o
# RUN: ld.lld -shared %t.so.o -o %t.so
# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o %t.so -o %t.exe
# RUN: llvm-objdump -d -s -t %t.exe | FileCheck -check-prefix=DIS %s
# RUN: llvm-readobj -r -mips-plt-got %t.exe | FileCheck %s

# REQUIRES: mips

# DIS:      __start:
# DIS-NEXT:    20000:   24 62 80 28   addiu   $2, $3, -32728
# DIS-NEXT:    20004:   24 62 80 38   addiu   $2, $3, -32712
# DIS-NEXT:    20008:   8f 82 80 20   lw      $2, -32736($gp)
# DIS-NEXT:    2000c:   24 62 80 48   addiu   $2, $3, -32696

# DIS:      Contents of section .got:
# DIS_NEXT:  30008 00000000 00000000 80000000 00000000
# DIS_NEXT:  30018 00000000 00020000 00000000 00000000
# DIS_NEXT:  30028 00000000 00000004 00000000 00000000
# DIS_NEXT:  30038 00000000 00000000 00000000 00000004

# DIS: 0000000000030000 l       .tdata          00000000 .tdata
# DIS: 0000000000030000 l       .tdata          00000000 loc
# DIS: 0000000000000004 g       .tdata          00000000 foo

# CHECK:      Relocations [
# CHECK-NEXT:   Section (7) .rela.dyn {
# CHECK-NEXT:     0x30020 R_MIPS_TLS_DTPMOD64/R_MIPS_NONE/R_MIPS_NONE - 0x0
# CHECK-NEXT:     0x30028 R_MIPS_TLS_DTPREL64/R_MIPS_NONE/R_MIPS_NONE - 0x0
# CHECK-NEXT:     0x30030 R_MIPS_TLS_DTPMOD64/R_MIPS_NONE/R_MIPS_NONE - 0x0
# CHECK-NEXT:     0x30040 R_MIPS_TLS_TPREL64/R_MIPS_NONE/R_MIPS_NONE - 0x4
# CHECK-NEXT:   }
# CHECK-NEXT: ]
# CHECK-NEXT: Primary GOT {
# CHECK-NEXT:   Canonical gp value: 0x37FF8
# CHECK-NEXT:   Reserved entries [
# CHECK-NEXT:     Entry {
# CHECK-NEXT:       Address: 0x30008
# CHECK-NEXT:       Access: -32752
# CHECK-NEXT:       Initial: 0x0
# CHECK-NEXT:       Purpose: Lazy resolver
# CHECK-NEXT:     }
# CHECK-NEXT:     Entry {
# CHECK-NEXT:       Address: 0x30010
# CHECK-NEXT:       Access: -32744
# CHECK-NEXT:       Initial: 0x80000000
# CHECK-NEXT:       Purpose: Module pointer (GNU extension)
# CHECK-NEXT:     }
# CHECK-NEXT:   ]
# CHECK-NEXT:   Local entries [
# CHECK-NEXT:   ]
# CHECK-NEXT:   Global entries [
# CHECK-NEXT:     Entry {
# CHECK-NEXT:       Address: 0x30018
# CHECK-NEXT:       Access: -32736
# CHECK-NEXT:       Initial: 0x0
# CHECK-NEXT:       Value: 0x0
# CHECK-NEXT:       Type: Function
# CHECK-NEXT:       Section: Undefined
# CHECK-NEXT:       Name: foo0
# CHECK-NEXT:     }
# CHECK-NEXT:   ]
# CHECK-NEXT:   Number of TLS and multi-GOT entries: 5
#               ^-- 0x30020 / -32728 - R_MIPS_TLS_GD  - R_MIPS_TLS_DTPMOD32 foo
#               ^-- 0x30028 / -32720                  - R_MIPS_TLS_DTPREL32 foo
#               ^-- 0x30030 / -32712 - R_MIPS_TLS_LDM - R_MIPS_TLS_DTPMOD32 loc
#               ^-- 0x30038 / -32704
#               ^-- 0x30040 / -32696 - R_MIPS_TLS_GOTTPREL - R_MIPS_TLS_TPREL32

  .text
  .global  __start
__start:
  addiu $2, $3, %tlsgd(foo)     # R_MIPS_TLS_GD
  addiu $2, $3, %tlsldm(loc)    # R_MIPS_TLS_LDM
  lw    $2, %got(foo0)($gp)
  addiu $2, $3, %gottprel(foo)  # R_MIPS_TLS_GOTTPREL

 .section .tdata,"awT",%progbits
 .global foo
loc:
 .word 0
foo:
 .word 0
