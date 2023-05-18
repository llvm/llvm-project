# REQUIRES: mips
# Check MIPS R_MIPS_TLS_DTPREL_HI16/LO16 and R_MIPS_TLS_TPREL_HI16/LO16
# relocations handling.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe
# RUN: llvm-objdump --no-print-imm-hex -d -t --no-show-raw-insn %t.exe | FileCheck --check-prefix=DIS %s
# RUN: llvm-readobj -r -A %t.exe | FileCheck %s

# RUN: not ld.lld %t.o -shared -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

# ERR: error: relocation R_MIPS_TLS_TPREL_HI16 against loc0 cannot be used with -shared
# ERR: error: relocation R_MIPS_TLS_TPREL_LO16 against loc0 cannot be used with -shared

# DIS: 00000000 l      .tdata          00000000 loc0

# DIS:      <__start>:
# DIS-NEXT:    addiu   $2, $3, 0
#                              ^-- %hi(loc0 - .tdata - 0x8000)
# DIS-NEXT:    addiu   $2, $3, -32768
#                              ^-- %lo(loc0 - .tdata - 0x8000)
# DIS-NEXT:    addiu   $2, $3, 0
#                              ^-- %hi(loc0 - .tdata - 0x7000)
# DIS-NEXT:    addiu   $2, $3, -28672
#                              ^-- %lo(loc0 - .tdata - 0x7000)

# CHECK:      Relocations [
# CHECK-NEXT: ]
# CHECK-NOT:  Primary GOT

# SO:      Relocations [
# SO-NEXT: ]
# SO:      Primary GOT {
# SO:        Local entries [
# SO-NEXT:   ]
# SO-NEXT:   Global entries [
# SO-NEXT:   ]
# SO-NEXT:   Number of TLS and multi-GOT entries: 0
# SO-NEXT: }

  .text
  .globl  __start
  .type __start,@function
__start:
  addiu $2, $3, %dtprel_hi(loc0)  # R_MIPS_TLS_DTPREL_HI16
  addiu $2, $3, %dtprel_lo(loc0)  # R_MIPS_TLS_DTPREL_LO16
  addiu $2, $3, %tprel_hi(loc0)   # R_MIPS_TLS_TPREL_HI16
  addiu $2, $3, %tprel_lo(loc0)   # R_MIPS_TLS_TPREL_LO16

 .section .tdata,"awT",%progbits
loc0:
 .word 0
