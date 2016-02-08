# Check creating of R_MIPS_COPY dynamic relocation.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         %S/Inputs/mips-dynamic.s -o %t.so.o
# RUN: ld.lld %t.so.o -shared -o %t.so
# RUN: ld.lld %t.o %t.so -o %t.exe
# RUN: llvm-readobj -r -mips-plt-got %t.exe | FileCheck %s

# REQUIRES: mips

# CHECK:      Relocations [
# CHECK-NEXT:   Section (7) .rel.dyn {
# CHECK-NEXT:     0x{{[0-9A-F]+}} R_MIPS_COPY data0 0x0
# CHECK-NEXT:     0x{{[0-9A-F]+}} R_MIPS_COPY data1 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]

# CHECK:      Primary GOT {
# CHECK:        Local entries [
# CHECK-NEXT:   ]
# CHECK-NEXT:   Global entries [
# CHECK-NEXT:   ]
# CHECK-NEXT:   Number of TLS and multi-GOT entries: 0
# CHECK-NEXT: }

  .text
  .globl __start
__start:
  lui    $t0,%hi(data0)    # R_MIPS_HI16 requires COPY rel for DSO defined data.
  addi   $t0,$t0,%lo(data0)
  lui    $t0,%hi(gd)       # Does not require COPY rel for locally defined data.
  addi   $t0,$t0,%lo(gd)
  lui    $t0,%hi(ld)       # Does not require COPY rel for local data.
  addi   $t0,$t0,%lo(ld)

  .data
  .globl gd
gd:
  .word 0
ld:
  .word data1+8-.          # R_MIPS_PC32 requires COPY rel for DSO defined data.
