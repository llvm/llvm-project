# REQUIRES: hexagon
# RUN: llvm-mc -mno-fixup -filetype=obj -triple=hexagon-unknown-elf %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %S/Inputs/hexagon-shared.s -o %t2.o
# RUN: ld.lld -shared %t2.o -soname so -o %t3.so
# RUN: ld.lld -shared %t.o %t3.so -o %t4.so
# RUN: ld.lld -Bsymbolic -shared %t.o %t3.so -o %t5.so
# RUN: llvm-objdump --no-print-imm-hex -d -j .plt %t4.so | FileCheck --check-prefix=PLT %s
# RUN: llvm-objdump --no-print-imm-hex -d -j .text %t4.so | FileCheck --check-prefix=TEXT %s
# RUN: llvm-objdump --no-print-imm-hex -D -j .got %t4.so | FileCheck --check-prefix=GOT %s
# RUN: llvm-readelf -r  %t4.so | FileCheck --check-prefix=RELO %s
# RUN: llvm-readelf -r  %t5.so | FileCheck --check-prefix=SYMBOLIC %s

.global _start, foo, hidden_symbol
.hidden hidden_symbol
_start:
# When -Bsymbolic is specified calls to locally resolvables should
# not generate a plt.
call ##foo
# Calls to hidden_symbols should not trigger a plt.
call ##hidden_symbol

# _HEX_32_PCREL
.word _DYNAMIC - .
call ##bar

# R_HEX_PLT_B22_PCREL
call bar@PLT
# R_HEX_B15_PCREL_X
if (p0) jump bar
# R_HEX_B9_PCREL_X
{ r0 = #0; jump bar }

# R_HEX_GOT_11_X and R_HEX_GOT_32_6_X
r2=add(pc,##_GLOBAL_OFFSET_TABLE_@PCREL)
r0 = memw (r2+##bar@GOT)
jumpr r0

# R_HEX_GOT_16_X
r0 = add(r1,##bar@GOT)

# R_HEX_GOT_16_X, duplex
{ r0 = add(r0,##bar@GOT)
  memw(r0) = r2 }

# R_HEX_GOT_16_X, pred add
if (p0) r0 = add(r0,##bar@GOT)
if (!p0) r0 = add(r0,##bar@GOT)
{ p0 = cmp.gtu(r0, r1)
  if (p0.new) r0 = add(r0,##bar@GOT) }
{ p0 = cmp.gtu(r0, r1)
  if (!p0.new) r0 = add(r0,##bar@GOT) }

# foo is local so no plt will be generated
foo:
  jumpr lr

hidden_symbol:
  jumpr lr

# R_HEX_32
.data
.global var
.type var,@object
.p2align 2
var:
   .word 10
   .size var, 4
.global pvar
.type pvar,@object
pvar:
   .word var
   .size pvar, 4


# PLT: { immext(#131264
# PLT-NEXT: r28 = add(pc,##131268) }
# PLT-NEXT: { r14 -= add(r28,#16)
# PLT-NEXT: r15 = memw(r28+#8)
# PLT-NEXT: r28 = memw(r28+#4) }
# PLT-NEXT: { r14 = asr(r14,#2)
# PLT-NEXT: jumpr r28 }
# PLT-NEXT: { trap0(#219) }
# PLT-NEXT: immext(#131200)
# PLT-NEXT: r14 = add(pc,##131252) }
# PLT-NEXT: r28 = memw(r14+#0) }
# PLT-NEXT: jumpr r28 }

# TEXT:  bc 00 01 00 000100bc
# TEXT: { 	call 0x10300 }
# TEXT: if (p0) jump:nt 0x10300
# TEXT: r0 = #0 ; jump 0x10300
# TEXT: r0 = add(r1,##-65548)
# TEXT: r0 = add(r0,##-65548); memw(r0+#0) = r2 }
# TEXT: if (p0) r0 = add(r0,##-65548)
# TEXT: if (!p0) r0 = add(r0,##-65548)
# TEXT: if (p0.new) r0 = add(r0,##-65548)
# TEXT: if (!p0.new) r0 = add(r0,##-65548)

# GOT: .got:
# GOT:  00 00 00 00 00000000 <unknown>

# RELO: R_HEX_GLOB_DAT
# RELO: R_HEX_32
# RELO: Relocation section '.rela.plt' at offset 0x22c contains 2 entries:
# RELO: R_HEX_JMP_SLOT {{.*}} foo
# RELO-NEXT: R_HEX_JMP_SLOT {{.*}} bar
# RELO-NOT: R_HEX_JMP_SLOT {{.*}} hidden

# Make sure that no PLT is generated for a local call.
# SYMBOLIC: Relocation section '.rela.plt' at offset 0x22c contains 1 entries:
# SYMBOLIC: R_HEX_JMP_SLOT {{.*}} bar
# SYMBOLIC-NOT: R_HEX_JMP_SLOT {{.*}} foo
