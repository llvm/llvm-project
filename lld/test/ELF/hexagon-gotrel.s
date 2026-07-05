# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %S/Inputs/hexagon-shared.s -o %t2.o
# RUN: ld.lld -shared %t2.o -soname=so -o %t2.so
# RUN: ld.lld -shared %t.o %t2.so -o %t3.so
# RUN: llvm-objdump --print-imm-hex -d -j .text %t3.so | FileCheck --check-prefix=TEXT %s

.global foo
foo:

.Lpc:

# R_HEX_GOTREL_LO16
  r0.l = #LO(.Lpc@GOTREL)
# R_HEX_GOTREL_HI16
  r0.h = #HI(.Lpc@GOTREL)
# R_HEX_GOTREL_11_X
  r0 = memw(r1+##.Lpc@GOTREL)
# R_HEX_GOTREL_32_6_X and R_HEX_GOTREL_16_X
  r0 = ##(.Lpc@GOTREL)

# TEXT: r0.l = #0xffa8 }
# TEXT: r0.h = #0xfffd }
# TEXT: immext(#0xfffdff80)
# TEXT: r0 = memw(r1+##-0x20058) }
# TEXT: immext(#0xfffdff80)
# TEXT: r0 = ##-0x20058 }
