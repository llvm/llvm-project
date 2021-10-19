# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=ppc64le %s -o %t.o
# RUN: ld.lld %t.o --defsym=a=0x0123456789abcdef --defsym=b=0x76543210 -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# RUN: llvm-objdump -s --no-show-raw-insn %t | FileCheck --check-prefixes=HEX,HEXLE %s

# HEX-LABEL:  section .R_PPC64_ADDR32:
# HEXLE-NEXT:   10325476
# HEXBE-NEXT:   76543210
.section .R_PPC64_ADDR32,"a",@progbits
  .long b

# CHECK-LABEL: <.R_PPC64_ADDR16_LO>:
# CHECK-NEXT:    addi 4, 4, 12816
.section .R_PPC64_ADDR16_LO,"ax",@progbits
  addi 4, 4, b@l

# CHECK-LABEL: <.R_PPC64_ADDR16_HI>:
# CHECK-NEXT:    lis 4, 30292
.section .R_PPC64_ADDR16_HI,"ax",@progbits
  lis 4, b@h

# CHECK-LABEL: <.R_PPC64_ADDR16_HA>:
# CHECK-NEXT:    lis 4, 30292
.section .R_PPC64_ADDR16_HA,"ax",@progbits
  lis 4, b@ha

# CHECK-LABEL: <.R_PPC64_ADDR16_HIGH>:
# CHECK-NEXT:    lis 4, -30293
.section .R_PPC64_ADDR16_HIGH,"ax",@progbits
  lis 4, a@high

# CHECK-LABEL: <.R_PPC64_ADDR16_HIGHER>:
# CHECK-NEXT:    li 3, 17767
.section .R_PPC64_ADDR16_HIGHER,"ax",@progbits
  li 3, a@higher

# CHECK-LABEL: <.R_PPC64_ADDR16_HIGHERA>:
# CHECK-NEXT:    li 3, 17767
.section .R_PPC64_ADDR16_HIGHERA,"ax",@progbits
  li 3, a@highera

# CHECK-LABEL: <.R_PPC64_ADDR16_HIGHEST>:
# CHECK-NEXT:    li 3, 291
.section .R_PPC64_ADDR16_HIGHEST,"ax",@progbits
  li 3, a@highest

# CHECK-LABEL: <.R_PPC64_ADDR16_HIGHESTA>:
# CHECK-NEXT:    li 3, 291
.section .R_PPC64_ADDR16_HIGHESTA,"ax",@progbits
  li 3, a@highesta
