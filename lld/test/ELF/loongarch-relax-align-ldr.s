# REQUIRES: loongarch

# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s -o %t.64.o
# RUN: ld.lld -r %t.64.o %t.64.o -o %t.64.r
# RUN: llvm-objdump -dr --no-show-raw-insn %t.64.r | FileCheck %s

# CHECK:      <.text>:
# CHECK-NEXT:   break 1
# CHECK-NEXT:   nop
# CHECK-NEXT:   {{0*}}04:  R_LARCH_ALIGN        .text+0x804
# CHECK-NEXT:   nop
# CHECK-NEXT:   nop
# CHECK-NEXT:   break 2
# CHECK-NEXT:   break 0
# CHECK-NEXT:   break 0
# CHECK-NEXT:   break 0
# CHECK-NEXT:   break 1
# CHECK-NEXT:   nop
# CHECK-NEXT:   {{0*}}24:  R_LARCH_ALIGN        .text+0x804
# CHECK-NEXT:   nop
# CHECK-NEXT:   nop
# CHECK-NEXT:   break 2

.text
break 1
.p2align 4, , 8
break 2
