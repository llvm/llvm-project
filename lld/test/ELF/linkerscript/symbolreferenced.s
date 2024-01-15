# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

# Provide new symbol. The value should be 1, like set in PROVIDE()
# RUN: echo "SECTIONS { PROVIDE(newsym = 1);}" > a1.t
# RUN: ld.lld -o a1 -T a1.t a.o
# RUN: llvm-objdump -t a1 | FileCheck --check-prefix=PROVIDE1 %s
# PROVIDE1: 0000000000000001 g       *ABS*  0000000000000000 newsym

# Provide new symbol (hidden). The value should be 1
# RUN: echo "SECTIONS { PROVIDE_HIDDEN(newsym = 1);}" > a2.t
# RUN: ld.lld -o a2 -T a2.t a.o
# RUN: llvm-objdump -t a2 | FileCheck --check-prefix=HIDDEN1 %s
# HIDDEN1: 0000000000000001 l       *ABS*  0000000000000000 .hidden newsym

# RUN: echo 'SECTIONS { PROVIDE_HIDDEN("newsym" = 1);}' > a2.t
# RUN: ld.lld -o a2 -T a2.t a.o
# RUN: llvm-objdump -t a2 | FileCheck --check-prefix=HIDDEN1 %s

# RUN: ld.lld -o chain -T chain.t a.o
# RUN: llvm-nm chain | FileCheck %s

# CHECK:      0000000000001000 a f1
# CHECK-NEXT: 0000000000001000 A f2
# CHECK-NEXT: 0000000000001000 a g1
# CHECK-NEXT: 0000000000001000 A g2
# CHECK-NEXT: 0000000000001000 A newsym

# RUN: not ld.lld -T chain2.t a.o 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:
# ERR-COUNT-3: error: chain2.t:1: symbol not found: undef

#--- a.s
.global _start
_start:
 nop

.globl patatino
patatino:
  movl newsym, %eax

#--- chain.t
PROVIDE(f2 = 0x1000);
PROVIDE_HIDDEN(f1 = f2);
PROVIDE(newsym = f1);

PROVIDE(g2 = 0x1000);
PROVIDE_HIDDEN(g1 = g2);
PROVIDE(unused = g1);

#--- chain2.t
PROVIDE(f2 = undef);
PROVIDE_HIDDEN(f1 = f2);
PROVIDE(newsym = f1);
