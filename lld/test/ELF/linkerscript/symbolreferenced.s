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

# CHECK-NOT: another_unused
# CHECK:      0000000000007000 a f1
# CHECK-NEXT: 0000000000007000 A f2
# CHECK-NEXT: 0000000000007000 A f3
# CHECK-NEXT: 0000000000007000 A f4
# CHECK-NEXT: 0000000000006000 A f5
# CHECK-NEXT: 0000000000003000 A f6
# CHECK-NEXT: 0000000000001000 A f7
# CHECK-NOT: g1
# CHECK-NOT: g2
# CHECK-NEXT: 0000000000007500 A newsym
# CHECK: 0000000000002000 A u
# CHECK-NOT: unused
# CHECK-NEXT: 0000000000002000 A v
# CHECK-NEXT: 0000000000002000 A w


# RUN: ld.lld -o chain_with_cycle -T chain_with_cycle.t a.o
# RUN: llvm-nm chain_with_cycle | FileCheck %s --check-prefix=CHAIN_WITH_CYCLE

# CHAIN_WITH_CYCLE: 000 A f1
# CHAIN_WITH_CYCLE: 000 A f2
# CHAIN_WITH_CYCLE: 000 A f3
# CHAIN_WITH_CYCLE: 000 A f4
# CHAIN_WITH_CYCLE: 000 A newsym

# RUN: not ld.lld -T chain2.t a.o 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:
# ERR-COUNT-3: error: chain2.t:1: symbol not found: undef

## _start in a lazy object file references PROVIDE symbols. We extract _start
## earlier to avoid spurious "symbol not found" errors.
# RUN: llvm-mc -filetype=obj -triple=x86_64 undef.s -o undef.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 start.s -o start.o
# RUN: ld.lld -T chain2.t undef.o --start-lib start.o --end-lib -o lazy
# RUN: llvm-nm lazy | FileCheck %s --check-prefix=LAZY
# RUN: ld.lld -e 0 -T chain2.t --undefined-glob '_start*' undef.o --start-lib start.o --end-lib -o lazy
# RUN: llvm-nm lazy | FileCheck %s --check-prefix=LAZY

# LAZY:      T _start
# LAZY-NEXT: t f1
# LAZY-NEXT: T f2
# LAZY-NEXT: T newsym
# LAZY-NEXT: T unde

#--- a.s
.global _start
_start:
 nop

.globl patatino
patatino:
  movl newsym, %eax

#--- chain.t
PROVIDE(f7 = 0x1000);
PROVIDE(f5 = f6 + 0x3000);
PROVIDE(f6 = f7 + 0x2000);
PROVIDE(f4 = f5 + 0x1000);
PROVIDE(f3 = f4);
PROVIDE(f2 = f3);
PROVIDE_HIDDEN(f1 = f2);
PROVIDE(newsym = f1 + 0x500);

u = v;
PROVIDE(w = 0x2000);
PROVIDE(v = w);

PROVIDE(g2 = 0x1000);
PROVIDE_HIDDEN(g1 = g2);
PROVIDE(unused = g1);
PROVIDE_HIDDEN(another_unused = g1);

#--- chain_with_cycle.t
PROVIDE("f1" = f2 + f3);
PROVIDE(f2 = f3 + f4);
PROVIDE(f3 = f4);
PROVIDE(f4 = f1);
PROVIDE(newsym = f1);

#--- chain2.t
PROVIDE(f2 = undef);
PROVIDE_HIDDEN(f1 = f2);
PROVIDE(newsym = f1);

#--- undef.s
.globl undef
undef: ret

#--- start.s
.globl _start
_start: ret
.data
.quad newsym
