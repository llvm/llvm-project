# REQUIRES: loongarch

# RUN: llvm-mc --filetype=obj --triple=loongarch32 %s -o %t.32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 %s -o %t.64.o

# RUN: ld.lld %t.32.o -o %t.32
# RUN: llvm-nm -p %t.32 | FileCheck --check-prefixes=NM %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.32 | FileCheck --check-prefixes=LE %s

# RUN: ld.lld %t.64.o -o %t.64
# RUN: llvm-objdump -d --no-show-raw-insn %t.64 | FileCheck --check-prefixes=LE %s

# RUN: not ld.lld -shared %t.32.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

# ERR: error: relocation R_LARCH_TLS_LE_HI20 against .LANCHOR0 cannot be used with -shared
# ERR: error: relocation R_LARCH_TLS_LE_LO12 against .LANCHOR0 cannot be used with -shared
# ERR: error: relocation R_LARCH_TLS_LE_HI20 against a cannot be used with -shared
# ERR: error: relocation R_LARCH_TLS_LE_LO12 against a cannot be used with -shared

# NM: {{0*}}00000008 b .LANCHOR0
# NM: {{0*}}00000800 B a

## .LANCHOR0@tprel = 8
## a@tprel = 0x800
# LE:      lu12i.w $a0, 0
# LE-NEXT: ori $a0, $a0, 8
# LE-NEXT: lu12i.w $a1, 0
# LE-NEXT: ori $a1, $a1, 2048
# LE-EMPTY:

.text
_start:
la.tls.le $a0, .LANCHOR0
la.tls.le $a1, a

.section .tbss,"awT",@nobits
.space 8
.LANCHOR0:
.space 0x800-8
.globl a
a:
.zero 4
