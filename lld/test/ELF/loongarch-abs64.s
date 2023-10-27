# REQUIRES: loongarch

# RUN: llvm-mc --filetype=obj --triple=loongarch64-unknown-elf %s -o %t.la64.o

# RUN: ld.lld %t.la64.o --defsym foo=0 --defsym bar=42 -o %t.la64.1
# RUN: llvm-objdump --no-show-raw-insn -d %t.la64.1 | FileCheck --check-prefix=CASE1 %s
# CASE1:      lu12i.w $a0, 0
# CASE1-NEXT: ori     $a0, $a0, 0
# CASE1-NEXT: lu32i.d $a0, 0
# CASE1-NEXT: lu52i.d $a0, $a0, 0
# CASE1-NEXT: lu12i.w $a1, 0
# CASE1-NEXT: ori     $a1, $a1, 42
# CASE1-NEXT: lu32i.d $a1, 0
# CASE1-NEXT: lu52i.d $a1, $a1, 0

# RUN: ld.lld %t.la64.o --defsym foo=0x12345678 --defsym bar=0x87654321 -o %t.la64.2
# RUN: llvm-objdump --no-show-raw-insn -d %t.la64.2 | FileCheck --check-prefix=CASE2 %s
# CASE2:      lu12i.w $a0, 74565
# CASE2-NEXT: ori     $a0, $a0, 1656
# CASE2-NEXT: lu32i.d $a0, 0
# CASE2-NEXT: lu52i.d $a0, $a0, 0
# CASE2-NEXT: lu12i.w $a1, -493996
# CASE2-NEXT: ori     $a1, $a1, 801
# CASE2-NEXT: lu32i.d $a1, 0
# CASE2-NEXT: lu52i.d $a1, $a1, 0

# RUN: ld.lld %t.la64.o --defsym foo=0x12345fedcb678 --defsym bar=0xfedcb12345000 -o %t.la64.3
# RUN: llvm-objdump --no-show-raw-insn -d %t.la64.3 | FileCheck --check-prefix=CASE3 %s
# CASE3:      lu12i.w $a0, -4661
# CASE3-NEXT: ori     $a0, $a0, 1656
# CASE3-NEXT: lu32i.d $a0, 74565
# CASE3-NEXT: lu52i.d $a0, $a0, 0
# CASE3-NEXT: lu12i.w $a1, 74565
# CASE3-NEXT: ori     $a1, $a1, 0
# CASE3-NEXT: lu32i.d $a1, -4661
# CASE3-NEXT: lu52i.d $a1, $a1, 0

# RUN: ld.lld %t.la64.o --defsym foo=0xfffffeeeeeddd --defsym bar=0xfff00000f1111222 -o %t.la64.4
# RUN: llvm-objdump --no-show-raw-insn -d %t.la64.4 | FileCheck --check-prefix=CASE4 %s
# CASE4:      lu12i.w $a0, -69906
# CASE4-NEXT: ori     $a0, $a0, 3549
# CASE4-NEXT: lu32i.d $a0, -1
# CASE4-NEXT: lu52i.d $a0, $a0, 0
# CASE4-NEXT: lu12i.w $a1, -61167
# CASE4-NEXT: ori     $a1, $a1, 546
# CASE4-NEXT: lu32i.d $a1, 0
# CASE4-NEXT: lu52i.d $a1, $a1, -1

.global _start

_start:
1:
    lu12i.w $a0, %abs_hi20(foo)
.reloc 1b, R_LARCH_MARK_LA, foo
    ori     $a0, $a0, %abs_lo12(foo)
    lu32i.d $a0, %abs64_lo20(foo)
    lu52i.d $a0, $a0, %abs64_hi12(foo)

2:
    lu12i.w $a1, %abs_hi20(bar)
.reloc 1b, R_LARCH_MARK_LA, bar
    ori     $a1, $a1, %abs_lo12(bar)
    lu32i.d $a1, %abs64_lo20(bar)
    lu52i.d $a1, $a1, %abs64_hi12(bar)
