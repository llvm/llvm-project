# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t

# RUN: llvm-mc --filetype=obj --triple=loongarch32 %t/a.s -o %t/a.la32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 %t/a.s -o %t/a.la64.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 %t/extreme.s -o %t/extreme.o

# RUN: ld.lld %t/a.la32.o --section-start=.rodata=0x11000 --section-start=.text=0x11ffc -o %t/case1.la32
# RUN: ld.lld %t/a.la64.o --section-start=.rodata=0x11000 --section-start=.text=0x11ffc -o %t/case1.la64
# RUN: llvm-objdump -d --no-show-raw-insn %t/case1.la32 | FileCheck %s --check-prefix=CASE1
# RUN: llvm-objdump -d --no-show-raw-insn %t/case1.la64 | FileCheck %s --check-prefix=CASE1
# CASE1:      pcalau12i $a0, 0
# CASE1-NEXT: ld.w      $a0, $a0, 0

# RUN: ld.lld %t/a.la32.o --section-start=.rodata=0x11000 --section-start=.text=0x12000 -o %t/case2.la32
# RUN: ld.lld %t/a.la64.o --section-start=.rodata=0x11000 --section-start=.text=0x12000 -o %t/case2.la64
# RUN: llvm-objdump -d --no-show-raw-insn %t/case2.la32 | FileCheck %s --check-prefix=CASE2
# RUN: llvm-objdump -d --no-show-raw-insn %t/case2.la64 | FileCheck %s --check-prefix=CASE2
# CASE2:      pcalau12i $a0, -1
# CASE2-NEXT: ld.w      $a0, $a0, 0

# RUN: ld.lld %t/a.la32.o --section-start=.rodata=0x117ff --section-start=.text=0x12000 -o %t/case3.la32
# RUN: ld.lld %t/a.la64.o --section-start=.rodata=0x117ff --section-start=.text=0x12000 -o %t/case3.la64
# RUN: llvm-objdump -d --no-show-raw-insn %t/case3.la32 | FileCheck %s --check-prefix=CASE3
# RUN: llvm-objdump -d --no-show-raw-insn %t/case3.la64 | FileCheck %s --check-prefix=CASE3
# CASE3:      pcalau12i $a0, -1
# CASE3-NEXT: ld.w      $a0, $a0, 2047

# RUN: ld.lld %t/a.la32.o --section-start=.rodata=0x11800 --section-start=.text=0x12000 -o %t/case4.la32
# RUN: ld.lld %t/a.la64.o --section-start=.rodata=0x11800 --section-start=.text=0x12000 -o %t/case4.la64
# RUN: llvm-objdump -d --no-show-raw-insn %t/case4.la32 | FileCheck %s --check-prefix=CASE4
# RUN: llvm-objdump -d --no-show-raw-insn %t/case4.la64 | FileCheck %s --check-prefix=CASE4
# CASE4:      pcalau12i $a0, 0
# CASE4-NEXT: ld.w      $a0, $a0, -2048

# RUN: ld.lld %t/a.la32.o --section-start=.rodata=0x12004 --section-start=.text=0x11ffc -o %t/case5.la32
# RUN: ld.lld %t/a.la64.o --section-start=.rodata=0x12004 --section-start=.text=0x11ffc -o %t/case5.la64
# RUN: llvm-objdump -d --no-show-raw-insn %t/case5.la32 | FileCheck %s --check-prefix=CASE5
# RUN: llvm-objdump -d --no-show-raw-insn %t/case5.la64 | FileCheck %s --check-prefix=CASE5
# CASE5:      pcalau12i $a0, 1
# CASE5-NEXT: ld.w      $a0, $a0, 4

# RUN: ld.lld %t/a.la32.o --section-start=.rodata=0x12800 --section-start=.text=0x11ffc -o %t/case6.la32
# RUN: ld.lld %t/a.la64.o --section-start=.rodata=0x12800 --section-start=.text=0x11ffc -o %t/case6.la64
# RUN: llvm-objdump -d --no-show-raw-insn %t/case6.la32 | FileCheck %s --check-prefix=CASE6
# RUN: llvm-objdump -d --no-show-raw-insn %t/case6.la64 | FileCheck %s --check-prefix=CASE6
# CASE6:      pcalau12i $a0, 2
# CASE6-NEXT: ld.w      $a0, $a0, -2048

# RUN: ld.lld %t/a.la32.o --section-start=.rodata=0x7ffff123 --section-start=.text=0x0 -o %t/case7.la32
# RUN: ld.lld %t/a.la64.o --section-start=.rodata=0x7ffff123 --section-start=.text=0x0 -o %t/case7.la64
# RUN: llvm-objdump -d --no-show-raw-insn %t/case7.la32 | FileCheck %s --check-prefix=CASE7
# RUN: llvm-objdump -d --no-show-raw-insn %t/case7.la64 | FileCheck %s --check-prefix=CASE7
# CASE7:      pcalau12i $a0, 524287
# CASE7-NEXT: ld.w      $a0, $a0, 291

# RUN: ld.lld %t/a.la32.o --section-start=.rodata=0x7ffffabc --section-start=.text=0x0 -o %t/case8.la32
# RUN: ld.lld %t/a.la64.o --section-start=.rodata=0x7ffffabc --section-start=.text=0x0 -o %t/case8.la64
# RUN: llvm-objdump -d --no-show-raw-insn %t/case8.la32 | FileCheck %s --check-prefix=CASE8
# RUN: llvm-objdump -d --no-show-raw-insn %t/case8.la64 | FileCheck %s --check-prefix=CASE8
# CASE8:      pcalau12i $a0, -524288
# CASE8-NEXT: ld.w      $a0, $a0, -1348

# RUN: ld.lld %t/a.la32.o --section-start=.rodata=0x10123 --section-start=.text=0x80010000 -o %t/case9.la32
# RUN: ld.lld %t/a.la64.o --section-start=.rodata=0x10123 --section-start=.text=0x80010000 -o %t/case9.la64
# RUN: llvm-objdump -d --no-show-raw-insn %t/case9.la32 | FileCheck %s --check-prefix=CASE9
# RUN: llvm-objdump -d --no-show-raw-insn %t/case9.la64 | FileCheck %s --check-prefix=CASE9
# CASE9:      pcalau12i $a0, -524288
# CASE9-NEXT: ld.w      $a0, $a0, 291

## page delta = 0x4443333322222000, page offset = 0x111
## %pc_lo12   = 0x111 = 273
## %pc_hi20   = 0x22222 = 139810
## %pc64_lo20 = 0x33333 = 209715
## %pc64_hi12 = 0x444 = 1092
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0x4443333334567111 --section-start=.text=0x0000000012345678 -o %t/extreme0
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme0 | FileCheck %s --check-prefix=EXTREME0
# EXTREME0:      addi.d $t0, $zero, 273
# EXTREME0-NEXT: pcalau12i $t1, 139810
# EXTREME0-NEXT: lu32i.d   $t0, 209715
# EXTREME0-NEXT: lu52i.d   $t0, $t0, 1092

## page delta = 0x4443333222223000, page offset = 0x888
## %pc_lo12   = 0x888 = -1912
## %pc_hi20   = 0x22223 = 139811
## %pc64_lo20 = 0x33332 = 209714
## %pc64_hi12 = 0x444 = 1092
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0x4443333334567888 --section-start=.text=0x0000000012345678 -o %t/extreme1
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme1 | FileCheck %s --check-prefix=EXTREME1
# EXTREME1:      addi.d $t0, $zero, -1912
# EXTREME1-NEXT: pcalau12i $t1, 139811
# EXTREME1-NEXT: lu32i.d   $t0, 209714
# EXTREME1-NEXT: lu52i.d   $t0, $t0, 1092

## page delta = 0x4443333499999000, page offset = 0x111
## %pc_lo12   = 0x111 = 273
## %pc_hi20   = 0x99999 = -419431
## %pc64_lo20 = 0x33334 = 209716
## %pc64_hi12 = 0x444 = 1092
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0x44433333abcde111 --section-start=.text=0x0000000012345678 -o %t/extreme2
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme2 | FileCheck %s --check-prefix=EXTREME2
# EXTREME2:      addi.d $t0, $zero, 273
# EXTREME2-NEXT: pcalau12i $t1, -419431
# EXTREME2-NEXT: lu32i.d   $t0, 209716
# EXTREME2-NEXT: lu52i.d   $t0, $t0, 1092

## page delta = 0x444333339999a000, page offset = 0x888
## %pc_lo12   = 0x888 = -1912
## %pc_hi20   = 0x9999a = -419430
## %pc64_lo20 = 0x33333 = 209715
## %pc64_hi12 = 0x444 = 1092
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0x44433333abcde888 --section-start=.text=0x0000000012345678 -o %t/extreme3
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme3 | FileCheck %s --check-prefix=EXTREME3
# EXTREME3:      addi.d $t0, $zero, -1912
# EXTREME3-NEXT: pcalau12i $t1, -419430
# EXTREME3-NEXT: lu32i.d   $t0, 209715
# EXTREME3-NEXT: lu52i.d   $t0, $t0, 1092

## page delta = 0x444aaaaa22222000, page offset = 0x111
## %pc_lo12   = 0x111 = 273
## %pc_hi20   = 0x22222 = 139810
## %pc64_lo20 = 0xaaaaa = -349526
## %pc64_hi12 = 0x444 = 1092
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0x444aaaaa34567111 --section-start=.text=0x0000000012345678 -o %t/extreme4
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme4 | FileCheck %s --check-prefix=EXTREME4
# EXTREME4:      addi.d $t0, $zero, 273
# EXTREME4-NEXT: pcalau12i $t1, 139810
# EXTREME4-NEXT: lu32i.d   $t0, -349526
# EXTREME4-NEXT: lu52i.d   $t0, $t0, 1092

## page delta = 0x444aaaa922223000, page offset = 0x888
## %pc_lo12   = 0x888 = -1912
## %pc_hi20   = 0x22223 = 139811
## %pc64_lo20 = 0xaaaa9 = -349527
## %pc64_hi12 = 0x444 = 1092
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0x444aaaaa34567888 --section-start=.text=0x0000000012345678 -o %t/extreme5
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme5 | FileCheck %s --check-prefix=EXTREME5
# EXTREME5:      addi.d $t0, $zero, -1912
# EXTREME5-NEXT: pcalau12i $t1, 139811
# EXTREME5-NEXT: lu32i.d   $t0, -349527
# EXTREME5-NEXT: lu52i.d   $t0, $t0, 1092

## page delta = 0x444aaaab99999000, page offset = 0x111
## %pc_lo12   = 0x111 = 273
## %pc_hi20   = 0x99999 = -419431
## %pc64_lo20 = 0xaaaab = -349525
## %pc64_hi12 = 0x444 = 1092
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0x444aaaaaabcde111 --section-start=.text=0x0000000012345678 -o %t/extreme6
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme6 | FileCheck %s --check-prefix=EXTREME6
# EXTREME6:      addi.d $t0, $zero, 273
# EXTREME6-NEXT: pcalau12i $t1, -419431
# EXTREME6-NEXT: lu32i.d   $t0, -349525
# EXTREME6-NEXT: lu52i.d   $t0, $t0, 1092

## page delta = 0x444aaaaa9999a000, page offset = 0x888
## %pc_lo12   = 0x888 = -1912
## %pc_hi20   = 0x9999a = -419430
## %pc64_lo20 = 0xaaaaa = -349526
## %pc64_hi12 = 0x444 = 1092
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0x444aaaaaabcde888 --section-start=.text=0x0000000012345678 -o %t/extreme7
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme7 | FileCheck %s --check-prefix=EXTREME7
# EXTREME7:      addi.d $t0, $zero, -1912
# EXTREME7-NEXT: pcalau12i $t1, -419430
# EXTREME7-NEXT: lu32i.d   $t0, -349526
# EXTREME7-NEXT: lu52i.d   $t0, $t0, 1092

## page delta = 0xbbb3333322222000, page offset = 0x111
## %pc_lo12   = 0x111 = 273
## %pc_hi20   = 0x22222 = 139810
## %pc64_lo20 = 0x33333 = 209715
## %pc64_hi12 = 0xbbb = -1093
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0xbbb3333334567111 --section-start=.text=0x0000000012345678 -o %t/extreme8
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme8 | FileCheck %s --check-prefix=EXTREME8
# EXTREME8:      addi.d $t0, $zero, 273
# EXTREME8-NEXT: pcalau12i $t1, 139810
# EXTREME8-NEXT: lu32i.d   $t0, 209715
# EXTREME8-NEXT: lu52i.d   $t0, $t0, -1093

## page delta = 0xbbb3333222223000, page offset = 0x888
## %pc_lo12   = 0x888 = -1912
## %pc_hi20   = 0x22223 = 139811
## %pc64_lo20 = 0x33332 = 209714
## %pc64_hi12 = 0xbbb = -1093
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0xbbb3333334567888 --section-start=.text=0x0000000012345678 -o %t/extreme9
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme9 | FileCheck %s --check-prefix=EXTREME9
# EXTREME9:      addi.d $t0, $zero, -1912
# EXTREME9-NEXT: pcalau12i $t1, 139811
# EXTREME9-NEXT: lu32i.d   $t0, 209714
# EXTREME9-NEXT: lu52i.d   $t0, $t0, -1093

## page delta = 0xbbb3333499999000, page offset = 0x111
## %pc_lo12   = 0x111 = 273
## %pc_hi20   = 0x99999 = -419431
## %pc64_lo20 = 0x33334 = 209716
## %pc64_hi12 = 0xbbb = -1093
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0xbbb33333abcde111 --section-start=.text=0x0000000012345678 -o %t/extreme10
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme10 | FileCheck %s --check-prefix=EXTREME10
# EXTREME10:      addi.d $t0, $zero, 273
# EXTREME10-NEXT: pcalau12i $t1, -419431
# EXTREME10-NEXT: lu32i.d   $t0, 209716
# EXTREME10-NEXT: lu52i.d   $t0, $t0, -1093

## page delta = 0xbbb333339999a000, page offset = 0x888
## %pc_lo12   = 0x888 = -1912
## %pc_hi20   = 0x9999a = -419430
## %pc64_lo20 = 0x33333 = 209715
## %pc64_hi12 = 0xbbb = -1093
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0xbbb33333abcde888 --section-start=.text=0x0000000012345678 -o %t/extreme11
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme11 | FileCheck %s --check-prefix=EXTREME11
# EXTREME11:      addi.d $t0, $zero, -1912
# EXTREME11-NEXT: pcalau12i $t1, -419430
# EXTREME11-NEXT: lu32i.d   $t0, 209715
# EXTREME11-NEXT: lu52i.d   $t0, $t0, -1093

## page delta = 0xbbbaaaaa22222000, page offset = 0x111
## %pc_lo12   = 0x111 = 273
## %pc_hi20   = 0x22222 = 139810
## %pc64_lo20 = 0xaaaaa = -349526
## %pc64_hi12 = 0xbbb = -1093
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0xbbbaaaaa34567111 --section-start=.text=0x0000000012345678 -o %t/extreme12
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme12 | FileCheck %s --check-prefix=EXTREME12
# EXTREME12:      addi.d $t0, $zero, 273
# EXTREME12-NEXT: pcalau12i $t1, 139810
# EXTREME12-NEXT: lu32i.d   $t0, -349526
# EXTREME12-NEXT: lu52i.d   $t0, $t0, -1093

## page delta = 0xbbbaaaa922223000, page offset = 0x888
## %pc_lo12   = 0x888 = -1912
## %pc_hi20   = 0x22223 = 139811
## %pc64_lo20 = 0xaaaa9 = -349527
## %pc64_hi12 = 0xbbb = -1093
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0xbbbaaaaa34567888 --section-start=.text=0x0000000012345678 -o %t/extreme13
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme13 | FileCheck %s --check-prefix=EXTREME13
# EXTREME13:      addi.d $t0, $zero, -1912
# EXTREME13-NEXT: pcalau12i $t1, 139811
# EXTREME13-NEXT: lu32i.d   $t0, -349527
# EXTREME13-NEXT: lu52i.d   $t0, $t0, -1093

## page delta = 0xbbbaaaab99999000, page offset = 0x111
## %pc_lo12   = 0x111 = 273
## %pc_hi20   = 0x99999 = -419431
## %pc64_lo20 = 0xaaaab = -349525
## %pc64_hi12 = 0xbbb = -1093
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0xbbbaaaaaabcde111 --section-start=.text=0x0000000012345678 -o %t/extreme14
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme14 | FileCheck %s --check-prefix=EXTREME14
# EXTREME14:      addi.d $t0, $zero, 273
# EXTREME14-NEXT: pcalau12i $t1, -419431
# EXTREME14-NEXT: lu32i.d   $t0, -349525
# EXTREME14-NEXT: lu52i.d   $t0, $t0, -1093

## page delta = 0xbbbaaaaa9999a000, page offset = 0x888
## %pc_lo12   = 0x888 = -1912
## %pc_hi20   = 0x9999a = -419430
## %pc64_lo20 = 0xaaaaa = -349526
## %pc64_hi12 = 0xbbb = -1093
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0xbbbaaaaaabcde888 --section-start=.text=0x0000000012345678 -o %t/extreme15
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme15 | FileCheck %s --check-prefix=EXTREME15
# EXTREME15:      addi.d $t0, $zero, -1912
# EXTREME15-NEXT: pcalau12i $t1, -419430
# EXTREME15-NEXT: lu32i.d   $t0, -349526
# EXTREME15-NEXT: lu52i.d   $t0, $t0, -1093

## FIXME: Correct %pc64_lo20 should be 0xfffff (-1) and %pc64_hi12 should be 0xfff (-1), but current values are:
## page delta = 0x0000000000000000, page offset = 0x888
## %pc_lo12   = 0x888 = -1912
## %pc_hi20   = 0x00000 = 0
## %pc64_lo20 = 0x00000 = 0
## %pc64_hi12 = 0x00000 = 0
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0x0000000012344888 --section-start=.text=0x0000000012345678 -o %t/extreme16
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme16 | FileCheck %s --check-prefix=EXTREME16
# EXTREME16:      addi.d $t0, $zero, -1912
# EXTREME16-NEXT: pcalau12i $t1, 0
# EXTREME16-NEXT: lu32i.d   $t0, 0
# EXTREME16-NEXT: lu52i.d   $t0, $t0, 0

## FIXME: Correct %pc64_lo20 should be 0x00000 (0) and %pc64_hi12 should be 0x000 (0), but current values are:
## page delta = 0xffffffff80000000, page offset = 0x888
## %pc_lo12   = 0x888 = -1912
## %pc_hi20   = 0x80000 = -524288
## %pc64_lo20 = 0xfffff = -1
## %pc64_hi12 = 0xfff = -1
# RUN: ld.lld %t/extreme.o --section-start=.rodata=0x000071238ffff888 --section-start=.text=0x0000712310000678 -o %t/extreme17
# RUN: llvm-objdump -d --no-show-raw-insn %t/extreme17 | FileCheck %s --check-prefix=EXTREME17
# EXTREME17:      addi.d $t0, $zero, -1912
# EXTREME17-NEXT: pcalau12i $t1, -524288
# EXTREME17-NEXT: lu32i.d   $t0, -1
# EXTREME17-NEXT: lu52i.d   $t0, $t0, -1

#--- a.s
.rodata
x:
.word 10
.text
.global _start
_start:
    pcalau12i $a0, %pc_hi20(x)
    ld.w      $a0, $a0, %pc_lo12(x)

#--- extreme.s
.rodata
x:
.word 10
.text
.global _start
_start:
    addi.d    $t0, $zero, %pc_lo12(x)
    pcalau12i $t1, %pc_hi20(x)
    lu32i.d   $t0, %pc64_lo20(x)
    lu52i.d   $t0, $t0, %pc64_hi12(x)
