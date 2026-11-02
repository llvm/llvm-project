# REQUIRES: loongarch
## Test for `TLS IE` with extreme code model.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc --filetype=obj --triple=loongarch64 %t/ie64.s -o %t/ie64.o

## IE
# RUN: ld.lld -shared %t/ie64.o -o %t/ie64.so
# RUN: llvm-readobj -r -d %t/ie64.so | FileCheck %s --check-prefix=IE64-REL
# RUN: llvm-objdump -d --no-show-raw-insn %t/ie64.so | FileCheck %s --check-prefixes=IE64

## IE -> LE
# RUN: ld.lld %t/ie64.o -o %t/64
# RUN: llvm-readelf -r %t/64 | FileCheck %s --check-prefix=NOREL
# RUN: llvm-readelf -x .got %t/64 2>&1 | FileCheck --check-prefix=LE64-GOT %s
# RUN: llvm-objdump -d --no-show-raw-insn %t/64 | FileCheck %s --check-prefixes=LE64

# IE64-REL:      FLAGS STATIC_TLS
# IE64-REL:      .rela.dyn {
# IE64-REL-NEXT:   0x20388 R_LARCH_TLS_TPREL64 - 0xC
# IE64-REL-NEXT:   0x20380 R_LARCH_TLS_TPREL64 a 0x0
# IE64-REL-NEXT: }

## &.got[0] - . = 0x20380 - 0x102a0: 0x10 pages, page offset 0x380
## &.got[1] - . = 0x20388 - 0x102b8: 0x10 pages, page offset 0x388
# IE64:       102a0: pcalau12i	$a0, 16
# IE64-NEXT:         addi.d	$a1, $zero, 896
# IE64-NEXT:         lu32i.d	$a1, 0
# IE64-NEXT:         lu52i.d	$a1, $a1, 0
# IE64-NEXT:         ldx.d	$a0, $a0, $a1
# IE64-NEXT:         add.d	$a0, $a0, $tp
# IE64-NEXT:         pcalau12i	$a1, 16
# IE64-NEXT:         addi.d	$a2, $zero, 904
# IE64-NEXT:         lu32i.d	$a2, 0
# IE64-NEXT:         lu52i.d	$a2, $a2, 0
# IE64-NEXT:         ldx.d	$a1, $a1, $a2
# IE64-NEXT:         add.d	$a1, $a1, $tp

# NOREL: no relocations

# a@tprel = st_value(a) = 0x8
# b@tprel = st_value(a) = 0xc
# LE64-GOT: section '.got':
# LE64-GOT-NEXT: 0x000301f8 08000000 00000000 0c000000 00000000

# LE64:      201c8: pcalau12i $a0, 16
# LE64-NEXT:        addi.d $a1, $zero, 504
# LE64-NEXT:        lu32i.d $a1, 0
# LE64-NEXT:        lu52i.d $a1, $a1, 0
# LE64-NEXT:        ldx.d $a0, $a0, $a1
# LE64:      201e0: pcalau12i $a1, 16
# LE64-NEXT:        addi.d $a2, $zero, 512
# LE64-NEXT:        lu32i.d $a2, 0
# LE64-NEXT:        lu52i.d $a2, $a2, 0
# LE64-NEXT:        ldx.d $a1, $a1, $a2

#--- ie64.s
## la.tls.ie $a0, $a1, a expands to the 5-instruction extreme form:
## pcalau12i/addi.d/lu32i.d/lu52i.d + ldx.d with R_LARCH_TLS_IE_PC_HI20/LO12,
## R_LARCH_TLS_IE64_PC_LO20/HI12.
la.tls.ie $a0, $a1, a
add.d $a0, $a0, $tp
la.tls.ie $a1, $a2, b
add.d $a1, $a1, $tp

.section .tbss,"awT",@nobits
.global a
.zero 8
a:
.zero 4
b:
