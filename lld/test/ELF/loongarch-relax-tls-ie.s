# REQUIRES: loongarch
## Test LA64 IE -> LE in various cases.

# RUN: llvm-mc --filetype=obj --triple=loongarch64 -mattr=+relax %s -o %t.o

## Also check --emit-relocs.
# RUN: ld.lld --emit-relocs %t.o -o %t
# RUN: llvm-readelf -x .got %t 2>&1 | FileCheck --check-prefix=LE-GOT %s
# RUN: llvm-objdump -dr --no-show-raw-insn %t | FileCheck --check-prefixes=LER %s

# RUN: ld.lld --emit-relocs --no-relax %t.o -o %t.norelax
# RUN: llvm-readelf -x .got %t.norelax 2>&1 | FileCheck --check-prefix=LE-GOT %s
# RUN: llvm-objdump -dr --no-show-raw-insn %t.norelax | FileCheck --check-prefixes=LE %s

# LE-GOT: could not find section '.got'

# a@tprel = st_value(a) = 0xfff
# b@tprel = st_value(a) = 0x1000
# LE:      20158: nop
# LE-NEXT:          R_LARCH_TLS_IE_PC_HI20 a
# LE-NEXT:          R_LARCH_RELAX   *ABS*
# LE-NEXT:        ori     $a0, $zero, 4095
# LE-NEXT:          R_LARCH_TLS_IE_PC_LO12 a
# LE-NEXT:          R_LARCH_RELAX   *ABS*
# LE-NEXT:        add.d   $a0, $a0, $tp
# LE-NEXT: 20164: lu12i.w $a1, 1
# LE-NEXT:          R_LARCH_TLS_IE_PC_HI20 b
# LE-NEXT:        ori     $a1, $a1, 0
# LE-NEXT:          R_LARCH_TLS_IE_PC_LO12 b
# LE-NEXT:        add.d   $a1, $a1, $tp
# LE-NEXT: 20170: nop
# LE-NEXT:          R_LARCH_TLS_IE_PC_HI20 a
# LE-NEXT:          R_LARCH_RELAX   *ABS*
# LE-NEXT:        lu12i.w $a3, 1
# LE-NEXT:          R_LARCH_TLS_IE_PC_HI20 b
# LE-NEXT:          R_LARCH_RELAX   *ABS*
# LE-NEXT:        ori     $a2, $zero, 4095
# LE-NEXT:          R_LARCH_TLS_IE_PC_LO12 a
# LE-NEXT:        ori     $a3, $a3, 0
# LE-NEXT:          R_LARCH_TLS_IE_PC_LO12 b
# LE-NEXT:        add.d   $a2, $a2, $tp
# LE-NEXT:        add.d   $a3, $a3, $tp

# LER:      20158: ori     $a0, $zero, 4095
# LER-NEXT:          R_LARCH_TLS_IE_PC_HI20 a
# LER-NEXT:          R_LARCH_RELAX   *ABS*
# LER-NEXT:          R_LARCH_TLS_IE_PC_LO12 a
# LER-NEXT:          R_LARCH_RELAX   *ABS*
# LER-NEXT:        add.d   $a0, $a0, $tp
# LER-NEXT: 20160: lu12i.w $a1, 1
# LER-NEXT:          R_LARCH_TLS_IE_PC_HI20 b
# LER-NEXT:        ori     $a1, $a1, 0
# LER-NEXT:          R_LARCH_TLS_IE_PC_LO12 b
# LER-NEXT:        add.d   $a1, $a1, $tp
# LER-NEXT: 2016c: lu12i.w $a3, 1
# LER-NEXT:          R_LARCH_TLS_IE_PC_HI20 a
# LER-NEXT:          R_LARCH_RELAX   *ABS*
# LER-NEXT:          R_LARCH_TLS_IE_PC_HI20 b
# LER-NEXT:          R_LARCH_RELAX   *ABS*
# LER-NEXT:        ori     $a2, $zero, 4095
# LER-NEXT:          R_LARCH_TLS_IE_PC_LO12 a
# LER-NEXT:        ori     $a3, $a3, 0
# LER-NEXT:          R_LARCH_TLS_IE_PC_LO12 b
# LER-NEXT:        add.d   $a2, $a2, $tp
# LER-NEXT:        add.d   $a3, $a3, $tp

la.tls.ie $a0, a    # relax
add.d $a0, $a0, $tp

# PCALAU12I does not have R_LARCH_RELAX. No relaxation.
pcalau12i $a1, %ie_pc_hi20(b)
ld.d $a1, $a1, %ie_pc_lo12(b)
add.d $a1, $a1, $tp

# Test instructions are interleaved.
# PCALAU12I has an R_LARCH_RELAX. We perform relaxation.
pcalau12i $a2, %ie_pc_hi20(a)
.reloc .-4, R_LARCH_RELAX, 0
pcalau12i $a3, %ie_pc_hi20(b)
.reloc .-4, R_LARCH_RELAX, 0
ld.d $a2, $a2, %ie_pc_lo12(a)
ld.d $a3, $a3, %ie_pc_lo12(b)
add.d $a2, $a2, $tp
add.d $a3, $a3, $tp

.section .tbss,"awT",@nobits
.globl a
.zero 0xfff ## Place a at 0xfff, LE needs only one ins.
a:
.zero 1  ## Place b at 0x1000, LE needs two ins.
b:
.zero 4
