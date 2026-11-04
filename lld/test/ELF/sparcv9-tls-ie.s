# REQUIRES: sparc
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=sparcv9 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=sparcv9 b.s -o b.o
# RUN: ld.lld -shared b.o -o b.so
# RUN: ld.lld -shared a.o b.so -o a.so
# RUN: llvm-readelf -S -r -d a.so | FileCheck %s --check-prefix=IE-REL
# RUN: llvm-objdump -d -j .text --no-show-raw-insn --no-print-imm-hex a.so | FileCheck %s --check-prefix=IE

## a0 is hidden, so its offset in the TLS block, 0x10010, is known and the
## dynamic relocation needs no symbol.
# IE-REL:      .got PROGBITS 00000000002003f8 {{[0-9a-f]+}} 000020
# IE-REL:      (FLAGS) STATIC_TLS
# IE-REL:      Relocation section '.rela.dyn' {{.*}} contains 3 entries:
# IE-REL:      0000000000200400 {{[0-9a-f]+}} R_SPARC_TLS_TPOFF64 10010
# IE-REL-NEXT: 0000000000200410 {{[0-9a-f]+}} R_SPARC_TLS_TPOFF64 {{.*}} b + 0
# IE-REL-NEXT: 0000000000200408 {{[0-9a-f]+}} R_SPARC_TLS_TPOFF64 {{.*}} a1 + 0

## .got[1] - _GLOBAL_OFFSET_TABLE_ = 0x200400 - 0x2003f8 = 8, then 16 and 24.
# IE-LABEL:   <_start>:
# IE-NEXT:      sethi 0, %o0
# IE-NEXT:      add %o0, 8, %o0
# IE-NEXT:      ldx [%l7+%o0], %o0
# IE-NEXT:      add %g7, %o0, %o0
# IE-NEXT:      sethi 0, %o1
# IE-NEXT:      add %o1, 16, %o1
# IE-NEXT:      ld [%l7+%o1], %o2
# IE-NEXT:      add %g7, %o2, %o2
# IE-NEXT:      sethi 0, %o3
# IE-NEXT:      add %o3, 24, %o3
# IE-NEXT:      ldx [%l7+%o3], %o4
# IE-NEXT:      add %g7, %o4, %o4

## The add becomes an xor over the complement the sethi holds, and the load
## becomes a register move, or a nop where it would move a register onto
## itself. b is preemptible, so only its sequence keeps the GOT load: .got
## holds the header and one entry, with one dynamic relocation.
# RUN: ld.lld a.o b.so -o a
# RUN: llvm-readelf -S -r a | FileCheck %s --check-prefix=LE-REL
# RUN: llvm-objdump -d -j .text --no-show-raw-insn --no-print-imm-hex a | FileCheck %s --check-prefix=LE

# LE-REL:      .got PROGBITS 0000000000300378 {{[0-9a-f]+}} 000010
# LE-REL:      Relocation section '.rela.dyn' {{.*}} contains 1 entries:
# LE-REL:      0000000000300380 {{[0-9a-f]+}} R_SPARC_TLS_TPOFF64 {{.*}} b + 0

## a0 - tp = -0x20008, a1 - tp = -0x30010.
# LE-LABEL:   <_start>:
# LE-NEXT:      sethi 128, %o0
# LE-NEXT:      xor %o0, -8, %o0
# LE-NEXT:      nop
# LE-NEXT:      add %g7, %o0, %o0
# LE-NEXT:      sethi 192, %o1
# LE-NEXT:      xor %o1, -16, %o1
# LE-NEXT:      mov %o1, %o2
# LE-NEXT:      add %g7, %o2, %o2
# LE-NEXT:      sethi 0, %o3
# LE-NEXT:      add %o3, 8, %o3
# LE-NEXT:      ldx [%l7+%o3], %o4
# LE-NEXT:      add %g7, %o4, %o4

#--- a.s
.globl _start
_start:
  sethi %tie_hi22(a0), %o0
  add   %o0, %tie_lo10(a0), %o0
  ldx   [%l7 + %o0], %o0, %tie_ldx(a0)
  add   %g7, %o0, %o0, %tie_add(a0)

## ld is the 32-bit load, handled like ldx.
  sethi %tie_hi22(a1), %o1
  add   %o1, %tie_lo10(a1), %o1
  ld    [%l7 + %o1], %o2, %tie_ld(a1)
  add   %g7, %o2, %o2, %tie_add(a1)

## b is defined in a DSO, so it stays preemptible in an executable.
  sethi %tie_hi22(b), %o3
  add   %o3, %tie_lo10(b), %o3
  ldx   [%l7 + %o3], %o4, %tie_ldx(b)
  add   %g7, %o4, %o4, %tie_add(b)

.section .tbss,"awT",@nobits
.globl a0, a1
.hidden a0
.space 8
a1:
  .xword 0
## Pad so that a0 and a1 are far from the thread pointer, and far enough apart,
## that their sethi fields are non-zero and differ.
.space 0x10000
a0:
  .xword 0
.space 0x20000

#--- b.s
.section .tbss,"awT",@nobits
.globl b
b:
  .xword 0
