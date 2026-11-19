# REQUIRES: sparc
## Test R_SPARC_GOTDATA_OP_HIX22, R_SPARC_GOTDATA_OP_LOX10 and R_SPARC_GOTDATA_OP.
## The GOT load is optimized to an add of the symbol's GOT-relative address
## unless the symbol is preemptible or absolute.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=sparcv9 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=sparcv9 b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=sparcv9 c.s -o c.o
# RUN: llvm-mc -filetype=obj -triple=sparcv9 ifunc.s -o ifunc.o

# RUN: ld.lld -shared a.o c.o -o a.so
# RUN: llvm-readelf -S -r a.so | FileCheck %s --check-prefix=RELOC
# RUN: llvm-objdump -d -j .text --no-show-raw-insn --no-print-imm-hex a.so | FileCheck %s

# RELOC:      [ 8] .got PROGBITS 0000000000200390 {{[0-9a-f]+}} 000020
# RELOC:      Relocation section '.rela.dyn' {{.*}} contains 1 entries:
# RELOC:      0000000000200398 {{.*}} R_SPARC_GLOB_DAT {{.*}} foo + 0

## foo is preemptible, so the GOT load stays.
## .got[1] - _GLOBAL_OFFSET_TABLE_ = 0x200398 - 0x200390 = 8.
# CHECK-LABEL: <_start>:
# CHECK-NEXT:   sethi 0, %l1
# CHECK-NEXT:   xor %l1, 8, %l1
# CHECK-NEXT:   ldx [%l7+%l1], %l2
## hid0 - _GLOBAL_OFFSET_TABLE_ = 0x3003b0 - 0x200390 = 0x100020.
# CHECK-NEXT:   sethi 1024, %g1
# CHECK-NEXT:   xor %g1, 32, %g1
# CHECK-NEXT:   add %l7, %g1, %g1
## hid1 - _GLOBAL_OFFSET_TABLE_ = 0x1002ec - 0x200390 = -0x1000a4. sethi holds
## the complement, which the sign-extended xor operand undoes.
# CHECK-NEXT:   sethi 1024, %o2
# CHECK-NEXT:   xor %o2, -164, %o2
# CHECK-NEXT:   add %i0, %o2, %i5
## An undefined symbol and an absolute symbol both have an absolute address, so
## neither is optimized.
# CHECK-NEXT:   sethi 0, %l1
# CHECK-NEXT:   xor %l1, 16, %l1
# CHECK-NEXT:   ldx [%l7+%l1], %l2
# CHECK-NEXT:   sethi 0, %g4
# CHECK-NEXT:   xor %g4, 24, %g4
# CHECK-NEXT:   ldx [%l7+%g4], %o3

## foo is not preemptible in an executable and is optimized as well. undefweak
## and absolute keep the GOT load in every mode.
# RUN: ld.lld a.o c.o -o a
# RUN: llvm-objdump -d -j .text --no-show-raw-insn --no-print-imm-hex a | FileCheck %s --check-prefix=NOPIE
# NOPIE-LABEL: <_start>:
# NOPIE-NEXT:   sethi 1024, %l1
# NOPIE-NEXT:   xor %l1, -4, %l1
# NOPIE-NEXT:   add %l7, %l1, %l2
# NOPIE:        sethi 0, %l1
# NOPIE-NEXT:   xor %l1, 8, %l1
# NOPIE-NEXT:   ldx [%l7+%l1], %l2
# NOPIE-NEXT:   sethi 0, %g4
# NOPIE-NEXT:   xor %g4, 16, %g4
# NOPIE-NEXT:   ldx [%l7+%g4], %o3

## .got holds nothing but the header, yet must be kept: %l7 points at it.
# RUN: ld.lld -shared b.o -o b.so
# RUN: llvm-readelf -S b.so | FileCheck %s --check-prefix=HEADER
# HEADER: [ 7] .got PROGBITS 00000000002002b8 {{[0-9a-f]+}} 000008

# RUN: not ld.lld -shared ifunc.o 2>&1 | FileCheck %s --check-prefix=IFUNC --implicit-check-not=error:
# IFUNC: error: {{.*}}relocation R_SPARC_WDISP19 out of range

## The optimization is decided before addresses are known, so an out-of-range
## offset is an error rather than a fallback to the GOT load. The offset is
## reported as signed, not as the complement the sethi encodes. Only hid0, in
## .data, is out of range; the other symbols stay next to .got.
# RUN: not ld.lld -shared a.o c.o --section-start=.text=0x10000 --section-start=.got=0x20000 \
# RUN:   --section-start=.data=0x200000000 2>&1 | FileCheck %s --check-prefix=ERR -DVAL=8589803520 --implicit-check-not=error:
# RUN: not ld.lld -shared a.o c.o --section-start=.data=0x10000 --section-start=.text=0x200000000 \
# RUN:   --section-start=.got=0x200020000 2>&1 | FileCheck %s --check-prefix=ERR -DVAL=-8590000128 --implicit-check-not=error:
# ERR: error: a.o:(.text+0xc): relocation R_SPARC_GOTDATA_OP_HIX22 out of range: [[VAL]] is not in [-4294967296, 4294967295]; references 'hid0'

#--- a.s
.text
.globl _start
_start:
  sethi %gdop_hix22(foo), %l1
  xor   %l1, %gdop_lox10(foo), %l1
  ldx   [%l7 + %l1], %l2, %gdop(foo)

  sethi %gdop_hix22(hid0), %g1
  xor   %g1, %gdop_lox10(hid0), %g1
  ldx   [%l7 + %g1], %g1, %gdop(hid0)

## The add keeps rs1, rs2 and rd of the load, and ld is optimized like ldx.
  sethi %gdop_hix22(hid1), %o2
  xor   %o2, %gdop_lox10(hid1), %o2
  ld    [%i0 + %o2], %i5, %gdop(hid1)

  sethi %gdop_hix22(undefweak), %l1
  xor   %l1, %gdop_lox10(undefweak), %l1
  ldx   [%l7 + %l1], %l2, %gdop(undefweak)

  sethi %gdop_hix22(absolute), %g4
  xor   %g4, %gdop_lox10(absolute), %g4
  ldx   [%l7 + %g4], %o3, %gdop(absolute)

.globl foo, hid0, hid1
.hidden hid0, hid1
foo:
hid1:
  nop

.weak undefweak
.hidden undefweak

.data
hid0:
  .xword 0

#--- b.s
.globl _start
_start:
  sethi %gdop_hix22(local), %l1
  xor   %l1, %gdop_lox10(local), %l1
  ldx   [%l7 + %l1], %l2, %gdop(local)
local:
  nop

#--- c.s
## Separate file to test %gdop_lox10(absolute) relocations in a.s
.globl absolute
.hidden absolute
absolute = 0x1234

#--- ifunc.s
.globl _start
_start:
  sethi %gdop_hix22(ifunc), %l1
  xor   %l1, %gdop_lox10(ifunc), %l1
  ldx   [%l7 + %l1], %l2, %gdop(ifunc)

.globl ifunc
.hidden ifunc
.type ifunc, @gnu_indirect_function
ifunc:
  nop
