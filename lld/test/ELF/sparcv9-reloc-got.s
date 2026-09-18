# REQUIRES: sparc
# RUN: llvm-mc -filetype=obj -triple=sparcv9 %s -o %t.o
# RUN: echo '.globl b; b:' | llvm-mc -filetype=obj -triple=sparcv9 - -o %t1.o
# RUN: ld.lld -shared %t1.o -soname=t1.so -o %t1.so

# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=RELOC %s
# RUN: llvm-nm %t | FileCheck --check-prefix=NM %s
# RUN: llvm-objdump -d -s --no-show-raw-insn --no-print-imm-hex %t | \
# RUN:   FileCheck %s --check-prefixes=HEX,CHECK

# RELOC:      .rela.dyn {
# RELOC-NEXT:   0x300368 R_SPARC_GLOB_DAT b 0x0
# RELOC-NEXT: }

# NM: 00000000003002a0 d _DYNAMIC
# NM: 0000000000300360 d _GLOBAL_OFFSET_TABLE_
# NM: 0000000000400378 d a

## .got[0] holds the address of _DYNAMIC.
# HEX:      Contents of section .got:
# HEX-NEXT: 300360 00000000 003002a0 00000000 00000000
# HEX-NEXT: 300370 00000000 00400378

## a: &.got[2] - _GLOBAL_OFFSET_TABLE_ = 0x300370 - 0x300360 = 16
## b: &.got[1] - _GLOBAL_OFFSET_TABLE_ = 0x300368 - 0x300360 = 8
# CHECK:      sethi 0, %o0
# CHECK-NEXT: or %o0, 16, %o0
# CHECK-NEXT: sethi 0, %o1
# CHECK-NEXT: or %o1, 8, %o1
# CHECK-NEXT: ldx [%l7+16], %o2

# RUN: ld.lld --apply-dynamic-relocs %t.o %t1.so -o %t2
# RUN: llvm-objdump -s -j .got %t2 | FileCheck --check-prefix=HEX %s

  sethi %got22(a), %o0
  or    %o0, %got10(a), %o0
  sethi %got22(b), %o1
  or    %o1, %got10(b), %o1
  ldx   [%l7 + %got13(a)], %o2

.data
a:
.quad _GLOBAL_OFFSET_TABLE_
