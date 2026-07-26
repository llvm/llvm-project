# REQUIRES: sparc
# RUN: llvm-mc -filetype=obj -triple=sparcv9 %s -o %t.o
# RUN: echo '.globl b; b:' | llvm-mc -filetype=obj -triple=sparcv9 - -o %t1.o
# RUN: ld.lld -shared %t1.o -soname=t1.so -o %t1.so

# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=RELOC %s
# RUN: llvm-nm %t | FileCheck --check-prefix=NM %s
# RUN: llvm-readelf -x .got %t | FileCheck --check-prefix=HEX %s
# RUN: llvm-objdump -d --no-show-raw-insn --no-print-imm-hex %t | FileCheck %s

# RELOC:      .rela.dyn {
# RELOC-NEXT:   0x300358 R_SPARC_GLOB_DAT b 0x0
# RELOC-NEXT: }

# NM: 0000000000300358 d _GLOBAL_OFFSET_TABLE_
# NM: 0000000000400368 d a

# HEX:      section '.got':
# HEX-NEXT: 0x00300358 00000000 00000000 00000000 00400368

## a: &.got[1] - _GLOBAL_OFFSET_TABLE_ = 0x300360 - 0x300358 = 8
## b: &.got[0] - _GLOBAL_OFFSET_TABLE_ = 0x300358 - 0x300358 = 0
# CHECK:      sethi 0, %o0
# CHECK-NEXT: or %o0, 8, %o0
# CHECK-NEXT: sethi 0, %o1
# CHECK-NEXT: or %o1, 0, %o1

# RUN: ld.lld --apply-dynamic-relocs %t.o %t1.so -o %t2
# RUN: llvm-readelf -x .got %t2 | FileCheck --check-prefix=HEX %s

  sethi %got22(a), %o0
  or    %o0, %got10(a), %o0
  sethi %got22(b), %o1
  or    %o1, %got10(b), %o1

.data
a:
.quad _GLOBAL_OFFSET_TABLE_
