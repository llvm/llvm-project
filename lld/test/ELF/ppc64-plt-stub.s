# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %p/Inputs/shared-ppc64.s -o %t2.o
# RUN: ld.lld -shared %t2.o -soname=t2.so -o %t2.so
# RUN: ld.lld %t.o %t2.so -o %t
# RUN: llvm-readelf -S -d %t | FileCheck --check-prefix=SEC %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %p/Inputs/shared-ppc64.s -o %t2.o
# RUN: ld.lld -shared %t2.o -soname=t2.so -o %t2.so
# RUN: ld.lld %t.o %t2.so -o %t
# RUN: llvm-readelf -S -d %t | FileCheck --check-prefix=SEC %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

## DT_PLTGOT points to .plt
# SEC: .plt NOBITS 0000000010030400 000400 000018
# SEC: 0x0000000000000003 (PLTGOT) 0x10030400
# SEC: 0x0000000070000000 (PPC64_GLINK) 0x100102e0
# SEC: 0x0000000070000003 (PPC64_OPT) 0x0

## .plt[0] holds the address of _dl_runtime_resolve.
## .plt[1] holds the link map.
## The JMP_SLOT relocation is stored at .plt[2]
# RELOC: 0x10030010 R_PPC64_JMP_SLOT foo 0x0

# CHECK:      <_start>:
# CHECK:      10010298: bl 0x100102b0

# CHECK-LABEL: 00000000100102b0 <__plt_foo>:
# CHECK-NEXT:      std 2, 24(1)
# CHECK-NEXT:      addis 12, 2, 1
# CHECK-NEXT:      ld 12, -32744(12)
# CHECK-NEXT:      mtctr 12
# CHECK-NEXT:      bctr

# CHECK-LABEL: <.glink>:
# CHECK-NEXT:  100102c4: mflr 0
# CHECK-NEXT:            bcl 20, 31, 0x100102cc <.glink+0x8>
# CHECK-NEXT:            mflr 11
# CHECK-NEXT:            mtlr 0
# CHECK-NEXT:            sub     12, 12, 11
# CHECK-NEXT:            addi 0, 12, -52
# CHECK-NEXT:            rldicl 0, 0, 62, 2
# CHECK-NEXT:            ld 12, 44(11)
# CHECK-NEXT:            add 11, 12, 11
# CHECK-NEXT:            ld 12, 0(11)
# CHECK-NEXT:            ld 11, 8(11)
# CHECK-NEXT:            mtctr 12
# CHECK-NEXT:            bctr
# CHECK-NEXT:            <unknown>
# CHECK-NEXT:            <unknown>
## DT_PPC64_GLINK+0x20 = 0x10010300
# CHECK-NEXT:  10010300: b 0x100102c4 <.glink>
# CHECK-NOT:   {{.}}

        .text
        .abiversion 2
        .globl  _start
        .p2align        4
        .type   _start,@function
_start:
.Lfunc_begin0:
.Lfunc_gep0:
  addis 2, 12, .TOC.-.Lfunc_gep0@ha
  addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
  .localentry     _start, .Lfunc_lep0-.Lfunc_gep0
  bl foo
  nop
  li 0, 1
  sc
  .size _start, .-.Lfunc_begin0
