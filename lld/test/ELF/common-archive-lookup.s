# REQUIRES: ppc

# RUN: rm -rf %t.dir
# RUN: split-file %s %t.dir
# RUN: cd %t.dir

## Object files.
# RUN: llvm-mc -triple=powerpc64le -filetype=obj ref.s -o %t1.o
# RUN: llvm-mc -triple=powerpc64le -filetype=obj refanddef.s -o %t2.o
# RUN: llvm-mc -triple=powerpc64le -filetype=obj def.s -o %tstrong_data_only.o
# RUN: llvm-mc -triple=powerpc64le -filetype=obj weak.s -o %tweak_data_only.o

# RUN: llvm-mc -triple=powerpc64le -filetype=obj main.s -o %tmain.o

## Object file archives.
# RUN: llvm-ar crs %t1.a %t1.o %tstrong_data_only.o
# RUN: llvm-ar crs %t2.a %t1.o %tweak_data_only.o
# RUN: llvm-ar crs %t3.a %t2.o %tstrong_data_only.o

## Bitcode files.
# RUN: llvm-as -o %t1.bc commonblock.ll
# RUN: llvm-as -o %t2.bc blockdata.ll

## Bitcode archive.
# RUN: llvm-ar crs %t4.a %t1.bc %t2.bc

# RUN: ld.lld -o %t1 %tmain.o %t1.a
# RUN: llvm-objdump -D -j .data %t1 | FileCheck --check-prefix=TEST1 %s

# RUN: ld.lld -o %t2 %tmain.o --start-lib %t1.o %tstrong_data_only.o --end-lib
# RUN: llvm-objdump -D -j .data %t2 | FileCheck --check-prefix=TEST1 %s

# RUN: ld.lld -o %t3 %tmain.o %t2.a
# RUN: llvm-objdump -D -j .data %t3 | FileCheck --check-prefix=TEST1 %s

# RUN: ld.lld -o %t4 %tmain.o --start-lib %t1.o %tweak_data_only.o --end-lib
# RUN: llvm-objdump -D -j .data %t4 | FileCheck --check-prefix=TEST1 %s

# RUN: ld.lld -o %t5 %tmain.o %t3.a --print-map | FileCheck --check-prefix=MAP %s

# RUN: ld.lld -o %t6 %tmain.o %t2.o %t1.a
# RUN: llvm-objdump -D -j .data %t6 | FileCheck --check-prefix=TEST2 %s

# RUN: ld.lld -o %t7 %tmain.o %t2.o --start-lib %t1.o %tstrong_data_only.o --end-lib
# RUN: llvm-objdump -D -j .data %t7 | FileCheck --check-prefix=TEST2 %s

# RUN: not ld.lld -o %t8 %tmain.o %t1.a %tstrong_data_only.o 2>&1 | \
# RUN:   FileCheck --check-prefix=ERR %s

# RUN: not ld.lld -o %t9 %tmain.o --start-lib %t1.o %t2.o --end-lib  %tstrong_data_only.o 2>&1 | \
# RUN:   FileCheck --check-prefix=ERR %s

# ERR: ld.lld: error: duplicate symbol: block

# RUN: ld.lld --no-fortran-common -o %t10 %tmain.o %t1.a
# RUN: llvm-readobj --syms %t10 | FileCheck --check-prefix=NFC %s

# RUN: ld.lld --no-fortran-common -o %t11 %tmain.o --start-lib %t1.o %tstrong_data_only.o --end-lib
# RUN: llvm-readobj --syms %t11 | FileCheck --check-prefix=NFC %s

# RUN: ld.lld -o - %tmain.o %t4.a --lto-emit-asm | FileCheck --check-prefix=ASM %s

# RUN: ld.lld -o - %tmain.o  --start-lib %t1.bc %t2.bc --end-lib --lto-emit-asm | \
# RUN:   FileCheck --check-prefix=ASM %s

## Old FORTRAN that mixes use of COMMON blocks and BLOCK DATA requires that we
## search through archives for non-tentative definitions (from the BLOCK DATA)
## to replace the tentative definitions (from the COMMON block(s)).

## Ensure we have used the initialized definition of 'block' instead of a
## common definition.
# TEST1-LABEL:  Disassembly of section .data:
# TEST1:          <block>:
# TEST1-NEXT:       ea 2e 44 54
# TEST1-NEXT:       fb 21 09 40
# TEST1-NEXT:       ...


# NFC:       Name: block
# NFC-NEXT:  Value:
# NFC-NEXT:  Size: 40
# NFC-NEXT:  Binding: Global (0x1)
# NFC-NEXT:  Type: Object (0x1)
# NFC-NEXT:  Other: 0
# NFC-NEXT:  Section: .bss

## Expecting the strong definition from the object file, and the defintions from
## the archive do not interfere.
# TEST2-LABEL: Disassembly of section .data:
# TEST2:         <block>:
# TEST2-NEXT:     03 57 14 8b
# TEST2-NEXT:     0a bf 05 40
# TEST2-NEXT:     ...

# MAP:       28 8 {{.*}}tmp3.a(common-archive-lookup.s.tmp2.o):(.data)
# MAP-NEXT:  28 1 block

# ASM:         .type   block,@object
# ASM:       block:
# ASM-NEXT:    .long 5
# ASM:         .size   block, 20

#--- ref.s
  .text
  .abiversion 2
  .global bar
  .type bar,@function
bar:
  addis 4, 2, block@toc@ha
  addi  4, 4, block@toc@l

## Tentative definition of 'block'.
  .comm block,40,8

#--- refanddef.s
## An alternate strong definition of block, in the same file as
## a different referenced symbol.
  .text
  .abiversion 2
  .global bar
  .type bar,@function
bar:
  addis 4, 2, block@toc@ha
  addi  4, 4, block@toc@l

  .data
  .type block,@object
  .global block
  .p2align 3
block:
  .quad   0x4005bf0a8b145703              # double 2.7182818284589998
  .space  32
  .size   block, 40

#--- def.s
## Strong definition of 'block'.
  .data
  .type block,@object
  .global block
  .p2align 3
block:
  .quad   0x400921fb54442eea              # double 3.1415926535900001
  .space  32
  .size   block, 40

#--- weak.s
## Weak definition of `block`.
  .data
  .type block,@object
  .weak block
  .p2align 3
block:
  .quad   0x400921fb54442eea              # double 3.1415926535900001
  .space  32
  .size   block, 40

#--- main.s
  .global _start
_start:
  bl bar
  blr


#--- blockdata.ll
target datalayout = "e-m:e-i64:64-n32:64-v256:256:256-v512:512:512"
target triple = "powerpc64le-unknown-linux-gnu"

@block = dso_local local_unnamed_addr global [5 x i32] [i32 5, i32 0, i32 0, i32 0, i32 0], align 4

#--- commonblock.ll
target datalayout = "e-m:e-i64:64-n32:64-v256:256:256-v512:512:512"
target triple = "powerpc64le-unknown-linux-gnu"

@block =  common dso_local local_unnamed_addr global [5 x i32] zeroinitializer, align 4

define dso_local i32 @bar(i32 signext %i) local_unnamed_addr {
entry:
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds [5 x i32], [5 x i32]* @block, i64 0, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 8
  ret i32 %0
}
