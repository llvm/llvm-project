# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/b.s -o %t/b.o

# RUN: %lld -arch arm64 -e _main -o %t/a.out %t/a.o %t/b.o --bp-compression-sort-section="*" --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=ALL
# RUN: %lld -arch arm64 -e _main -o %t/a.out %t/a.o %t/b.o --bp-compression-sort-section="__DATA*" --bp-compression-sort-section="__TEXT*" --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=ALL
# RUN: %lld -arch arm64 -e _main -o %t/a.out %t/a.o %t/b.o --bp-compression-sort-section="__DATA__custom" --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=DATA

# ALL: Sections for compression: 7
# ALL:   __DATA__custom __DATA__data __TEXT__text

# DATA: Sections for compression: 2
# DATA:   __DATA__custom

#--- a.s
  .text
  .globl _main
_main:
  ret

  .data
data_01:
  .ascii "data_01"
data_02:
  .ascii "data_02"
data_03:
  .ascii "data_03"

  .section __DATA,__custom
custom_06:
  .ascii "custom_06"
custom_07:
  .ascii "custom_07"

  .bss
bss0:
  .zero 10

.subsections_via_symbols

#--- b.s
  .data
data_11:
  .ascii "data_11"

.subsections_via_symbols
