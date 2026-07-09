# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/b.s -o %t/b.o

## Wildcard glob: all sections go to a single group
# RUN: %lld -arch arm64 -e _main -o %t/a.out %t/a.o %t/b.o \
# RUN:   --bp-compression-sort-section="*" \
# RUN:   --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=WILDCARD

# WILDCARD: Sections for compression: 7
# WILDCARD: Compression groups: 1
# WILDCARD:   *: 7 sections

## Two globs: sections are grouped by the winning glob
# RUN: %lld -arch arm64 -e _main -o %t/a.out %t/a.o %t/b.o \
# RUN:   --bp-compression-sort-section="__DATA*" \
# RUN:   --bp-compression-sort-section="__TEXT*" \
# RUN:   --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=TWO-GLOBS

## Deprecated --bp-compression-sort=both still works
# RUN: %lld -arch arm64 -e _main -o %t/a.out %t/a.o %t/b.o \
# RUN:   --bp-compression-sort=both \
# RUN:   --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=LEGACY-BOTH

## Deprecated function/data modes still use the legacy buckets.
# RUN: %lld -arch arm64 -e _main -o %t/a.out %t/a.o %t/b.o \
# RUN:   --bp-compression-sort=function \
# RUN:   --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=LEGACY-FUNCTION
# RUN: %lld -arch arm64 -e _main -o %t/a.out %t/a.o %t/b.o \
# RUN:   --bp-compression-sort=data \
# RUN:   --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=LEGACY-DATA

# TWO-GLOBS: Sections for compression: 7
# TWO-GLOBS: Compression groups: 2
# TWO-GLOBS:   __DATA*: 6 sections
# TWO-GLOBS:   __TEXT*: 1 sections

# LEGACY-BOTH: Sections for compression: 7
# LEGACY-BOTH: Compression groups: 2
# LEGACY-BOTH:   legacy:function: 1 sections
# LEGACY-BOTH:   legacy:data: 6 sections

# LEGACY-FUNCTION: Sections for compression: 1
# LEGACY-FUNCTION: Compression groups: 1
# LEGACY-FUNCTION:   legacy:function: 1 sections

# LEGACY-DATA: Sections for compression: 6
# LEGACY-DATA: Compression groups: 1
# LEGACY-DATA:   legacy:data: 6 sections

## Single glob matching only TEXT
# RUN: %lld -arch arm64 -e _main -o %t/a.out %t/a.o %t/b.o \
# RUN:   --bp-compression-sort-section="__TEXT*" \
# RUN:   --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=TEXT

# TEXT: Sections for compression: 1
# TEXT: Compression groups: 1
# TEXT:   __TEXT*: 1 sections

## Exact section name glob
# RUN: %lld -arch arm64 -e _main -o %t/a.out %t/a.o %t/b.o \
# RUN:   --bp-compression-sort-section="__DATA__custom" \
# RUN:   --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=DATA

# DATA: Sections for compression: 2
# DATA: Compression groups: 1
# DATA:   __DATA__custom: 2 sections

## Match priority: explicit match_priority wins
# RUN: %lld -arch arm64 -e _main -o %t/match.out %t/a.o %t/b.o \
# RUN:   --bp-compression-sort-section="__DATA*" \
# RUN:   --bp-compression-sort-section="__DATA__custom=0=1" \
# RUN:   --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=MATCH

# MATCH: Compression groups: 2
# MATCH:   __DATA*: 4 sections
# MATCH:   __DATA__custom: 2 sections

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
