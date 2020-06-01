# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: ld.lld -pie --pack-dyn-relocs=relr -z max-page-size=4096 --verbose %t.o -o %t 2>&1 | FileCheck %s
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=RELR %s

## This test makes sure we don't shrink .relr.dyn, otherwise its size may
## oscillate between 2 words and 3 words.

## The test is very sensitive to the exact section sizes and offsets,
## make sure .data is located at a page boundary.

# CHECK: .relr.dyn needs 1 padding word(s)

# RELR:      .relr.dyn {
# RELR-NEXT:   0x2F30 R_AARCH64_RELATIVE - 0x0
# RELR-NEXT:   0x2F38 R_AARCH64_RELATIVE - 0x0
# RELR-NEXT:   0x3000 R_AARCH64_RELATIVE - 0x0
# RELR-NEXT: }

.section .data.rel.ro
.align 3
.space 0xcd0
foo:
## Encoded by the first word of .relr.dyn
.quad foo

## Encoded by the second word of .relr.dyn
.quad foo

.section .data
.align 3
## If .data is at 0x3000, the relocation will be encoded by the second word.
## If we shrink .relr.dyn, the end of .dynamic will be at 0x2ff8 and .data
## will be at 0x3ff8, we will need the third word to encode this relocation,
## which will cause .relr.dyn to expand again.
.quad foo
