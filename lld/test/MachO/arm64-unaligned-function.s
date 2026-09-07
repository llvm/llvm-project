# REQUIRES: aarch64, x86
## Like ld64, warn about arm64/arm64_32 function symbols that are not 4-byte
## aligned, since arm64 instructions cannot be executed from an unaligned
## address.

# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos %t/unaligned.s -o %t/unaligned.o
## Run twice with separate prefixes: WARN-NOT would otherwise only apply to
## output after the WARN-DAG matches, not the whole output.
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 11.0 11.0 \
# RUN:   -o /dev/null %t/unaligned.o 2>&1 | tee %t/warnings.txt | \
# RUN:   FileCheck %s --check-prefix=WARN
# RUN: FileCheck %s --check-prefix=NOWARN < %t/warnings.txt
# RUN: llvm-objcopy --redefine-sym lmisaligned_local=Lmisaligned_local \
# RUN:   %t/unaligned.o %t/unaligned-L.o
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 11.0 11.0 \
# RUN:   -o /dev/null %t/unaligned-L.o 2>&1 | FileCheck %s --check-prefix=L
# WARN-DAG: warning: arm64 function not 4-byte aligned: _underaligned_section from {{.*}}unaligned.o
# WARN-DAG: warning: arm64 function not 4-byte aligned: _misaligned_offset from {{.*}}unaligned.o
# WARN-DAG: warning: arm64 function not 4-byte aligned: _misaligned_alt_entry from {{.*}}unaligned.o
# WARN-DAG: warning: arm64 function not 4-byte aligned: lmisaligned_local from {{.*}}unaligned.o
# WARN-DAG: warning: arm64 function not 4-byte aligned: lmisaligned_ext from {{.*}}unaligned.o
# WARN-DAG: warning: arm64 function not 4-byte aligned: ltmp9 from {{.*}}unaligned.o
# WARN-DAG: warning: arm64 function not 4-byte aligned: _section_end from {{.*}}unaligned.o
## _aligned, _underaligned_data, the assembler's non-external anchors (each
## section gets an ltmp<N>; __underaligned's own ltmp1 is itself misaligned
## yet must not be diagnosed), L-prefixed labels, and empty sections are exempt.
# NOWARN-NOT: warning: arm64 function not 4-byte aligned: _aligned
# NOWARN-NOT: warning: arm64 function not 4-byte aligned: _underaligned_data
# NOWARN-NOT: warning: arm64 function not 4-byte aligned: ltmp0
# NOWARN-NOT: warning: arm64 function not 4-byte aligned: ltmp1
# NOWARN-NOT: warning: arm64 function not 4-byte aligned: ltmp2
# NOWARN-NOT: warning: arm64 function not 4-byte aligned: ltmp3
# NOWARN-NOT: warning: arm64 function not 4-byte aligned: Lloh
# NOWARN-NOT: warning: arm64 function not 4-byte aligned: _empty_section
# L-NOT: warning: arm64 function not 4-byte aligned: Lmisaligned_local

## arm64_32 has the same 4-byte instruction alignment requirement.
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos %t/unaligned.s -o %t/unaligned-arm64_32.o
# RUN: %no-arg-lld -dylib -arch arm64_32 -platform_version watchos 4.0 4.0 \
# RUN:   -o /dev/null %t/unaligned-arm64_32.o 2>&1 | FileCheck %s --check-prefix=WARN32
# WARN32: warning: arm64 function not 4-byte aligned: _underaligned_section from {{.*}}unaligned-arm64_32.o

## Aligned functions, data symbols and non-arm64 targets are not diagnosed.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos %t/aligned.s -o %t/aligned.o
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 11.0 11.0 \
# RUN:   -o /dev/null %t/aligned.o 2>&1 | count 0

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/unaligned.s -o %t/unaligned-x86_64.o
# RUN: %no-arg-lld -dylib -arch x86_64 -platform_version macos 11.0 11.0 \
# RUN:   -o /dev/null %t/unaligned-x86_64.o 2>&1 | count 0

#--- unaligned.s
.subsections_via_symbols

## The section itself is only 1-byte aligned.
.section __TEXT,__underaligned,regular,pure_instructions
.globl _underaligned_section
_underaligned_section:
  ret

.text
.p2align 2
.globl _aligned
_aligned:
  ret
  .byte 0

## Aligned section, but the symbol sits at a misaligned offset within it.
.globl _misaligned_offset
_misaligned_offset:
  ret
  .byte 0

## An alt_entry does not start a new subsection, but is still an entry point.
.globl _misaligned_alt_entry
.alt_entry _misaligned_alt_entry
_misaligned_alt_entry:
  ret

## User-defined l-prefixed symbols are diagnosed, whether local or external;
## ld64 only exempts the assembler's own non-external ltmp* anchors, so a
## global ltmp9 is diagnosed while the assembler's ltmp0 is not.
lmisaligned_local:
  ret
.globl lmisaligned_ext
lmisaligned_ext:
  ret
.globl ltmp9
ltmp9:
  ret

## Underaligned data is not a function, so it is not diagnosed.
.section __DATA,__mydata
.globl _underaligned_data
_underaligned_data:
  .byte 1

## An empty section contains no instructions, so a symbol in it names no
## function and is not diagnosed.
.section __TEXT,__empty,regular,pure_instructions
.globl _empty_section
_empty_section:

## A symbol at the very end of a non-empty section is diagnosed.
.text
.p2align 2
  ret
  .byte 0
.globl _section_end
_section_end:

#--- aligned.s
.subsections_via_symbols
.text
.p2align 2
.globl _aligned
_aligned:
  ret
