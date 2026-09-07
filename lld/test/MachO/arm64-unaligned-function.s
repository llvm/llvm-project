# REQUIRES: aarch64, x86
## Like ld64, warn about arm64 function symbols that are not 4-byte aligned,
## since arm64 instructions cannot be executed from an unaligned address.

# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos %t/unaligned.s -o %t/unaligned.o
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version macos 11.0 11.0 \
# RUN:   -o /dev/null %t/unaligned.o 2>&1 | FileCheck %s --check-prefix=WARN
# WARN-DAG: warning: arm64 function not 4-byte aligned: _underaligned_section from {{.*}}unaligned.o
# WARN-DAG: warning: arm64 function not 4-byte aligned: _misaligned_offset from {{.*}}unaligned.o
# WARN-DAG: warning: arm64 function not 4-byte aligned: _misaligned_alt_entry from {{.*}}unaligned.o
## _aligned and _underaligned_data must not be diagnosed.
# WARN-NOT: warning: arm64 function not 4-byte aligned: _aligned
# WARN-NOT: warning: arm64 function not 4-byte aligned: _underaligned_data

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

## Underaligned data is not a function, so it is not diagnosed.
.section __DATA,__mydata
.globl _underaligned_data
_underaligned_data:
  .byte 1

#--- aligned.s
.subsections_via_symbols
.text
.p2align 2
.globl _aligned
_aligned:
  ret
