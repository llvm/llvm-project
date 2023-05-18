# REQUIRES: aarch64

## Check that LLD does not crash if it encounters DWARF sections
## without __debug_info compile unit DIEs being present.

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 %t.o -o /dev/null

.text
.globl _main
_main:
  ret

.section  __DWARF,__debug_abbrev,regular,debug
  .byte 0
