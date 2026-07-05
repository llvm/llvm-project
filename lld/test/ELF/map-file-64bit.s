# REQUIRES: x86
## Test how we display the upper half of the address space, which is commonly
## used by operating system kernels.

# RUN: llvm-mc -filetype=obj -triple=x86_64 /dev/null -o %t.o
# RUN: ld.lld -M -T %s %t.o -o /dev/null | FileCheck %s --match-full-lines --strict-whitespace

## . = 0xffffffff80000000; has a misaligned Size column.
#      CHECK:             VMA              LMA     Size Align Out     In      Symbol
# CHECK-NEXT:               0                0 ffffffff80000000     1 . = 0xffffffff80000000
# CHECK-NEXT:ffffffff80000000                0   100000     1 . += 0x100000
# CHECK-NEXT:ffffffff80100000                0        0     1 _text = .
# CHECK-NEXT:ffffffff80100000 ffffffff80100000        0     4 .text

SECTIONS {
  . = 0xffffffff80000000;
  . += 0x100000;
  _text = .;
}
