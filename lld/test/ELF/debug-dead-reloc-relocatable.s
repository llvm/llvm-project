# REQUIRES: x86
## Test we resolve symbolic relocations in .debug_* sections to a tombstone
## value if the referenced symbol is discarded when linking a relocatable object.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t1.o
# RUN: ld.lld -r %t1.o %t1.o -o %t2.o
# RUN: llvm-readelf -r -x .debug_ranges %t2.o | FileCheck %s

## Relocations for a discarded section are changed to R_*_NONE.
# CHECK:      Relocation section '.rela.debug_ranges' at offset [[#%#x,]] contains 4 entries:
# CHECK-NEXT:     Offset          Info        Type       Symbol's Value  Symbol's Name + Addend
# CHECK-NEXT: 0000000000000000  [[#%x,]] R_X86_64_64    0000000000000000 .text.foo + 0
# CHECK-NEXT: 0000000000000008  [[#%x,]] R_X86_64_64    0000000000000000 .text.foo + 1
# CHECK-NEXT: 0000000000000020  [[#%x,]] R_X86_64_NONE
# CHECK-NEXT: 0000000000000028  [[#%x,]] R_X86_64_NONE

## References to a discarded section are changed to tombstone values.
# CHECK:      Hex dump of section '.debug_ranges':
# CHECK-NEXT: 0x00000000 00000000 00000000 00000000 00000000
# CHECK-NEXT: 0x00000010 00000000 00000000 00000000 00000000
# CHECK-NEXT: 0x00000020 01000000 00000000 01000000 00000000
# CHECK-NEXT: 0x00000030 00000000 00000000 00000000 00000000

.weak foo

  .section .text.foo,"axG",@progbits,foo,comdat
.Lfoo:
foo:
  ret
.Lfoo_end:

  .section .debug_ranges,"",@progbits
  .quad .Lfoo
  .quad .Lfoo_end
  .quad	0
  .quad	0
