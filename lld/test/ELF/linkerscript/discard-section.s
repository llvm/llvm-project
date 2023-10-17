# REQUIRES: x86
## Test relocations referencing symbols defined relative to sections discarded by /DISCARD/.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: ld.lld -T a.lds a.o b.o -z undefs -o /dev/null 2>&1 | count 0
# RUN: ld.lld -T a.lds a.o b.o -o /dev/null 2>&1 | count 0
# RUN: ld.lld -r -T a.lds a.o b.o -o a.ro 2>&1 | count 0
# RUN: llvm-readelf -r -s a.ro | FileCheck %s --check-prefix=RELOC

# RELOC:      Relocation section '.rela.bbb' at offset {{.*}} contains 1 entries:
# RELOC-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# RELOC-NEXT: 0000000000000000  0000000000000000 R_X86_64_NONE                             0
# RELOC-EMPTY:
# RELOC-NEXT: Relocation section '.rela.data' at offset {{.*}} contains 4 entries:
# RELOC-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# RELOC-NEXT: 0000000000000000  0000000000000001 R_X86_64_64                               0
# RELOC-NEXT: 0000000000000008  0000000000000001 R_X86_64_64                               0
# RELOC-NEXT: 0000000000000010  0000000000000001 R_X86_64_64                               0
# RELOC-NEXT: 0000000000000018  0000000000000001 R_X86_64_64                               0

# RELOC:      Num:    Value          Size Type    Bind   Vis      Ndx Name
# RELOC-NEXT:   0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
# RELOC-NEXT:   1: 0000000000000000     0 SECTION LOCAL  DEFAULT    1 .text
# RELOC-NEXT:   2: 0000000000000000     0 SECTION LOCAL  DEFAULT    2 .bbb
# RELOC-NEXT:   3: 0000000000000000     0 SECTION LOCAL  DEFAULT    4 .data
# RELOC-NEXT:   4: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT    1 _start
# RELOC-EMPTY:

#--- a.s
.globl _start
_start:

.section .aaa,"a"
.globl global, weakref1
.weak weak, weakref2
global:
weak:
weakref1:
weakref2:
  .quad 0

.section .bbb,"aw"
  .quad .aaa

#--- b.s
.weak weakref1, weakref2
.section .data,"aw"
  .quad global
  .quad weak
  .quad weakref1
  .quad weakref2

#--- a.lds
SECTIONS { /DISCARD/ : { *(.aaa) } }
