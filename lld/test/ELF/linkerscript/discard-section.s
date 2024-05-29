# REQUIRES: x86
## Test relocations referencing symbols defined relative to sections discarded by /DISCARD/.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: not ld.lld --threads=1 -T a.lds a.o b.o -z undefs -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOCAL --implicit-check-not=error:
# RUN: not ld.lld --threads=1 -T a.lds a.o b.o -o /dev/null 2>&1 | FileCheck %s --check-prefixes=LOCAL,NONLOCAL --implicit-check-not=error:
# RUN: ld.lld -r -T a.lds a.o b.o -o a.ro 2>&1 | FileCheck %s --check-prefix=WARNING --implicit-check-not=warning:
# RUN: llvm-readelf -r -s a.ro | FileCheck %s --check-prefix=RELOC

# RUN: ld.lld -r --gc-sections -T a.lds a.o b.o -o a.gc.ro --no-fatal-warnings
# RUN: llvm-readelf -r -s a.gc.ro | FileCheck %s --check-prefix=RELOC-GC

# LOCAL:      error: relocation refers to a discarded section: .aaa
# LOCAL-NEXT: >>> defined in a.o
# LOCAL-NEXT: >>> referenced by a.o:(.bbb+0x0)

# NONLOCAL:      error: relocation refers to a symbol in a discarded section: global
# NONLOCAL-NEXT: >>> defined in a.o
# NONLOCAL-NEXT: >>> referenced by b.o:(.data+0x0)

# NONLOCAL:      error: relocation refers to a symbol in a discarded section: weak
# NONLOCAL-NEXT: >>> defined in a.o
# NONLOCAL-NEXT: >>> referenced by b.o:(.data+0x8)

# NONLOCAL:      error: relocation refers to a symbol in a discarded section: weakref1
# NONLOCAL-NEXT: >>> defined in a.o
# NONLOCAL-NEXT: >>> referenced by b.o:(.data+0x10)

# NONLOCAL:      error: relocation refers to a symbol in a discarded section: weakref2
# NONLOCAL-NEXT: >>> defined in a.o
# NONLOCAL-NEXT: >>> referenced by b.o:(.data+0x18)

# WARNING:      warning: relocation refers to a discarded section: .aaa
# WARNING-NEXT: >>> referenced by a.o:(.rela.bbb+0x0)

## GNU ld reports "defined in discarded secion" errors even in -r mode.
## We set the symbol index to 0.
# RELOC:      Relocation section '.rela.bbb' at offset {{.*}} contains 1 entries:
# RELOC-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# RELOC-NEXT: 0000000000000000  0000000000000000 R_X86_64_NONE                             0
# RELOC-EMPTY:
# RELOC-NEXT: Relocation section '.rela.data' at offset {{.*}} contains 4 entries:
# RELOC-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# RELOC-NEXT: 0000000000000000  0000000000000001 R_X86_64_64                             0
# RELOC-NEXT: 0000000000000008  0000000000000001 R_X86_64_64                             0
# RELOC-NEXT: 0000000000000010  0000000000000001 R_X86_64_64                             0
# RELOC-NEXT: 0000000000000018  0000000000000001 R_X86_64_64                             0

# RELOC:      Num:    Value          Size Type    Bind   Vis      Ndx Name
# RELOC-NEXT:   0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
# RELOC-NEXT:   1: 0000000000000000     0 SECTION LOCAL  DEFAULT    1 .text
# RELOC-NEXT:   2: 0000000000000000     0 SECTION LOCAL  DEFAULT    2 .bbb
# RELOC-NEXT:   3: 0000000000000000     0 SECTION LOCAL  DEFAULT    4 .data
# RELOC-NEXT:   4: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT    1 _start
# RELOC-EMPTY:

# RELOC-GC:   There are no relocations in this file.

#--- a.s
.globl _start
_start:

.section .aaa,"a"
.globl global, weakref1, unused
.weak weak, weakref2
global:
weak:
weakref1:
weakref2:
## Eliminate `unused` just like GC discarded definitions.
## Linux kernel's CONFIG_DEBUG_FORCE_WEAK_PER_CPU=y configuration expects
## that the unreferenced `unused` is not emitted to .symtab.
unused:
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
