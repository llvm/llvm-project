# REQUIRES: x86
## Test relocations referencing symbols defined relative to sections discarded by /DISCARD/.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: not ld.lld --threads=1 -T a.lds a.o b.o -z undefs -o /dev/null 2>&1 | FileCheck %s --check-prefix=SECTION --implicit-check-not=error:
# RUN: not ld.lld --threads=1 -T a.lds a.o b.o -o /dev/null 2>&1 | FileCheck %s --check-prefixes=SECTION,SYMBOL --implicit-check-not=error:
# RUN: ld.lld -r -T a.lds a.o b.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARNING --implicit-check-not=error:

# SECTION: error: relocation refers to a discarded section: .aaa
# SECTION-NEXT: >>> defined in a.o
# SECTION-NEXT: >>> referenced by a.o:(.bbb+0x0)

# SYMBOL: error: relocation refers to a symbol in a discarded section: global
# SYMBOL-NEXT: >>> defined in a.o
# SYMBOL-NEXT: >>> referenced by b.o:(.data+0x0)

# SYMBOL: error: relocation refers to a symbol in a discarded section: weak
# SYMBOL-NEXT: >>> defined in a.o
# SYMBOL-NEXT: >>> referenced by b.o:(.data+0x8)

# SYMBOL: error: relocation refers to a symbol in a discarded section: weakref1
# SYMBOL-NEXT: >>> defined in a.o
# SYMBOL-NEXT: >>> referenced by b.o:(.data+0x10)

# SYMBOL: error: relocation refers to a symbol in a discarded section: weakref2
# SYMBOL-NEXT: >>> defined in a.o
# SYMBOL-NEXT: >>> referenced by b.o:(.data+0x18)

# WARNING: warning: relocation refers to a discarded section: .aaa
# WARNING-NEXT: >>> referenced by a.o:(.rela.bbb+0x0)

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
