# REQUIRES: x86
## --retain-symbols-file removes unlisted symbols from .dynsym

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

# RUN: ld.lld -shared --gc-sections --retain-symbols-file=retain a.o -o sym.so
# RUN: llvm-readelf --dyn-syms -s sym.so | FileCheck %s --check-prefix=SYM

## Separate-argument form.
# RUN: ld.lld -shared --gc-sections --retain-symbols-file retain a.o -o sym2.so
# RUN: cmp sym.so sym2.so

## .dynsym keeps only the listed retain1 and retain2, plus the undefined und.
# SYM:      Symbol table '.dynsym' contains 4 entries:
# SYM:      GLOBAL DEFAULT UND und
# SYM-NEXT: GLOBAL DEFAULT {{.*}} retain1
# SYM-NEXT: GLOBAL DEFAULT {{.*}} retain2
## .symtab is unaffected.
# SYM:      Symbol table '.symtab' contains 9 entries:

## --emit-relocs keeps symbols referenced by emitted relocations.
# RUN: ld.lld -shared --emit-relocs --retain-symbols-file=retain a.o -o emit-dn.so
# RUN: llvm-readelf -rs emit-dn.so | FileCheck %s --check-prefix=EMIT
# EMIT:      R_X86_64_PLT32 {{.*}} used - 4
# EMIT-NEXT: R_X86_64_PLT32 {{.*}} und - 4
# EMIT:      Symbol table '.symtab' contains 12 entries:
# EMIT:      [[#]] local

## --discard-locals behaves like the default here.
# RUN: ld.lld -shared --emit-relocs --discard-locals --retain-symbols-file=retain a.o -o emit-dl.so
# RUN: llvm-readelf -rs emit-dl.so | FileCheck %s --check-prefix=EMIT

# RUN: ld.lld -shared --emit-relocs --discard-all --retain-symbols-file=retain a.o -o emit-da.so
# RUN: llvm-readelf -rs emit-da.so | FileCheck %s --check-prefix=EMIT-DA
# EMIT-DA:      Symbol table '.symtab' contains 11 entries:
# EMIT-DA-NOT:  local

## nonalloc_referenced is referenced only by the non-alloc .nonalloc section, and
## the emitted relocation keeps a valid symbol index.
# RUN: ld.lld -shared --gc-sections --emit-relocs --discard-none --retain-symbols-file=retain a.o -o na.so
# RUN: llvm-readelf -r na.so | FileCheck %s --check-prefix=NA
# NA: R_X86_64_64 {{.*}} nonalloc_referenced + 0

## An empty file localizes every defined symbol out of .dynsym.
# RUN: ld.lld -shared --retain-symbols-file=/dev/null a.o -o empty.so
# RUN: llvm-readelf --dyn-syms -s empty.so | FileCheck %s --check-prefix=EMPTY
# EMPTY: Symbol table '.dynsym' contains 2 entries:
# EMPTY: GLOBAL DEFAULT UND und
# EMPTY: Symbol table '.symtab' contains 9 entries:

#--- retain
retain1
retain2
#--- a.s
.text
.globl _start
_start:
  call used@PLT
  call und@PLT

.globl retain1, retain2, nonalloc_referenced, used

.type retain1,@function
retain1:
  nop
retain2:
  nop
nonalloc_referenced:
  nop
used:
  retq

.type local,@function
local:
  retq

.section .nonalloc,"",@progbits
.quad nonalloc_referenced
