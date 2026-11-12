# REQUIRES: x86
## --retain-symbols-file filters .symtab, not .dynsym, matching GNU ld.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o

## --gc-sections marks referenced symbols used; unlisted ones are still dropped
## from .symtab, testing that the used flag does not override the filter.
# RUN: ld.lld -shared --gc-sections --retain-symbols-file=retain a.o -o sym.so
# RUN: llvm-readelf --dyn-syms -s sym.so | FileCheck %s --check-prefix=SYM

## Separate-argument form.
# RUN: ld.lld -shared --gc-sections --retain-symbols-file retain a.o -o sym2.so
# RUN: cmp sym.so sym2.so

## .dynsym keeps every exported symbol, including unlisted ones.
# SYM:      Symbol table '.dynsym' contains 7 entries:
# SYM:      GLOBAL DEFAULT UND und
# SYM-NEXT: GLOBAL DEFAULT {{.*}} _start
# SYM-NEXT: GLOBAL DEFAULT {{.*}} used
# SYM-NEXT: GLOBAL DEFAULT {{.*}} retain1
# SYM-NEXT: GLOBAL DEFAULT {{.*}} retain2
# SYM-NEXT: GLOBAL DEFAULT {{.*}} nonalloc_referenced
## .symtab keeps only the listed retain1 and retain2.
# SYM:      Symbol table '.symtab' contains 3 entries:
# SYM:      GLOBAL DEFAULT {{.*}} retain1
# SYM-NEXT: GLOBAL DEFAULT {{.*}} retain2

## --emit-relocs additionally keeps symbols referenced by emitted relocations.
## --discard-locals/--discard-all match the default.
# RUN: ld.lld -shared --emit-relocs --retain-symbols-file=retain a.o -o emit.so
# RUN: llvm-readelf -rs emit.so | FileCheck %s --check-prefix=EMIT
# RUN: ld.lld -shared --emit-relocs --discard-locals --retain-symbols-file=retain a.o -o emit-dl.so
# RUN: llvm-readelf -rs emit-dl.so | FileCheck %s --check-prefix=EMIT
# RUN: ld.lld -shared --emit-relocs --discard-all --retain-symbols-file=retain a.o -o emit-da.so
# RUN: llvm-readelf -rs emit-da.so | FileCheck %s --check-prefix=EMIT
# EMIT:      R_X86_64_PLT32 {{.*}} used - 4
# EMIT-NEXT: R_X86_64_PLT32 {{.*}} und - 4
# EMIT:      Symbol table '.symtab' contains 9 entries:
# EMIT:      GLOBAL DEFAULT {{.*}} used
# EMIT-NEXT: GLOBAL DEFAULT UND und
# EMIT-NEXT: GLOBAL DEFAULT {{.*}} retain1
# EMIT-NEXT: GLOBAL DEFAULT {{.*}} retain2
# EMIT-NEXT: GLOBAL DEFAULT {{.*}} nonalloc_referenced

## --discard-none additionally keeps the unlisted `local`.
# RUN: ld.lld -shared --emit-relocs --discard-none --retain-symbols-file=retain a.o -o emit-dn.so
# RUN: llvm-readelf -rs emit-dn.so | FileCheck %s --check-prefix=EMIT-DN
# EMIT-DN:      R_X86_64_PLT32 {{.*}} used - 4
# EMIT-DN-NEXT: R_X86_64_PLT32 {{.*}} und - 4
# EMIT-DN:      Symbol table '.symtab' contains 10 entries:
# EMIT-DN:      LOCAL DEFAULT {{.*}} local

## nonalloc_referenced is referenced only by the non-alloc .nonalloc section, and
## the emitted relocation keeps a valid symbol index.
# RUN: ld.lld -shared --gc-sections --emit-relocs --discard-none --retain-symbols-file=retain a.o -o na.so
# RUN: llvm-readelf -r na.so | FileCheck %s --check-prefix=NA
# NA: R_X86_64_64 {{.*}} nonalloc_referenced + 0

## An empty file drops every symbol from .symtab; .dynsym is unaffected.
# RUN: ld.lld -shared --retain-symbols-file=/dev/null a.o -o empty.so
# RUN: llvm-readelf --dyn-syms -s empty.so | FileCheck %s --check-prefix=EMPTY
# EMPTY: Symbol table '.dynsym' contains 7 entries:
# EMPTY: GLOBAL DEFAULT UND und
# EMPTY: Symbol table '.symtab' contains 1 entries:

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
