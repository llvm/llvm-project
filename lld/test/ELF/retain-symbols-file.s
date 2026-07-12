# REQUIRES: x86
## --retain-symbols-file filters .symtab, not .dynsym, matching GNU ld.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo foo > %t.retain
# RUN: echo bar >> %t.retain

# RUN: ld.lld -shared --retain-symbols-file=%t.retain %t.o -o %t.so
# RUN: llvm-readelf --dyn-syms -s %t.so | FileCheck %s --check-prefix=SYM

## Separate-argument form.
# RUN: ld.lld -shared --retain-symbols-file %t.retain %t.o -o %t2.so
# RUN: cmp %t.so %t2.so

## .dynsym keeps every exported symbol, including unlisted ones.
# SYM-DAG: NOTYPE GLOBAL DEFAULT UND und
# SYM-DAG: NOTYPE GLOBAL DEFAULT {{.*}} _start
# SYM-DAG: FUNC GLOBAL DEFAULT {{.*}} zed
# SYM-DAG: FUNC GLOBAL DEFAULT {{.*}} foo
# SYM-DAG: FUNC GLOBAL DEFAULT {{.*}} bar

## .symtab keeps only the listed foo and bar.
# SYM:      Symbol table '.symtab'
# SYM-NOT:  _start
# SYM-NOT:  zed
# SYM-NOT:  und
# SYM-NOT:  loc
# SYM:      FUNC GLOBAL DEFAULT {{.*}} foo
# SYM:      FUNC GLOBAL DEFAULT {{.*}} bar

## --emit-relocs keeps unlisted symbols referenced by emitted relocations.
# RUN: ld.lld -shared --emit-relocs --retain-symbols-file=%t.retain %t.o -o %t.er.so
# RUN: llvm-readelf --symbols %t.er.so | FileCheck %s --check-prefix=EMIT
# RUN: llvm-readelf -r %t.er.so | FileCheck %s --check-prefix=REL
# EMIT:      Symbol table '.symtab'
# EMIT-DAG:  FUNC GLOBAL DEFAULT {{.*}} zed
# EMIT-DAG:  NOTYPE GLOBAL DEFAULT UND und
# EMIT-DAG:  FUNC GLOBAL DEFAULT {{.*}} foo
# EMIT-DAG:  FUNC GLOBAL DEFAULT {{.*}} bar
# REL-DAG:   R_X86_64_PLT32 {{.*}} zed - 4
# REL-DAG:   R_X86_64_PLT32 {{.*}} und - 4

## --discard-none: relocation-referenced symbols are kept by --retain-symbols-file
## itself, not --discard-*.
# RUN: ld.lld -shared --emit-relocs --discard-none --retain-symbols-file=%t.retain %t.o -o %t.dn.so
# RUN: llvm-readelf -r %t.dn.so | FileCheck %s --check-prefix=REL

## An empty file drops every symbol from .symtab, unlike omitting the option.
# RUN: ld.lld -shared --retain-symbols-file=/dev/null %t.o -o %t.empty.so
# RUN: llvm-readelf --dyn-syms --symbols %t.empty.so | FileCheck %s --check-prefix=EMPTY
# EMPTY:      Symbol table '.dynsym'
# EMPTY-DAG:  FUNC GLOBAL DEFAULT {{.*}} foo
# EMPTY-DAG:  FUNC GLOBAL DEFAULT {{.*}} bar
# EMPTY-DAG:  FUNC GLOBAL DEFAULT {{.*}} zed
# EMPTY:      Symbol table '.symtab'
# EMPTY-NOT:  foo
# EMPTY-NOT:  bar
# EMPTY-NOT:  zed

.text
.globl _start
_start:
  call zed@PLT
  call und@PLT

.globl foo
.type foo,@function
foo:
  retq

.globl bar
.type bar,@function
bar:
  retq

.globl zed
.type zed,@function
zed:
  retq

.type loc,@function
loc:
  retq
