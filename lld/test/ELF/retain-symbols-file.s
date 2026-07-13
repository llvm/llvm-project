# REQUIRES: x86
## --retain-symbols-file removes unlisted symbols from .dynsym.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo foo > %t.retain
# RUN: echo bar >> %t.retain

# RUN: ld.lld -shared --retain-symbols-file=%t.retain %t.o -o %t.so
# RUN: llvm-readelf --dyn-syms %t.so | FileCheck %s --check-prefix=DYN

## Separate-argument form.
# RUN: ld.lld -shared --retain-symbols-file %t.retain %t.o -o %t2.so
# RUN: cmp %t.so %t2.so

# DYN:      Symbol table '.dynsym'
# DYN-DAG:  NOTYPE GLOBAL DEFAULT UND und
# DYN-DAG:  FUNC GLOBAL DEFAULT {{.*}} foo
# DYN-DAG:  FUNC GLOBAL DEFAULT {{.*}} bar
# DYN-NOT:  zed
# DYN-NOT:  _start

## --emit-relocs/-r preserves symbols referenced by emitted relocations.
# RUN: ld.lld -shared --emit-relocs --discard-none --retain-symbols-file=%t.retain %t.o -o %t.dn.so
# RUN: llvm-readelf -r %t.dn.so | FileCheck %s --check-prefix=REL
# REL-DAG:  R_X86_64_PLT32 {{.*}} zed - 4
# REL-DAG:  R_X86_64_PLT32 {{.*}} und - 4

## An empty file lists nothing, localizing every defined symbol out of .dynsym.
# RUN: ld.lld -shared --retain-symbols-file=/dev/null %t.o -o %t.empty.so
# RUN: llvm-readelf --dyn-syms %t.empty.so | FileCheck %s --check-prefix=EMPTY
# EMPTY:      Symbol table '.dynsym'
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
