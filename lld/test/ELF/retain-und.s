# REQUIRES: x86
## An unlisted undefined symbol referenced by a dynamic relocation stays in
## .dynsym (keeping the relocation valid) but is dropped from .symtab.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo > %t.retain
# RUN: ld.lld -shared --retain-symbols-file=%t.retain %t.o -o %t.so
# RUN: llvm-readelf -r --dyn-syms --symbols %t.so | FileCheck %s

# CHECK:      Relocation section '.rela.dyn' {{.+}} contains 1
# CHECK:      R_X86_64_64 {{.*}} foo + 0
# CHECK:      Symbol table '.dynsym'
# CHECK:      NOTYPE WEAK DEFAULT UND foo
# CHECK:      Symbol table '.symtab'
# CHECK-NOT:  foo

.data
.quad foo
.weak foo
