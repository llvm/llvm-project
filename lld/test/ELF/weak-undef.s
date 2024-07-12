# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o %t --export-dynamic
# RUN: llvm-readelf -r --dyn-syms --hex-dump=.data %t | \
# RUN:   FileCheck %s --check-prefixes=NORELOC,COMMON

# NORELOC: There are no relocations in this file.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/dummy-shared.s -o %t1.o
# RUN: ld.lld %t1.o -shared -o %t1.so
# RUN: ld.lld %t.o -o %t %t1.so -pie
# RUN: llvm-readelf -r --dyn-syms --hex-dump=.data %t | \
# RUN:   FileCheck %s --check-prefixes=RELOC,COMMON

# RELOC:      Relocation section '.rela.dyn' at offset {{.*}} contains 1 entries:
# RELOC-NEXT: Offset Info Type Symbol's Value Symbol's Name + Addend
# RELOC-NEXT: {{.*}} 0000000100000001 R_X86_64_64 0000000000000000 foo + 0

# NORELOC-NOT: Symbol table '.dynsym'
# RELOC:       Symbol table '.dynsym' contains 2 entries:
# RELOC-NEXT:  Num: Value Size Type Bind Vis Ndx Name
# RELOC-NEXT:  0: 0000000000000000 0 NOTYPE LOCAL DEFAULT UND
# RELOC-NEXT:  1: 0000000000000000 0 NOTYPE WEAK DEFAULT UND foo
# COMMON:      Hex dump of section '.data':
# COMMON-NEXT: {{.*}} 00000000 00000000 
# COMMON-EMPTY:

.weak foo

.data
  .dc.a foo
