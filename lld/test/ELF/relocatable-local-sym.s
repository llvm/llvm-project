# REQUIRES: x86
## Test relocations referencing non-STT_SECTION local symbols in SHF_ALLOC and non-SHF_ALLOC sections for -r.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -r -o %t %t.o %t.o
# RUN: llvm-readelf -r -x .nonalloc %t | FileCheck --check-prefix=RELA %s

# RUN: llvm-mc -filetype=obj -triple=i686 --defsym X86_32=1 %s -o %t1.o
# RUN: ld.lld -r -o %t1 %t1.o %t1.o
# RUN: llvm-readelf -r -x .nonalloc %t1 | FileCheck --check-prefix=REL %s

# RELA:       Relocation section '.rela.data' at offset {{.*}} contains 2 entries:
# RELA:         Offset          Info         Type      Symbol's Value  Symbol's Name + Addend
# RELA-NEXT:  0000000000000000  {{.*}} R_X86_64_32   0000000000000000 ifunc + 9
# RELA-NEXT:  0000000000000004  {{.*}} R_X86_64_32   0000000000000004 ifunc + 9
# RELA:       Relocation section '.rela.nonalloc' at offset {{.*}} contains 2 entries:
# RELA:         Offset          Info         Type      Symbol's Value  Symbol's Name + Addend
# RELA-NEXT:  0000000000000000  {{.*}} R_X86_64_32   0000000000000000 ifunc + 9
# RELA-NEXT:  0000000000000004  {{.*}} R_X86_64_32   0000000000000004 ifunc + 9
# RELA:       Hex dump of section '.nonalloc':
# RELA-NEXT:  0x00000000 00000000 00000000                   ........

# REL:         Offset   Info   Type         Sym. Value  Symbol's Name
# REL-NEXT:  00000000  {{.*}} R_386_32        00000000   ifunc
# REL-NEXT:  00000004  {{.*}} R_386_32        00000004   ifunc
# REL-EMPTY:
# REL:         Offset   Info   Type         Sym. Value  Symbol's Name
# REL-NEXT:  00000000  {{.*}} R_386_32        00000000   ifunc
# REL-NEXT:  00000004  {{.*}} R_386_32        00000004   ifunc
# REL:       Hex dump of section '.nonalloc':
# REL-NEXT:  0x00000000 09000000 09000000                   ........

resolver: ret
.type ifunc, @gnu_indirect_function
.set ifunc, resolver

.data
.long ifunc+9

.section .nonalloc
## The relocation references ifunc instead of the STT_SECTION symbol.
.long ifunc+9
