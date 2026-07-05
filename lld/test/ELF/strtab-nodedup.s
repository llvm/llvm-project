# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/b.s -o %t/b.o

## Non-empty local symbol names are not deduplicated. This helps parallel
## .symtab write. We used to perform deduplication at -O2.
# RUN: ld.lld %t/a.o %t/b.o -o %t/a
# RUN: llvm-readelf -p .strtab %t/a | FileCheck %s --check-prefix=NODEDUP
# RUN: ld.lld -r -O2 %t/a.o %t/b.o -o %t/a.ro
# RUN: llvm-readelf -p .strtab %t/a.ro | FileCheck %s --check-prefix=NODEDUP

# NODEDUP:        [     1]  local
# NODEDUP-NEXT:   [     7]  local
# NODEDUP-NEXT:   [     d]  foo
# NODEDUP-EMPTY:

# RUN: llvm-readelf -s %t/a.ro | FileCheck %s --check-prefix=SYMTAB

# SYMTAB:    0: {{0+}} 0 NOTYPE  LOCAL  DEFAULT UND
# SYMTAB-NEXT:           NOTYPE  LOCAL  DEFAULT [[#]] local
# SYMTAB-NEXT:           SECTION LOCAL  DEFAULT [[#]] .text
# SYMTAB-NEXT:           NOTYPE  LOCAL  DEFAULT [[#]] local
# SYMTAB-NEXT:           NOTYPE  GLOBAL DEFAULT [[#]] foo

#--- a.s
.global foo
foo:
local:
  ret

#--- b.s
.weak foo
foo:
local:
  ret
