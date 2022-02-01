# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/b.s -o %t/b.o

## By default local symbol names are not deduplicated.
# RUN: ld.lld %t/a.o %t/b.o -o %t/a
# RUN: llvm-readelf -p .strtab %t/a | FileCheck %s --check-prefix=NODEDUP

# NODEDUP:        [     1]  local
# NODEDUP-NEXT:   [     7]  local
# NODEDUP-NEXT:   [     d]  foo
# NODEDUP-EMPTY:

## -O2 deduplicates local symbol names.
# RUN: ld.lld -O2 %t/a.o %t/b.o -o %t/a
# RUN: llvm-readelf -p .strtab %t/a | FileCheck %s --check-prefix=DEDUP

# DEDUP:        [     1]  local
# DEDUP-NEXT:   [     7]  foo
# DEDUP-EMPTY:

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
