# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: ld.lld --static %t.o -o %t
# RUN: llvm-readelf -S -s -d %t | FileCheck %s

## Verify that R_AARCH64_AUTH_RELATIVE relocations are included within the
## bounds of __rela_iplt_start/end, as relative relocations still emitted for
## static PDEs due to needing run-time signing. Historically this would not be
## the case if added to .rela.dyn with sharding.

# CHECK: .rela.dyn         RELA            0000000000200158 000158 000018 18   A  0   0  8
# CHECK: 0000000000200158     0 NOTYPE  LOCAL  HIDDEN      1 __rela_iplt_start
# CHECK: 0000000000200170     0 NOTYPE  LOCAL  HIDDEN      1 __rela_iplt_end

adrp x0, __rela_iplt_start
adrp x0, __rela_iplt_end

.data
foo:
.quad foo@AUTH(da,42)
