# REQUIRES: sparc
# RUN: llvm-mc -filetype=obj -triple=sparcv9 %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readelf -r %t.so | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=sparcv9 --defsym ERR=1 %s -o %t.err.o
# RUN: not ld.lld -shared %t.err.o -o /dev/null 2>&1 | FileCheck --check-prefix=ERR %s

# CHECK:      R_SPARC_RELATIVE
# CHECK:      R_SPARC_16 {{.*}} external
# CHECK:      R_SPARC_32 {{.*}} external
# CHECK:      R_SPARC_64 {{.*}} external
# CHECK:      R_SPARC_UA16 {{.*}} external
# CHECK:      R_SPARC_UA32 {{.*}} external
# CHECK:      R_SPARC_UA64 {{.*}} external

# ERR-DAG: error: relocation R_SPARC_16 at offset {{[0-9]+}} against non-preemptible symbol local cannot be converted to R_SPARC_RELATIVE
# ERR-DAG: error: relocation R_SPARC_32 at offset {{[0-9]+}} against non-preemptible symbol local cannot be converted to R_SPARC_RELATIVE
# ERR-DAG: error: relocation R_SPARC_UA64 at offset {{[0-9]+}} against non-preemptible symbol local cannot be converted to R_SPARC_RELATIVE

.data
.p2align 3
aligned_local64:
  .xword 0
  .reloc aligned_local64, R_SPARC_UA64, local

.p2align 1
aligned_external16:
  .half 0
  .reloc aligned_external16, R_SPARC_UA16, external

.p2align 2
aligned_external32:
  .word 0
  .reloc aligned_external32, R_SPARC_UA32, external

.p2align 3
aligned_external64:
  .xword 0
  .reloc aligned_external64, R_SPARC_UA64, external

.p2align 1
  .byte 0
unaligned_external16:
  .half 0
  .reloc unaligned_external16, R_SPARC_16, external

.p2align 2
  .byte 0
unaligned_external32:
  .word 0
  .reloc unaligned_external32, R_SPARC_32, external

.p2align 3
  .byte 0
unaligned_external64:
  .xword 0
  .reloc unaligned_external64, R_SPARC_64, external

.ifdef ERR
.p2align 1
aligned_local16:
  .half 0
  .reloc aligned_local16, R_SPARC_UA16, local

.p2align 2
aligned_local32:
  .word 0
  .reloc aligned_local32, R_SPARC_UA32, local

.p2align 3
  .byte 0
unaligned_local64:
  .xword 0
  .reloc unaligned_local64, R_SPARC_64, local
.endif

.hidden local
local:
  .xword 0
