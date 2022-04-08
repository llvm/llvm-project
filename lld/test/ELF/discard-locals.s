# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 -save-temp-labels %s -o %t.o

# RUN: ld.lld --discard-locals %t.o -o %tlocal
# RUN: llvm-readelf -s %tlocal | FileCheck --check-prefixes=DISCARD-LOCALS,DISCARD-LOCALS-NOGC %s

## --gc-sections can discard symbols relative to GCed sections (including STT_SECTION).
# RUN: ld.lld --discard-locals --gc-sections %t.o -o %tlocal.gc
# RUN: llvm-readelf -s %tlocal.gc | FileCheck --check-prefixes=DISCARD-LOCALS,DISCARD-LOCALS-GC %s

# RUN: ld.lld --discard-all %t.o -o %tall
# RUN: llvm-readelf -s %tall | FileCheck --check-prefix=DISCARD-ALL %s

# RUN: ld.lld --discard-all --gc-sections %t.o -o %tall.gc
# RUN: llvm-readelf -s %tall.gc | FileCheck --check-prefixes=DISCARD-ALL,DISCARD-ALL-GC %s

## --discard-locals removes local symbols which start with ".L"
# DISCARD-LOCALS:    0: {{0+}} 0 NOTYPE  LOCAL  DEFAULT UND
# DISCARD-LOCALS-GC-NEXT:        NOTYPE  LOCAL  DEFAULT [[#]] .Lused
# DISCARD-LOCALS-NEXT:           NOTYPE  LOCAL  DEFAULT [[#]] used
# DISCARD-LOCALS-NEXT:           NOTYPE  LOCAL  DEFAULT [[#]] unused
# DISCARD-LOCALS-NOGC-NEXT:      NOTYPE  LOCAL  DEFAULT [[#]] unused_gc
# DISCARD-LOCALS-NEXT:           NOTYPE  GLOBAL DEFAULT [[#]] _start

## --discard-all removes all regular local symbols.
# DISCARD-ALL:       0: {{0+}} 0 NOTYPE  LOCAL  DEFAULT UND
# DISCARD-ALL-GC-NEXT:           NOTYPE  LOCAL  DEFAULT [[#]] .Lused
# DISCARD-ALL-GC-NEXT:           NOTYPE  LOCAL  DEFAULT [[#]] used
# DISCARD-ALL-NEXT:              NOTYPE  GLOBAL DEFAULT [[#]] _start

.globl _start
_start:
  call text@plt
  jmp .Lused@plt
  call used@plt

.section text,"ax"
.Lunused:
.Lused:
unused:
used:

.section gc,"ax"
.Lunused_gc:
unused_gc:
  ret
