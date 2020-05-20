# REQUIRES: x86
## Test that --emit-relocs keeps local symbols which are used in relocations
## even when --discard-{locals,all} is given.

## There are two separate code paths that mark used local symbols when
## --gc-sections is specified and when is not. The test checks both.

# RUN: llvm-mc -filetype=obj -triple=x86_64 -save-temp-labels %s -o %t.o

# RUN: ld.lld --emit-relocs --discard-locals %t.o -o %tlocal
# RUN: llvm-readelf -s %tlocal | FileCheck --check-prefixes=DISCARD-LOCALS,DISCARD-LOCALS-NOGC %s
# RUN: llvm-readobj -r %tlocal | FileCheck --check-prefix=REL %s

## --gc-sections can discard symbols relative to GCed sections (including STT_SECTION).
# RUN: ld.lld --emit-relocs --discard-locals --gc-sections %t.o -o %tlocal.gc
# RUN: llvm-readelf -s %tlocal.gc | FileCheck --check-prefix=DISCARD-LOCALS %s
# RUN: llvm-readobj -r %tlocal | FileCheck --check-prefix=REL %s

# RUN: ld.lld --emit-relocs --discard-all %t.o -o %tall
# RUN: llvm-readelf -s %tall | FileCheck --check-prefixes=DISCARD-ALL,DISCARD-ALL-NOGC %s
# RUN: llvm-readobj -r %tall | FileCheck --check-prefix=REL %s

# RUN: ld.lld --emit-relocs --discard-all --gc-sections %t.o -o %tall.gc
# RUN: llvm-readelf -s %tall.gc | FileCheck --check-prefix=DISCARD-ALL %s
# RUN: llvm-readobj -r %tall.gc | FileCheck --check-prefix=REL %s

## --discard-locals removes unused local symbols which start with ".L"
# DISCARD-LOCALS:    0: {{0+}} 0 NOTYPE  LOCAL  DEFAULT UND
# DISCARD-LOCALS-NEXT:           NOTYPE  LOCAL  DEFAULT {{.*}} .Lused
# DISCARD-LOCALS-NEXT:           NOTYPE  LOCAL  DEFAULT {{.*}} unused
# DISCARD-LOCALS-NOGC-NEXT:      NOTYPE  LOCAL  DEFAULT {{.*}} unused_gc
# DISCARD-LOCALS-NEXT:           NOTYPE  LOCAL  DEFAULT {{.*}} used
# DISCARD-LOCALS-NEXT:           SECTION LOCAL  DEFAULT {{.*}} .text
# DISCARD-LOCALS-NEXT:           SECTION LOCAL  DEFAULT {{.*}} text
# DISCARD-LOCALS-NOGC-NEXT:      SECTION LOCAL  DEFAULT {{.*}} gc
# DISCARD-LOCALS-NEXT:           SECTION LOCAL  DEFAULT {{.*}} .comment
# DISCARD-LOCALS-NEXT:           NOTYPE  GLOBAL DEFAULT {{.*}} _start

## --discard-all removes all unused regular local symbols.
# DISCARD-ALL:       0: {{0+}} 0 NOTYPE  LOCAL  DEFAULT UND
# DISCARD-ALL-NEXT:              NOTYPE  LOCAL  DEFAULT {{.*}} .Lused
# DISCARD-ALL-NEXT:              NOTYPE  LOCAL  DEFAULT {{.*}} used
# DISCARD-ALL-NEXT:              SECTION LOCAL  DEFAULT {{.*}} .text
# DISCARD-ALL-NEXT:              SECTION LOCAL  DEFAULT {{.*}} text
# DISCARD-ALL-NOGC-NEXT:         SECTION LOCAL  DEFAULT {{.*}} gc
# DISCARD-ALL-NEXT:              SECTION LOCAL  DEFAULT {{.*}} .comment
# DISCARD-ALL-NEXT:              NOTYPE  GLOBAL DEFAULT {{.*}} _start

# REL:      .rela.text {
# REL-NEXT:   R_X86_64_PLT32 text 0xFFFFFFFFFFFFFFFC
# REL-NEXT:   R_X86_64_PLT32 .Lused 0xFFFFFFFFFFFFFFFC
# REL-NEXT:   R_X86_64_PLT32 used 0xFFFFFFFFFFFFFFFC
# REL-NEXT: }

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
