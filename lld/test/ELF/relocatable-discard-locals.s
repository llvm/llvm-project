# REQUIRES: x86
## Test that -r keeps local symbols and overrides --discard-{locals,all}.
## Also see emit-relocs-discard-locals.s

# RUN: llvm-mc -filetype=obj -triple=x86_64 -save-temp-labels %s -o %t.o

# RUN: ld.lld -r --discard-locals %t.o -o %tlocal.ro
# RUN: llvm-readelf -s %tlocal.ro | FileCheck --check-prefixes=SYM,SYM-NOGC %s
# RUN: llvm-readobj -r %tlocal.ro | FileCheck --check-prefix=REL %s

# RUN: ld.lld -r --discard-all %t.o -o %tall.ro
# RUN: llvm-readelf -s %tall.ro | FileCheck --check-prefixes=SYM,SYM-NOGC %s
# RUN: llvm-readobj -r %tall.ro | FileCheck --check-prefix=REL %s

# SYM:           NOTYPE  LOCAL  DEFAULT {{.*}} .Lunused
# SYM-NOGC-NEXT: NOTYPE  LOCAL  DEFAULT {{.*}} .Lunused_gc
# SYM-NEXT:      NOTYPE  LOCAL  DEFAULT {{.*}} .Lused
# SYM-NEXT:      NOTYPE  LOCAL  DEFAULT {{.*}} unused
# SYM-NOGC-NEXT: NOTYPE  LOCAL  DEFAULT {{.*}} unused_gc
# SYM-NEXT:      NOTYPE  LOCAL  DEFAULT {{.*}} used
# SYM-NEXT:      SECTION LOCAL  DEFAULT {{.*}} .text
# SYM-NEXT:      SECTION LOCAL  DEFAULT {{.*}} text
# SYM-NOGC-NEXT: SECTION LOCAL  DEFAULT {{.*}} gc
# SYM-NEXT:      NOTYPE  GLOBAL DEFAULT {{.*}} _start

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
