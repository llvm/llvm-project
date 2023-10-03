# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -pie %t.o --icf=all --print-icf-sections -o /dev/null | \
# RUN:   FileCheck --check-prefixes=EXE %s

# RUN: ld.lld -shared %t.o --icf=all --print-icf-sections -o /dev/null | \
# RUN:   FileCheck --check-prefix=DSO %s

## Definitions are non-preemptible in an executable.
# EXE-NOT:  {{.}}
# EXE:      selected section {{.*}}:(.text.g1)
# EXE-NEXT:   removing identical section {{.*}}:(.text.g2)
# EXE-NEXT:   removing identical section {{.*}}:(.text.g3)
# EXE-NEXT: selected section {{.*}}:(.text.f1)
# EXE-NEXT:   removing identical section {{.*}}:(.text.f2)
# EXE-NEXT: selected section {{.*}}:(.text.h1)
# EXE-NEXT:   removing identical section {{.*}}:(.text.h2)
# EXE-NEXT:   removing identical section {{.*}}:(.text.h3)
# EXE-NOT:  {{.}}

## Definitions are preemptible in a DSO. Only leaf functions can be folded.
# DSO-NOT:  {{.}}
# DSO:      selected section {{.*}}:(.text.g1)
# DSO-NEXT:   removing identical section {{.*}}:(.text.g3)
# DSO-NEXT: selected section {{.*}}:(.text.f1)
# DSO-NEXT:   removing identical section {{.*}}:(.text.f2)
# DSO-NOT:  {{.}}

.globl _start, f1, f2, g1, g2, g3
_start:

.section .text.f1
f1: ret
.section .text.f2
f2: ret

## In -shared mode, .text.g1 and .text.g2 cannot be folded because f1 and f2 are
## preemptible and may refer to different functions at runtime.
.section .text.g1
g1: jmp f1@plt
.section .text.g2
g2: jmp f2@plt
.section .text.g3
g3: jmp f1@plt

## In -shared mode, the sections below cannot be folded because g1, g2 and g3
## are preemptible and may refer to different functions at runtime.
.section .text.h1
jmp g1@plt
.section .text.h2
jmp g2@plt
.section .text.h3
jmp g3@plt
