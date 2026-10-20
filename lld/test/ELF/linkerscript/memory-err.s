# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

## Check bad `ORIGIN`.

# RUN: echo 'MEMORY { ram (rwx) : XYZ = 0x8000 } }' > %t.script
# RUN: not ld.lld -o /dev/null -T %t.script %t.o 2>&1 | FileCheck --check-prefix=ERR1 %s
# ERR1: {{.*}}.script:1: expected one of: ORIGIN, org, or o

## Check bad `LENGTH`.

# RUN: echo 'MEMORY { ram (rwx) : ORIGIN = 0x8000, XYZ = 256K } }' > %t.script
# RUN: not ld.lld -o /dev/null -T %t.script %t.o 2>&1 | FileCheck --check-prefix=ERR2 %s
# ERR2: {{.*}}.script:1: expected one of: LENGTH, len, or l

## Check duplicate regions.

# RUN: echo 'MEMORY { ram (rwx) : o = 8, l = 256K ram (rx) : o = 0, l = 256K }' > %t.script
# RUN: not ld.lld -o /dev/null -T %t.script %t.o 2>&1 | FileCheck --check-prefix=ERR3 %s
# ERR3: {{.*}}.script:1: region 'ram' already defined

## Check no region available.

# RUN: echo 'MEMORY { ram (!rx) : ORIGIN = 0x8000, LENGTH = 256K } \
# RUN: SECTIONS { \
# RUN:   .text : { *(.text) } \
# RUN:   .data : { *(.data) } > ram \
# RUN: }' > %t.script
# RUN: not ld.lld -o /dev/null -T %t.script %t.o 2>&1 | FileCheck --check-prefix=ERR4 %s
# ERR4: error: no memory region specified for section '.text'

## Check undeclared region.

# RUN: echo 'SECTIONS { .text : { *(.text) } > ram }' > %t.script
# RUN: not ld.lld -o /dev/null -T %t.script %t.o 2>&1 | FileCheck --check-prefix=ERR5 %s
# ERR5: error: memory region 'ram' not declared

# RUN: echo 'SECTIONS { .text : { *(.text) } AT> ram }' > %t.script
# RUN: not ld.lld -o /dev/null -T %t.script %t.o 2>&1 | FileCheck --check-prefix=ERR5 %s

## Check region overflow.

# RUN: echo 'MEMORY { ram (rwx) : ORIGIN = 0, LENGTH = 2K } \
# RUN: SECTIONS { \
# RUN:   .text : { *(.text) } > ram \
# RUN:   .data : { *(.data) } > ram \
# RUN: }' > %t.script
# RUN: not ld.lld -o /dev/null -T %t.script %t.o 2>&1 | FileCheck --check-prefix=ERR6 %s
# ERR6: error: section '.data' will not fit in region 'ram': overflowed by 2049 bytes

## Check invalid region attributes.

# RUN: echo "MEMORY { ram (abc) : ORIGIN = 8000, LENGTH = 256K } }" > %t.script
# RUN: not ld.lld -o /dev/null -T %t.script %t.o 2>&1 | FileCheck --check-prefix=ERR7 %s
# ERR7: {{.*}}.script:1: invalid memory region attribute

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: echo 'MEMORY { name : ORIGIN = ., LENGTH = 1 }' > %t.script
# RUN: not ld.lld -shared -o /dev/null -T %t.script %t.o 2>&1 | FileCheck --check-prefix=ERR_DOT %s
# ERR_DOT: error: {{.*}}.script:1: unable to get location counter value

## ORIGIN/LENGTH can be simple symbolic expressions. If the expression
## requires interaction with memory regions, it may fail.

# RUN: echo 'MEMORY { ram : ORIGIN = symbol, LENGTH = 4094 } \
# RUN: SECTIONS { \
# RUN:   .text : { *(.text) } > ram \
# RUN:   symbol = .; \
# RUN:   .data : { *(.data) } > ram \
# RUN: }' > %t.script
# RUN: not ld.lld -T %t.script %t.o -o /dev/null 2>&1 | FileCheck --check-prefix=NOT_CONVERGE %s
# NOT_CONVERGE: error: address (0x14) of section '.text' does not converge

# RUN: echo 'MEMORY { ram : ORIGIN = symbol, LENGTH = 4094 ' > %t.script
# RUN: not ld.lld -T %t.script %t.o -o /dev/null 2>&1 | FileCheck --check-prefix=UNCLOSED %s
# UNCLOSED: error: {{.*}}:1: unexpected EOF

nop

.data
.zero 4096
