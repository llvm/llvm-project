# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { .text : { *(.text.1) } }" > %t.script

## Check we do not report orphans by default even with -verbose.
# RUN: ld.lld -shared -o %t.out --script %t.script %t.o 2>&1 -verbose \
# RUN:   | FileCheck %s --check-prefix=DEFAULT
# DEFAULT-NOT: placed

## Check --orphan-handling=place has the same behavior as default.
# RUN: ld.lld -shared --orphan-handling=place -o %t.out --script %t.script \
# RUN:   %t.o 2>&1 -verbose  -error-limit=0 | FileCheck %s --check-prefix=DEFAULT

## Check --orphan-handling=error reports errors about orphans.
# RUN: not ld.lld --orphan-handling=error -o /dev/null -T %t.script \
# RUN:   %t.o 2>&1 | FileCheck %s --check-prefixes=COMMON,SYMTAB

## --strip-all discards .strtab and .symtab sections. Don't error about them.
# RUN: not ld.lld --orphan-handling=error --strip-all -o /dev/null -T %t.script \
# RUN:   %t.o 2>&1 | FileCheck %s --check-prefix=COMMON

## -shared enables some .dynsym related sections.
# RUN: not ld.lld -shared --orphan-handling=error -o /dev/null -T %t.script \
# RUN:   %t.o 2>&1 | FileCheck %s --check-prefixes=COMMON,DYNSYM,SYMTAB

# COMMON:      {{.*}}.o:(.text) is being placed in '.text'
# COMMON-NEXT: {{.*}}.o:(.text.2) is being placed in '.text.2'
# COMMON-NEXT: <internal>:(.comment) is being placed in '.comment'
# DYNSYM-NEXT: <internal>:(.dynsym) is being placed in '.dynsym'
# DYNSYM-NEXT: <internal>:(.gnu.hash) is being placed in '.gnu.hash'
# DYNSYM-NEXT: <internal>:(.hash) is being placed in '.hash'
# DYNSYM-NEXT: <internal>:(.dynamic) is being placed in '.dynamic'
# DYNSYM-NEXT: <internal>:(.dynstr) is being placed in '.dynstr'
# SYMTAB-NEXT: <internal>:(.symtab) is being placed in '.symtab'
# COMMON-NEXT: <internal>:(.shstrtab) is being placed in '.shstrtab'
# SYMTAB-NEXT: <internal>:(.strtab) is being placed in '.strtab'
# COMMON-NOT: <internal>

## Check --orphan-handling=warn reports warnings about orphans.
# RUN: ld.lld --orphan-handling=warn -o /dev/null -T %t.script \
# RUN:   %t.o 2>&1 | FileCheck %s --check-prefixes=COMMON,SYMTAB

# RUN: not ld.lld --orphan-handling=foo -o /dev/null --script %t.script %t.o 2>&1 \
# RUN:   | FileCheck %s --check-prefix=UNKNOWN
# UNKNOWN: unknown --orphan-handling mode: foo

.section .text.1,"a"
 nop

.section .text.2,"a"
 nop
