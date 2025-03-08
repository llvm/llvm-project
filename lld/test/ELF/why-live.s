# REQUIRES: x86

# RUN: llvm-mc -n -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo -e ".globl test_shared\n .section .test_shared,\"ax\",@progbits\n test_shared: jmp test_shared" |\
# RUN:   llvm-mc -n -filetype=obj -triple=x86_64 -o %t.shared.o
# RUN: ld.lld -shared %t.shared.o -o %t.so

## Simple live section
.globl _start
.section ._start,"ax",@progbits
_start:
jmp test_simple
.quad .Lanonymous
.quad .Lanonymous_within_symbol
jmp test_shared
.quad test_local
.size _start, .-_start

.globl test_simple
.section .test_simple,"ax",@progbits
test_simple:
jmp test_simple
jmp test_from_unsized

# RUN: ld.lld %t.o %t.so -o /dev/null --gc-sections --why-live=test_simple | FileCheck %s --check-prefix=SIMPLE

# SIMPLE:      live symbol: test_simple
# SIMPLE-NEXT: >>> kept live by _start
# SIMPLE-NOT:  >>>

## Live only by being a member of .test_simple
.globl test_incidental
test_incidental:
jmp test_incidental

# RUN: ld.lld %t.o %t.so -o /dev/null --gc-sections --why-live=test_incidental | FileCheck %s --check-prefix=INCIDENTAL

# INCIDENTAL:      live symbol: test_incidental
# INCIDENTAL-NEXT: >>> kept live by {{.*}}.o:(.test_simple)
# INCIDENTAL-NEXT: >>> kept live by test_simple
# INCIDENTAL-NEXT: >>> kept live by _start
# INCIDENTAL-NOT:  >>>

## Reached from a reference in section .test_simple directly, since test_simple is an unsized symbol.
.globl test_from_unsized
.section .test_from_unsized,"ax",@progbits
test_from_unsized:
jmp test_from_unsized

# RUN: ld.lld %t.o %t.so -o /dev/null --gc-sections --why-live=test_from_unsized | FileCheck %s --check-prefix=FROM-UNSIZED

# FROM-UNSIZED:      live symbol: test_from_unsized
# FROM-UNSIZED-NEXT: >>> kept live by {{.*}}.o:(.test_simple)
# FROM-UNSIZED-NEXT: >>> kept live by test_simple
# FROM-UNSIZED-NEXT: >>> kept live by _start
# FROM-UNSIZED-NOT:  >>>

## Symbols in dead sections are dead and not reported.
.globl test_dead
.section .test_dead,"ax",@progbits
test_dead:
jmp test_dead

# RUN: ld.lld %t.o %t.so -o /dev/null --gc-sections --why-live=test_dead | count 0

## Undefined symbols are considered live, since they are not in dead sections.

# RUN: ld.lld %t.o %t.so -o /dev/null --gc-sections --why-live=test_undef -u test_undef | FileCheck %s --check-prefix=UNDEFINED

# UNDEFINED:     live symbol: test_undef
# UNDEFINED-NOT: >>>

## Defined symbols without input section parents are live.
.globl test_absolute
test_absolute = 1234

# RUN: ld.lld %t.o %t.so -o /dev/null --gc-sections --why-live=test_absolute | FileCheck %s --check-prefix=ABSOLUTE

# ABSOLUTE:     live symbol: test_absolute
# ABSOLUTE-NOT: >>>

## Retained sections are intrinsically live, and they make contained symbols live.
.globl test_retained
.section .test_retained,"axR",@progbits
test_retained:
jmp test_retained

# RUN: ld.lld %t.o %t.so -o /dev/null --gc-sections --why-live=test_retained | FileCheck %s --check-prefix=RETAINED

# RETAINED:      live symbol: test_retained
# RETAINED-NEXT: >>> kept live by {{.*}}:(.test_retained)
# RETAINED-NOT:  >>>

## Relocs that reference offsets from sections (e.g., from anonymous symbols) are considered to point to the section if no enclosing symbol exists.

.globl test_section_offset
.section .test_section_offset,"ax",@progbits
test_section_offset:
jmp test_section_offset
.Lanonymous:
jmp test_section_offset

# RUN: ld.lld %t.o %t.so -o /dev/null --gc-sections --why-live=test_section_offset | FileCheck %s --check-prefix=SECTION-OFFSET

# SECTION-OFFSET:      live symbol: test_section_offset
# SECTION-OFFSET-NEXT: >>> kept live by {{.*}}:(.test_section_offset)
# SECTION-OFFSET-NEXT: >>> kept live by _start
# SECTION-OFFSET-NOT:  >>>

## Relocs that reference offsets from sections (e.g., from anonymous symbols) are considered to point to the enclosing symbol if one exists.

.globl test_section_offset_within_symbol
.section .test_section_offset_within_symbol,"ax",@progbits
test_section_offset_within_symbol:
jmp test_section_offset_within_symbol
.Lanonymous_within_symbol:
jmp test_section_offset_within_symbol
.size test_section_offset_within_symbol, .-test_section_offset_within_symbol

# RUN: ld.lld %t.o %t.so -o /dev/null --gc-sections --why-live=test_section_offset_within_symbol | FileCheck %s --check-prefix=SECTION-OFFSET-WITHIN-SYMBOL

# SECTION-OFFSET-WITHIN-SYMBOL:      live symbol: test_section_offset_within_symbol
# SECTION-OFFSET-WITHIN-SYMBOL-NEXT: >>> kept live by _start
# SECTION-OFFSET-WITHIN-SYMBOL-NOT:  >>>

## Local symbols can be queried just like global symbols.

.section .test_local,"ax",@progbits
test_local:
jmp test_local
.size test_local, .-test_local

# RUN: ld.lld %t.o %t.so -o /dev/null --gc-sections --why-live=test_local | FileCheck %s --check-prefix=LOCAL

# LOCAL:      live symbol: {{.*}}:(test_local)
# LOCAL-NEXT: >>> kept live by _start
# LOCAL-NOT:  >>>

## Shared symbols

# RUN: ld.lld %t.o %t.so -o /dev/null --gc-sections %t.so --why-live=test_shared | FileCheck %s --check-prefix=SHARED

# SHARED:      live symbol: test_shared
# SHARED-NEXT: >>> kept live by _start
# SHARED-NOT:  >>>

## Globs match multiple cases. Multiple --why-live flags union.

# RUN: ld.lld %t.o %t.so -o /dev/null --gc-sections %t.so --why-live=test_s* | FileCheck %s --check-prefix=MULTIPLE
# RUN: ld.lld %t.o %t.so -o /dev/null --gc-sections %t.so --why-live=test_simple --why-live=test_shared | FileCheck %s --check-prefix=MULTIPLE

# MULTIPLE-DAG: live symbol: test_simple
# MULTIPLE-DAG: live symbol: test_shared
