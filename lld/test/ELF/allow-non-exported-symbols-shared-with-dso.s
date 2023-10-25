# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 weak.s -o weak.o

# RUN: llvm-mc -filetype=obj -triple=x86_64 def.s -o def.o
# RUN: ld.lld -shared -soname=def.so def.o -o def.so
# RUN: ld.lld a.o def.so -o /dev/null
# RUN: ld.lld a.o def.so -o /dev/null --no-allow-non-exported-symbols-shared-with-dso --allow-non-exported-symbols-shared-with-dso

# RUN: not ld.lld a.o def.so --no-allow-non-exported-symbols-shared-with-dso 2>&1 | FileCheck %s --check-prefix=CHECK1
# RUN: not ld.lld --gc-sections weak.o def.so a.o --no-allow-non-exported-symbols-shared-with-dso 2>&1 | FileCheck %s --check-prefix=CHECK1

# CHECK1:      error: non-exported symbol also defined by DSO: foo
# CHECK1-NEXT: >>> defined by {{.*}}a.o

# RUN: llvm-mc -filetype=obj -triple=x86_64 ref.s -o ref.o
# RUN: ld.lld -shared -soname=ref.so ref.o -o ref.so
# RUN: not ld.lld a.o ref.so --no-allow-non-exported-symbols-shared-with-dso 2>&1 | FileCheck %s --check-prefix=CHECK2

# CHECK2:      error: non-exported symbol also referenced by DSO: foo
# CHECK2-NEXT: >>> defined by {{.*}}a.o

## See also allow-shlib-undefined.s.

#--- a.s
.globl _start, foo
_start:

## The check kicks in even if .text.foo is discarded.
.section .text.foo,"ax"
.hidden foo
foo:

#--- weak.s
.weak foo
foo:

#--- def.s
.globl foo
foo:

#--- ref.s
call foo
