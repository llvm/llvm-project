# REQUIRES: x86
# RUN: echo 'v1 { f; }; v2 { g; };' > %t.ver
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -shared --version-script %t.ver %t.o -o %t.so

# RUN: ld.lld --version-script %t.ver %t.o %t.so -o /dev/null -y f@v1 | \
# RUN:   FileCheck --check-prefix=TRACE %s --implicit-check-not=f@v1

## TRACE:      {{.*}}.o: definition of f@v1
## TRACE-NEXT: {{.*}}.so: shared definition of f@v1

# RUN: echo '.symver f,f@v1; .symver g,g@v2; call f; call g' | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: ld.lld -shared %t1.o %t.so -o %t1.so

## Test that we can parse SHT_GNU_verneed to know that the undefined symbols in
## %t1.so are called 'f@v1' and 'g@v2', which can be satisfied by the executable.
## We will thus export the symbols.
# RUN: ld.lld -pie %t.o %t1.so -o %t
# RUN: llvm-nm -D %t | FileCheck --check-prefix=NM %s

# NM: T f
# NM: T g

## The default is --no-allow-shlib-undefined.
## Don't error because undefined symbols in %t1.so are satisfied by %t.so
# RUN: llvm-mc -filetype=obj -triple=x86_64 /dev/null -o %t2.o
# RUN: ld.lld %t2.o %t1.so %t.so -y f@v1 -o /dev/null | FileCheck %s

# CHECK:      {{.*}}1.so: reference to f@v1
# CHECK-NEXT: {{.*}}.so: shared definition of f@v1

.globl f_v1, g_v2
.symver f_v1,f@v1
.symver g_v2,g@v2
f_v1:
g_v2:
