# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux \
# RUN:   %p/Inputs/allow-shlib-undefined.s -o %t1.o
# RUN: ld.lld -shared %t1.o -o %t.so

# RUN: ld.lld --allow-shlib-undefined %t.o %t.so -o /dev/null
# RUN: not ld.lld --no-allow-shlib-undefined %t.o %t.so -o /dev/null 2>&1 | FileCheck %s
# Executable defaults to --no-allow-shlib-undefined
# RUN: not ld.lld %t.o %t.so -o /dev/null 2>&1 | FileCheck %s
# RUN: ld.lld %t.o %t.so --noinhibit-exec -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
# RUN: ld.lld %t.o %t.so --warn-unresolved-symbols -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
# -shared defaults to --allow-shlib-undefined
# RUN: ld.lld -shared %t.o %t.so -o /dev/null

# RUN: echo | llvm-mc -filetype=obj -triple=x86_64-unknown-linux -o %tempty.o
# RUN: ld.lld -shared %tempty.o -o %tempty.so
# RUN: ld.lld -shared %t1.o %tempty.so -o %t2.so
# RUN: ld.lld --no-allow-shlib-undefined %t.o %t2.so -o /dev/null

# DSO with undefines:
# should link with or without any of these options.
# RUN: ld.lld -shared %t1.o -o /dev/null
# RUN: ld.lld -shared --allow-shlib-undefined %t1.o -o /dev/null
# RUN: ld.lld -shared --no-allow-shlib-undefined %t1.o -o /dev/null

## Check that the error is reported if an unresolved symbol is first seen in a
## regular object file.
# RUN: echo 'callq _unresolved@PLT' | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64-unknown-linux - -o %tref.o
# RUN: not ld.lld --gc-sections %t.o %tref.o %t.so -o /dev/null 2>&1 | FileCheck %s

## Check that the error is reported for each shared library where the symbol
## is referenced.
# RUN: cp %t.so %t2.so
# RUN: not ld.lld %t.o %t.so %t2.so -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefixes=CHECK,CHECK2

.globl _start
_start:
  callq _shared@PLT

# CHECK: error: {{.*}}.so: undefined reference to _unresolved [--no-allow-shlib-undefined]
# CHECK2: error: {{.*}}2.so: undefined reference to _unresolved [--no-allow-shlib-undefined]
# WARN: warning: {{.*}}.so: undefined reference to _unresolved [--no-allow-shlib-undefined]
