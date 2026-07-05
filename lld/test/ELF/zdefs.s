# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o

## Unresolved references from object files are allowed by default for -shared.
# RUN: ld.lld -shared %t.o -o %t1.so

## -z defs disallows unresolved references.
# RUN: not ld.lld -z defs -shared %t.o -o /dev/null 2>&1 | FileCheck -check-prefix=ERR %s
# ERR: error: undefined symbol: foo
# ERR: >>> referenced by {{.*}}:(.text+0x1)

## -z undefs allows unresolved references.
# RUN: ld.lld -z defs -z undefs -shared %t.o -o /dev/null 2>&1 | count 0

callq foo@PLT
