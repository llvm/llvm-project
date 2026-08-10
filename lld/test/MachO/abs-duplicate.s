# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weakfoo.s -o %t/weakfoo.o
# RUN: not %lld -lSystem %t/test.o %t/weakfoo.o -o /dev/null 2>&1 | FileCheck %s

# CHECK:      error: duplicate symbol: _weakfoo
# CHECK-NEXT: >>> defined in {{.*}}/test.o
# CHECK-NEXT: >>> defined in {{.*}}/weakfoo.o

## Duplicate absolute symbols that will be dead stripped later should not fail.
# RUN: %lld -lSystem -dead_strip --dead-strip-duplicates -map %t/stripped-duplicate-map \
# RUN:     %t/test.o %t/weakfoo.o -o %t/test
# RUN: llvm-objdump --syms %t/test | FileCheck %s --check-prefix=DUP
# DUP-LABEL: SYMBOL TABLE:
# DUP-NEXT:   g F __TEXT,__text _main
# DUP-NEXT:   g F __TEXT,__text __mh_execute_header
# DUP-NEXT:   *UND* dyld_stub_binder

## Dead stripped non-section symbols don't show up in map files because there's no input section.
## Check that _weakfoo doesn't show up. This matches ld64.
# RUN: FileCheck --check-prefix=DUPMAP %s < %t/stripped-duplicate-map
# DUPMAP: _main
# DUPMAP-LABEL: Dead Stripped Symbols
# DUPMAP-NOT: _weakfoo

#--- weakfoo.s
.globl _weakfoo
## The weak attribute is ignored for absolute symbols, so we will have a
## duplicate symbol error for _weakfoo.
.weak_definition _weakfoo
_weakfoo = 0x1234

#--- test.s
.globl _main, _weakfoo
.weak_definition _weakfoo
_weakfoo = 0x5678

.text
_main:
  ret
