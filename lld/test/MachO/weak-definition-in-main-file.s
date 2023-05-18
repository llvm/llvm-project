# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

## Test that a weak symbol in a direct .o file wins over
## a weak symbol in a .a file.

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weakfoo.s -o %t/weakfoo.o

# RUN: llvm-ar --format=darwin rcs %t/weakfoo.a %t/weakfoo.o

# PREFER-DIRECT-OBJECT-NOT: O __TEXT,weak __foo

# RUN: %lld -lSystem -o %t/out %t/weakfoo.a %t/test.o
# RUN: llvm-objdump --syms %t/out | FileCheck %s --check-prefix=PREFER-DIRECT-OBJECT

#--- weakfoo.s
.globl __baz
__baz:
  ret

.section __TEXT,weak
.weak_definition __foo
.globl __foo
__foo:
  ret

.subsections_via_symbols

#--- test.s
.globl __foo
.weak_definition __foo
__foo:
  ret

.globl _main
_main:
  # This pulls in weakfoo.a due to the __baz undef, but __foo should
  # still be resolved against the weak symbol in this file.
  callq __baz
  callq __foo
  ret

.subsections_via_symbols
