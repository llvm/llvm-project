# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/archive-foo.s -o %t/archive-foo.o
# RUN: llvm-ar rcs %t/foo.a %t/archive-foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o

# RUN: %lld -force_load %t/foo.a %t/foo.o %t/test.o -o %t/test-force-load-first
# FORCE-LOAD-FIRST:  __TEXT,archive _foo
# RUN: llvm-objdump --syms %t/test-force-load-first | FileCheck %s --check-prefix=FORCE-LOAD-FIRST

# RUN: %lld %t/foo.o -force_load %t/foo.a %t/test.o -o %t/test-force-load-second
# RUN: llvm-objdump --syms %t/test-force-load-second | FileCheck %s --check-prefix=FORCE-LOAD-SECOND
# FORCE-LOAD-SECOND: __TEXT,obj _foo

#--- archive-foo.s
.section __TEXT,archive
.globl _foo
.weak_definition _foo
_foo:

#--- foo.s
.section __TEXT,obj
.globl _foo
.weak_definition _foo
_foo:

#--- test.s
.globl _main
_main:
  ret
