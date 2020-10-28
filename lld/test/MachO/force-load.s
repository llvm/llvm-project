# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: echo ".section __TEXT,archive; .globl _foo; .weak_definition _foo; _foo:" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/archive-foo.o
# RUN: rm -f %t/foo.a
# RUN: llvm-ar rcs %t/foo.a %t/archive-foo.o
# RUN: echo ".section __TEXT,obj; .globl _foo; .weak_definition _foo; _foo:" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o

# RUN: %lld -force_load %t/foo.a %t/foo.o %t/test.o -o %t/test-force-load-first
# FORCE-LOAD-FIRST:  __TEXT,archive _foo
# RUN: llvm-objdump --syms %t/test-force-load-first | FileCheck %s --check-prefix=FORCE-LOAD-FIRST

# RUN: %lld %t/foo.o -force_load %t/foo.a %t/test.o -o %t/test-force-load-second
# RUN: llvm-objdump --syms %t/test-force-load-second | FileCheck %s --check-prefix=FORCE-LOAD-SECOND
# FORCE-LOAD-SECOND: __TEXT,obj _foo

.globl _main
_main:
  ret
