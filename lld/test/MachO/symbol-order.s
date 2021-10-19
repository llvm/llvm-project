# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/g.s  -o %t/g.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/f1.s -o %t/f1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/f2.s -o %t/f2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/fg.s -o %t/fg.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: %lld -dylib -o %t/libf1.dylib %t/f1.o -lSystem

# RUN: llvm-ar rcs %t/libf2_g.a %t/f2.o %t/g.o
# RUN: llvm-ar rcs %t/libfg.a %t/fg.o

# RUN: %lld %t/libf1.dylib %t/libf2_g.a %t/test.o -o %t/test.out -lSystem
# RUN: llvm-objdump --syms --macho --lazy-bind %t/test.out | FileCheck %s --check-prefix DYLIB-FIRST
# DYLIB-FIRST:      SYMBOL TABLE:
# DYLIB-FIRST-DAG:  __TEXT,test_g g
# DYLIB-FIRST:      Lazy bind table:
# DYLIB-FIRST-NEXT: segment  section            address       dylib            symbol
# DYLIB-FIRST-NEXT: __DATA   __la_symbol_ptr    {{[0-9a-z]+}} libf1            f

# RUN: %lld %t/libf2_g.a %t/libf1.dylib %t/test.o -o %t/test.out -lSystem
# RUN: llvm-objdump --syms --macho --lazy-bind %t/test.out | FileCheck %s --check-prefix ARCHIVE-FIRST
# ARCHIVE-FIRST:      SYMBOL TABLE:
# ARCHIVE-FIRST-DAG:  __TEXT,test_f2 f
# ARCHIVE-FIRST-DAG:  __TEXT,test_g g
# ARCHIVE-FIRST:      Lazy bind table:
# ARCHIVE-FIRST-NEXT: segment  section            address       dylib            symbol
# ARCHIVE-FIRST-EMPTY:

# RUN: %lld %t/libf1.dylib %t/libfg.a %t/test.o -o %t/test.out -lSystem
# RUN: llvm-objdump --syms --macho --lazy-bind %t/test.out | FileCheck %s --check-prefix ARCHIVE-PRIORITY
# ARCHIVE-PRIORITY:      SYMBOL TABLE:
# ARCHIVE-PRIORITY-DAG:  __TEXT,test_fg f
# ARCHIVE-PRIORITY-DAG:  __TEXT,test_fg g
# ARCHIVE-PRIORITY:      Lazy bind table:
# ARCHIVE-PRIORITY-NEXT: segment  section            address       dylib            symbol
# ARCHIVE-PRIORITY-EMPTY:

#--- g.s
.global f, g
.section __TEXT,test_g
g:
  callq f

#--- f1.s
.global f
.section __TEXT,test_f1
f:
  ret

#--- f2.s
.global f
.section __TEXT,test_f2
f:
  ret

#--- fg.s
.global f, g
.section __TEXT,test_fg
f:
  ret
g:
  callq f

#--- test.s
.global g
.global _main
_main:
  callq g
  ret
