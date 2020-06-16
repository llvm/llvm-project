# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: echo ".global f, g; .section __TEXT,test_g; g: callq f"          | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/g.o
# RUN: echo ".global f; .section __TEXT,test_f1; f: ret"                | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/f1.o
# RUN: echo ".global f; .section __TEXT,test_f2; f: ret"                | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/f2.o
# RUN: echo ".global f, g; .section __TEXT,test_fg; f: ret; g: callq f" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/fg.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: lld -flavor darwinnew -arch x86_64 -L%S/Inputs/MacOSX.sdk/usr/lib -dylib -o %t/libf1.dylib %t/f1.o -lSystem

# RUN: rm -f %t/libf2_g.a
# RUN: llvm-ar rcs %t/libf2_g.a %t/f2.o %t/g.o

# RUN: rm -f %t/libfg.a
# RUN: llvm-ar rcs %t/libfg.a %t/fg.o

# RUN: lld -flavor darwinnew -arch x86_64 -L%S/Inputs/MacOSX.sdk/usr/lib %t/libf1.dylib %t/libf2_g.a %t/test.o -o %t/test.out -lSystem
# RUN: llvm-objdump --syms --macho --lazy-bind %t/test.out | FileCheck %s --check-prefix DYLIB-FIRST
# DYLIB-FIRST:      SYMBOL TABLE:
# DYLIB-FIRST-DAG:  __TEXT,test_g g
# DYLIB-FIRST:      Lazy bind table:
# DYLIB-FIRST-NEXT: segment  section            address       dylib            symbol
# DYLIB-FIRST-NEXT: __DATA   __la_symbol_ptr    {{[0-9a-z]+}} libf1            f

# RUN: lld -flavor darwinnew -arch x86_64 -L%S/Inputs/MacOSX.sdk/usr/lib %t/libf2_g.a %t/libf1.dylib %t/test.o -o %t/test.out -lSystem
# RUN: llvm-objdump --syms --macho --lazy-bind %t/test.out | FileCheck %s --check-prefix ARCHIVE-FIRST
# ARCHIVE-FIRST:      SYMBOL TABLE:
# ARCHIVE-FIRST-DAG:  __TEXT,test_f2 f
# ARCHIVE-FIRST-DAG:  __TEXT,test_g g
# ARCHIVE-FIRST:      Lazy bind table:
# ARCHIVE-FIRST-NEXT: segment  section            address       dylib            symbol
# ARCHIVE-FIRST-EMPTY:

# RUN: lld -flavor darwinnew -arch x86_64 -L%S/Inputs/MacOSX.sdk/usr/lib %t/libf1.dylib %t/libfg.a %t/test.o -o %t/test.out -lSystem
# RUN: llvm-objdump --syms --macho --lazy-bind %t/test.out | FileCheck %s --check-prefix ARCHIVE-PRIORITY
# ARCHIVE-PRIORITY:      SYMBOL TABLE:
# ARCHIVE-PRIORITY-DAG:  __TEXT,test_fg f
# ARCHIVE-PRIORITY-DAG:  __TEXT,test_fg g
# ARCHIVE-PRIORITY:      Lazy bind table:
# ARCHIVE-PRIORITY-NEXT: segment  section            address       dylib            symbol
# ARCHIVE-PRIORITY-EMPTY:

.global g
.global _main
_main:
  callq g
  ret
