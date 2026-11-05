# REQUIRES: aarch64, x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11 %t/empty.s -o %t/empty-arm64.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11 %t/code.s -o %t/code-arm64.o
# RUN: %lld -arch arm64 -dylib %t/empty-arm64.o %t/code-arm64.o -o %t/arm64.dylib
# RUN: llvm-objdump --macho --syms --unwind-info %t/arm64.dylib | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos11 %t/empty.s -o %t/empty-x86_64.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos11 %t/code.s -o %t/code-x86_64.o
# RUN: %lld -dylib %t/empty-x86_64.o %t/code-x86_64.o -o %t/x86_64.dylib
# RUN: llvm-objdump --macho --syms --unwind-info %t/x86_64.dylib | FileCheck %s

## An empty input code section has the same output address as the following
## function. Its local symbol must not add a conflicting no-unwind entry there.
## A real function without unwind information must still get a zero entry.
## The alias also checks that same-address symbols in nonempty sections work.

# CHECK: SYMBOL TABLE:
# CHECK-DAG: [[#%x,FIRST:]] g F __TEXT,__text _first
# CHECK-DAG: [[#%x,FIRST]] g F __TEXT,__text _first_alias
# CHECK-DAG: [[#%x,NO_UNWIND:]] g F __TEXT,__text _no_unwind
# CHECK-DAG: [[#%x,LAST:]] g F __TEXT,__text _last
# CHECK: Second level indices:
# CHECK-NEXT: Second level index[0]:
# CHECK-NEXT: [0]: function offset=0x[[#%.8x,FIRST]], encoding{{.*}}=0x02000000
# CHECK-NEXT: [1]: function offset=0x[[#%.8x,NO_UNWIND]], encoding{{.*}}=0x00000000
# CHECK-NEXT: [2]: function offset=0x[[#%.8x,LAST]], encoding{{.*}}=0x02000000
# CHECK-NOT: function offset=

#--- empty.s
.section __TEXT,__text,regular,pure_instructions
l_empty:
.subsections_via_symbols

#--- code.s
.text
.globl _first, _first_alias, _no_unwind, _last
_first:
_first_alias:
  ret
_no_unwind:
  ret
_last:
  ret
Llast_end:

## Hand-written records make the unwind encodings identical on both targets.
.section __LD,__compact_unwind,regular,debug
.p2align 3
.quad _first
.long _no_unwind - _first
.long 0x02000000
.quad 0
.quad 0
.quad _last
.long Llast_end - _last
.long 0x02000000
.quad 0
.quad 0
.subsections_via_symbols
