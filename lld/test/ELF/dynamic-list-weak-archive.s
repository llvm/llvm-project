# REQUIRES: x86

## A weak reference does not fetch the lazy definition. Test foo is preemptable
## even in the presence of a dynamic list, so a dynamic relocation will be
## produced.

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t1.o
# RUN: echo '.globl foo; foo:' | llvm-mc -filetype=obj -triple=x86_64 - -o %t2.o
# RUN: rm -f %t.a
# RUN: llvm-ar rcs %t.a %t2.o
# RUN: echo "{ zed; };" > %t.list
# RUN: ld.lld -shared --dynamic-list %t.list %t1.o %t.a -o %t.so
# RUN: llvm-readobj -r %t.so | FileCheck %s

# CHECK:      Relocations [
# CHECK-NEXT:   Section ({{.*}}) .rela.plt {
# CHECK-NEXT:     R_X86_64_JUMP_SLOT foo
# CHECK-NEXT:   }
# CHECK-NEXT: ]

callq foo@PLT
.weak foo
