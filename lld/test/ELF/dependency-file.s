# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 /dev/null -o "%t/bar baz.o"
# RUN: llvm-mc -filetype=obj -triple=x86_64 /dev/null -o "%t/#quux$.o"
# RUN: ld.lld -o %t/foo.exe %t/foo.o %t/"bar baz.o" "%t/#quux$.o" --dependency-file=%t/foo.d
# RUN: FileCheck --match-full-lines -DFILE=%t %s < %t/foo.d

# CHECK:      [[FILE]]{{/|(\\)+}}foo.exe: \
# CHECK-NEXT:   [[FILE]]{{/|(\\)+}}foo.o \
# CHECK-NEXT:   [[FILE]]{{/|(\\)+}}bar\ baz.o \
# CHECK-NEXT:   [[FILE]]{{/|(\\)+}}\#quux$$.o
# CHECK-EMPTY:
# CHECK-NEXT: [[FILE]]{{/|(\\)+}}foo.o:
# CHECK-EMPTY:
# CHECK-NEXT: [[FILE]]{{/|(\\)+}}bar\ baz.o:
# CHECK-EMPTY:
# CHECK-NEXT: [[FILE]]{{/|(\\)+}}\#quux$$.o:

.global _start
_start:
