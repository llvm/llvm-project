# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows %s -o %t/foo.obj
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows /dev/null -o "%t/bar baz.obj"
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows /dev/null -o "%t/#quux$.obj"
# RUN: lld-link /out:%t/foo.exe /entry:main /subsystem:console %t/foo.obj \
# RUN:   %t/"bar baz.obj" "%t/#quux$.obj" /dependency-file:%t/foo.d
# RUN: FileCheck --match-full-lines -DFILE=%t %s < %t/foo.d

# CHECK:      [[FILE]]{{/|(\\)+}}foo.exe: \
# CHECK-NEXT:   [[FILE]]{{/|(\\)+}}foo.obj \
# CHECK-NEXT:   [[FILE]]{{/|(\\)+}}bar\ baz.obj \
# CHECK-NEXT:   [[FILE]]{{/|(\\)+}}\#quux$$.obj
# CHECK-EMPTY:
# CHECK-NEXT: [[FILE]]{{/|(\\)+}}foo.obj:
# CHECK-EMPTY:
# CHECK-NEXT: [[FILE]]{{/|(\\)+}}bar\ baz.obj:
# CHECK-EMPTY:
# CHECK-NEXT: [[FILE]]{{/|(\\)+}}\#quux$$.obj:

## The target is escaped too.
# RUN: lld-link "/out:%t/foo bar.exe" /entry:main /subsystem:console \
# RUN:   %t/foo.obj /dependency-file:%t/foo2.d
# RUN: FileCheck --check-prefix=TARGET --match-full-lines -DFILE=%t %s < %t/foo2.d

# TARGET:      [[FILE]]{{/|(\\)+}}foo\ bar.exe: \
# TARGET-NEXT:   [[FILE]]{{/|(\\)+}}foo.obj

.globl main
main:
  ret
