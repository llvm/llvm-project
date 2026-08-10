# REQUIRES: x86
## --exclude-libs can hide version symbols.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/b.s -o %t/b.o
# RUN: llvm-ar rc %t/b.a %t/b.o
# RUN: ld.lld -shared %t/a.o %t/b.a --version-script=%t/ver -o %t.so
# RUN: llvm-readelf --dyn-syms %t.so | FileCheck %s
# RUN: ld.lld -shared %t/a.o %t/b.a --exclude-libs=b.a --version-script=%t/ver -o %t.so
# RUN: llvm-readelf --dyn-syms %t.so | FileCheck %s --check-prefix=NO

# CHECK: foo@@v2
# CHECK: bar@v1
# NO-NOT: foo@@v2
# NO-NOT: bar@v1

#--- a.s
.globl _start
_start:
  call foo

#--- b.s
.symver bar_v1, bar@v1
.globl foo, bar_v1
foo:
bar_v1:
  ret

#--- ver
v1 {};
v2 { foo; };
