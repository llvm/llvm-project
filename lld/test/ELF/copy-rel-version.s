# REQUIRES: x86
## Copy relocate a versioned symbol which has a versioned alias.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/copy-rel-version.s -o %t.o
# RUN: echo 'v1 {}; v2 {}; v3 {};' > %t.ver
# RUN: ld.lld %t.o -shared -soname t.so --version-script=%t.ver -o %t.so

## Copy relocate the default version symbol.
# RUN: ld.lld %t1.o %t.so -o %t1
# RUN: llvm-readelf --dyn-syms %t1 | FileCheck %s --check-prefix=CHECK1

# CHECK1:       1: {{.+}}            12 OBJECT  GLOBAL DEFAULT [[#]] foo@v3
# CHECK1-EMPTY:

## Copy relocate the non-default version symbol.
# RUN: llvm-objcopy --redefine-sym foo=foo@v1 %t1.o %t2.o
# RUN: ld.lld %t2.o %t.so -o %t2
# RUN: llvm-readelf --dyn-syms %t2 | FileCheck %s --check-prefix=CHECK2

# CHECK2:       1: [[ADDR:[0-9a-f]+]] 4 OBJECT  GLOBAL DEFAULT [[#]] foo@v1
# CHECK2-NEXT:  2: [[ADDR]]          12 OBJECT  GLOBAL DEFAULT [[#]] foo@v3
# CHECK2-EMPTY:

.global _start
_start:
  leaq foo, %rax
