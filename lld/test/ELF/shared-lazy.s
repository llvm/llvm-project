# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o && ld.lld a.o -shared -o a.so
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o && ld.lld b.o -shared -o b.so
# RUN: llvm-ar rc a.a a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux ref.s -o ref.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux ref2.s -o ref2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux weakref2.s -o weakref2.o
# RUN: ld.lld a.a b.so ref.o -shared -o 1.so
# RUN: llvm-readelf --dyn-syms 1.so | FileCheck %s
# RUN: ld.lld a.so a.a ref.o -shared -o 1.so
# RUN: llvm-readelf --dyn-syms 1.so | FileCheck %s

## The definitions from a.so are used and we don't extract a member from the
## archive.

# CHECK:      0000000000000000     0 NOTYPE  GLOBAL DEFAULT UND x1
# CHECK-NEXT: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT UND x2

## The extracted x1 is defined as STB_GLOBAL.
# RUN: ld.lld ref.o a.a b.so -o 2.so -shared
# RUN: llvm-readelf --dyn-symbols 2.so | FileCheck %s --check-prefix=CHECK2
# RUN: ld.lld a.a ref.o b.so -o 2.so -shared
# RUN: llvm-readelf --dyn-symbols 2.so | FileCheck %s --check-prefix=CHECK2

# CHECK2:      {{.*}}               0 NOTYPE  GLOBAL DEFAULT [[#]] x1
# CHECK2-NEXT: {{.*}}               0 NOTYPE  WEAK   DEFAULT [[#]] x2

## The extracted x2 is defined as STB_WEAK. x1 is not referenced by any relocatable object file.
# RUN: ld.lld a.a ref2.o b.so -o 2.so -shared
# RUN: llvm-readelf --dyn-syms 2.so | FileCheck %s --check-prefix=CHECK2
# RUN: ld.lld a.a a.so ref2.o -o 3.so -shared
# RUN: llvm-readelf --dyn-syms 3.so | FileCheck %s --check-prefix=CHECK3
# RUN: ld.lld a.so a.a ref2.o -o 3.so -shared
# RUN: llvm-readelf --dyn-syms 3.so | FileCheck %s --check-prefix=CHECK3

# CHECK3:       1: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT UND x2
# CHECK3-EMPTY:

# RUN: ld.lld a.a weakref2.o a.so -o 4.so -shared
# RUN: llvm-readelf --dyn-syms 4.so | FileCheck %s --check-prefix=CHECK4

# CHECK4:       1: 0000000000000000     0 NOTYPE  WEAK   DEFAULT UND x2
# CHECK4-EMPTY:

# RUN: ld.lld a.a --as-needed a.so -o noneeded.so -shared
# RUN: llvm-readelf -d noneeded.so | FileCheck %s --check-prefix=NONEEDED

# NONEEDED-NOT: NEEDED

#--- a.s
.global x1
x1:
.weak x2
x2:
#--- b.s
.globl x1, x2
x1:
x2:
  .byte 0
.size x1, .-x1
.size x2, .-x2
#--- ref.s
.globl x1
.globl x2
#--- ref2.s
.globl x2
#--- weakref2.s
.weak x2
