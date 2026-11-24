# REQUIRES: x86
## A symbol whose definition was already pulled in from an archive must not
## pull in a second definition from a later archive, even when a dylib with a
## weak definition of the same name sits in between.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/b.s -o %t/b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/weak.s -o %t/weak.o
# RUN: llvm-ar rcs %t/liba.a %t/a.o
# RUN: llvm-ar rcs %t/libb.a %t/b.o
# RUN: %lld -dylib %t/weak.o -o %t/libweak.dylib
# RUN: %lld -lSystem %t/main.o %t/liba.a %t/libweak.dylib %t/libb.a \
# RUN:   -why_load -o %t/out | FileCheck %s

# CHECK:     _foo forced load of {{.*}}liba.a(a.o)
# CHECK-NOT: libb.a

#--- main.s
.globl _main
_main:
  callq _foo
  ret

#--- a.s
.globl _foo
_foo:
  ret

#--- b.s
.globl _foo
_foo:
  ret

#--- weak.s
.globl _foo
.weak_definition _foo
_foo:
  ret
