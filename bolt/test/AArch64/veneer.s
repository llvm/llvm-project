// This test checks that the veneer are properly handled by BOLT.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags -fPIC -pie %t.o -o %t.exe \
# RUN:    -nostartfiles -nodefaultlibs -Wl,-z,notext \
# RUN:    -fuse-ld=lld -Wl,--no-relax
# RUN: llvm-objdump -d --disassemble-symbols='myveneer' %t.exe | \
# RUN:   FileCheck --check-prefix=CHECKVENEER %s
# RUN: llvm-bolt %t.exe -o %t.bolt --elim-link-veneers=true --lite=0
# RUN: llvm-objdump -d --disassemble-symbols='_start' %t.bolt | FileCheck %s

.text
.balign 4
.global foo
.type foo, %function
foo:
  ret
.size foo, .-foo

.global myveneer
.type myveneer, %function
myveneer:
# CHECKVENEER: adrp
# CHECKVENEER-NEXT: add
  adrp x16, foo
  add x16, x16, #:lo12:foo
  br x16
  nop
.size myveneer, .-myveneer

.global _start
.type _start, %function
_start:
# CHECK: {{.*}} bl {{.*}} <foo>
  bl myveneer
  ret
.size _start, .-_start
