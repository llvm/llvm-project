// This test checks that the veneer are properly handled by BOLT.
// Strip .rela.mytext section to simulate inserted by a linker veneers
// that does not contain relocations.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags -fPIC -pie %t.o -o %t.exe -nostdlib \
# RUN:    -fuse-ld=lld -Wl,--no-relax -Wl,-q
# RUN: llvm-objdump -d --disassemble-symbols='myveneer' %t.exe | \
# RUN:   FileCheck --check-prefix=CHECKVENEER %s
# RUN: llvm-objcopy --remove-section .rela.mytext %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --elim-link-veneers=true --lite=0
# RUN: llvm-objdump -d -j .text --disassemble-symbols='myveneer' %t.bolt | \
# RUN:   FileCheck --check-prefix=CHECKOUTVENEER %s
# RUN: llvm-objdump -d --disassemble-symbols='_start' %t.bolt | FileCheck %s

.text
.balign 4
.global foo
.type foo, %function
foo:
  adrp x1, foo
  ret
.size foo, .-foo

.section ".mytext", "ax"
.balign 4
# CHECKOUTVENEER-NOT: {{.*}} <myveneer>:
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
