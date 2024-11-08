// This test checks that the gold linker style veneer are properly handled
// by BOLT.
// Strip .rela.mytext section to simulate inserted by a linker veneers
// that does not contain relocations.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags -fPIC -pie %t.o -o %t.exe -nostdlib \
# RUN:    -fuse-ld=lld -Wl,--no-relax -Wl,-q
# RUN: llvm-objcopy --remove-section .rela.mytext %t.exe
# RUN: llvm-objdump -d -j .mytext %t.exe | \
# RUN:   FileCheck --check-prefix=CHECKVENEER %s
# RUN: llvm-bolt %t.exe -o %t.bolt --elim-link-veneers=true \
# RUN:   --lite=0 --use-old-text=0
# RUN: llvm-objdump -d -j .text %t.bolt | FileCheck %s

.text
.balign 4
.global dummy
.type dummy, %function
dummy:
  adrp x1, dummy
  ret
.size dummy, .-dummy

.section ".mytext", "ax"
.balign 4
.global foo
.type foo, %function
foo:
# CHECK: <foo>:
# CHECK-NEXT: {{.*}} bl {{.*}} <foo2>
  bl .L2
  ret
.size foo, .-foo

.global foo2
.type foo2, %function
foo2:
# CHECK: <foo2>:
# CHECK-NEXT: {{.*}} bl {{.*}} <foo2>
  bl .L2
  ret
.size foo2, .-foo2

.global _start
.type _start, %function
_start:
# CHECK: <_start>:
# CHECK-NEXT: {{.*}} bl {{.*}} <foo>
  bl .L1
  adr x0, .L0
  ret
.L0:
  .xword 0
.size _start, .-_start
.L1:
# CHECKVENEER: adrp
# CHECKVENEER-NEXT: add
# CHECKVENEER-NEXT: br
# CHECKVENEER-NEXT: nop
  adrp x16, foo
  add x16, x16, #:lo12:foo
  br x16
  nop
.L2:
# CHECKVENEER-NEXT: adrp
# CHECKVENEER-NEXT: add
# CHECKVENEER-NEXT: br
# CHECKVENEER-NEXT: nop
  adrp x16, foo2
  add x16, x16, #:lo12:foo2
  br x16
  nop
