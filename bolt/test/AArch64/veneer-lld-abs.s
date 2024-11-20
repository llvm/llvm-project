## Check that llvm-bolt correctly recognizes long absolute thunks generated
## by LLD.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -fno-PIC -no-pie %t.o -o %t.exe -nostdlib \
# RUN:    -fuse-ld=lld -Wl,-q
# RUN: llvm-objdump -d %t.exe | FileCheck --check-prefix=CHECK-INPUT %s
# RUN: llvm-objcopy --remove-section .rela.mytext %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --elim-link-veneers=true --lite=0
# RUN: llvm-objdump -d -j .text  %t.bolt | \
# RUN:   FileCheck --check-prefix=CHECK-OUTPUT %s

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

.global __AArch64AbsLongThunk_foo
.type __AArch64AbsLongThunk_foo, %function
__AArch64AbsLongThunk_foo:
  ldr x16, .L1
  br x16
# CHECK-INPUT-LABEL: <__AArch64AbsLongThunk_foo>:
# CHECK-INPUT-NEXT:    ldr
# CHECK-INPUT-NEXT:    br
.L1:
  .quad foo
.size __AArch64AbsLongThunk_foo, .-__AArch64AbsLongThunk_foo

## Check that the thunk was removed from .text and _start() calls foo()
## directly.

# CHECK-OUTPUT-NOT: __AArch64AbsLongThunk_foo

.global _start
.type _start, %function
_start:
# CHECK-INPUT-LABEL:  <_start>:
# CHECK-OUTPUT-LABEL: <_start>:
  bl __AArch64AbsLongThunk_foo
# CHECK-INPUT-NEXT:     bl {{.*}} <__AArch64AbsLongThunk_foo>
# CHECK-OUTPUT-NEXT:    bl {{.*}} <foo>
  ret
.size _start, .-_start
