## Test that BOLT correctly handles mold-style STT_FUNC symbols in PLT
## sections.

# REQUIRES: system-linux

## Build a shared library from the common stubs.
# RUN: %clang %cflags %p/../Inputs/stub.c -fPIC -shared -o %t.so

## Build and link the main binary. The linker creates a real PLT entry.
# RUN: llvm-mc -filetype=obj -triple aarch64-linux %s -o %t.o
# RUN: ld.lld -pie %t.o %t.so -o %t.exe --emit-relocs

## Inject a mold-style STT_FUNC symbol at the PLT entry for printf.
## Mold places "printf$plt" directly on the PLT stub; we simulate this
## with llvm-objcopy. The PLT header is 32 bytes on AArch64, and each
## entry is 16 bytes. printf is the only entry after the header.
# RUN: llvm-objcopy --add-symbol "printf\$plt=.plt:32,function,local" \
# RUN:   %t.exe %t

## Verify BOLT resolves the call as printf@PLT, not printf$plt.
# RUN: llvm-bolt %t -o %t.bolt --print-cfg --print-only=_start 2>&1 \
# RUN:   | FileCheck %s
# RUN: llvm-readobj --syms %t.bolt | grep -A7 printf$plt | FileCheck \
# RUN:   %s --check-prefix=CHECK-SYM

# CHECK:     bl printf@PLT
# CHECK-NOT: printf$plt
# CHECK-SYM: Section: .plt

  .text
  .globl _start
  .type _start, %function
_start:
  bl printf
  ret
  .size _start, .-_start
