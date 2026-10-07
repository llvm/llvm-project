# REQUIRES: aarch64
## The map file lists the entries of arm64e's __auth_got like those of __got.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/foo.o %t/foo.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/test.o %t/test.s
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 -dylib \
# RUN:   -install_name @executable_path/libfoo.dylib %t/foo.o -o %t/libfoo.dylib
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 %t/test.o \
# RUN:   %t/libfoo.dylib -map %t/map -o %t/test
# RUN: llvm-objdump --macho --section-headers %t/test > %t/out
# RUN: cat %t/map >> %t/out
# RUN: FileCheck %s < %t/out

# CHECK:      __auth_got {{[0-9a-f]+}} {{0*}}[[#%x,AUTH_GOT:]] DATA
# CHECK:      __got      {{[0-9a-f]+}} {{0*}}[[#%x,GOT:]] DATA
# CHECK-LABEL: # Sections:
# CHECK:      0x[[#%X,AUTH_GOT]] 0x00000008 __DATA_CONST __auth_got
# CHECK:      0x[[#%X,GOT]] 0x00000008 __DATA_CONST __got
# CHECK-LABEL: # Symbols:
# CHECK:      0x[[#%X,AUTH_GOT]] 0x00000008 [ 0] non-lazy-pointer-to-local: _foo
# CHECK:      0x[[#%X,GOT]] 0x00000008 [ 0] non-lazy-pointer-to-local: _foo

#--- foo.s
.globl _foo
_foo:
  ret

#--- test.s
.globl _main
.p2align 2
_main:
  bl _foo
  adrp x0, _foo@GOTPAGE
  ldr x0, [x0, _foo@GOTPAGEOFF]
  ret
