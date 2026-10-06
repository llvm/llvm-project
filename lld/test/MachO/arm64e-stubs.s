# REQUIRES: aarch64
## On arm64e, calls to dylib functions go through __stubs, which load the
## target from an unsigned __got slot, as on arm64.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/foo.o %t/foo.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/test.o %t/test.s
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 -dylib \
# RUN:   -install_name @executable_path/libfoo.dylib %t/foo.o -o %t/libfoo.dylib
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 %t/test.o \
# RUN:   %t/libfoo.dylib -o %t/test
# RUN: llvm-objdump --macho --section-headers %t/test > %t/out
# RUN: llvm-objdump --no-print-imm-hex --macho -d --no-show-raw-insn \
# RUN:   --section=__TEXT,__stubs %t/test >> %t/out
# RUN: llvm-objdump --macho --dyld-info %t/test >> %t/out
# RUN: FileCheck %s < %t/out --implicit-check-not=__auth

# CHECK:      __stubs
# CHECK-NEXT: __got {{[0-9a-f]+}} {{0*}}[[#%x,GOT:]] DATA

# CHECK-LABEL: _main:
# CHECK-NEXT:  bl {{.*}} ; symbol stub for: _foo
# CHECK-LABEL: Contents of (__TEXT,__stubs) section
# CHECK-NEXT:  adrp x16, {{.*}} ; 0x[[#GOT]]
# CHECK-NEXT:  ldr x16, [x16] ; literal pool symbol address: _foo
# CHECK-NEXT:  br x16

## The __got slot is an unauthenticated bind of _foo.
# CHECK-LABEL: dyld information:
# CHECK:       __DATA_CONST __got 0x[[#%X,GOT]] 0x4000000000000000 bind 0x0 libfoo _foo

#--- foo.s
.globl _foo
_foo:
  ret

#--- test.s
.globl _main
.p2align 2
_main:
  bl _foo
  ret
