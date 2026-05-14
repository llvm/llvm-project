# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11.0 %t/archive.s -o %t/archive.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11.0 %t/dylib.s -o %t/dylib.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11.0 %t/test.s -o %t/test.o

# RUN: llvm-ar rcs %t/libfoo.a %t/archive.o
# RUN: %no-lsystem-lld -arch arm64 -dylib -o %t/libfoo.dylib %t/dylib.o

## Archive appears before dylib. Symbols should be resolved to the archive.
## This should hold even with --read-workers enabled.

# RUN: %lld -arch arm64 -lSystem %t/test.o %t/libfoo.a %t/libfoo.dylib -o %t/test.out --read-workers=2
# RUN: llvm-nm %t/test.out | FileCheck %s

# CHECK-DAG: {{[0-9a-f]+}} {{[ST]}} _foo
# CHECK-DAG: {{[0-9a-f]+}} T _main

## Verify _foo came from the archive and not the dylib by checking its section
# RUN: llvm-objdump --syms --macho %t/test.out | FileCheck %s --check-prefix SECTION
# SECTION: __TEXT,archive_foo _foo

#--- archive.s
.globl _foo
.section __TEXT,archive_foo
_foo:
  ret

#--- dylib.s
.globl _foo
.section __TEXT,dylib_foo
_foo:
  ret

#--- test.s
.globl _main
_main:
  bl _foo
  ret
