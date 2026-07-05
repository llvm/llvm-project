# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/bar.s -o %t/bar.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/test.s -o %t/test.o

## Dylibs that don't do lazy dynamic calls don't need dyld_stub_binder.
# RUN: %no-lsystem-lld -arch arm64 -dylib %t/foo.o -o %t/libfoo.dylib
# RUN: llvm-nm -m %t/libfoo.dylib | FileCheck --check-prefix=NOSTUB %s

## Binaries that don't do lazy dynamic calls but are linked against
## libSystem.dylib get a reference to dyld_stub_binder even if it's
## not needed.
# RUN: %lld -arch arm64 -lSystem -dylib %t/foo.o -o %t/libfoo.dylib
# RUN: llvm-nm -m %t/libfoo.dylib | FileCheck --check-prefix=STUB %s

## Dylibs that do lazy dynamic calls do need dyld_stub_binder.
# RUN: not %no-lsystem-lld -arch arm64 -dylib %t/bar.o %t/libfoo.dylib \
# RUN:     -o %t/libbar.dylib -no_fixup_chains 2>&1 | \
# RUN:     FileCheck --check-prefix=MISSINGSTUB %s
# RUN: %lld -arch arm64 -lSystem -dylib %t/bar.o  %t/libfoo.dylib \
# RUN:     -o %t/libbar.dylib -no_fixup_chains
# RUN: llvm-nm -m %t/libbar.dylib | FileCheck --check-prefix=STUB %s

## As do executables.
# RUN: not %no-lsystem-lld -arch arm64 %t/libfoo.dylib %t/libbar.dylib %t/test.o \
# RUN:     -o %t/test -no_fixup_chains 2>&1 | FileCheck --check-prefix=MISSINGSTUB %s
# RUN: %lld -arch arm64 -lSystem %t/libfoo.dylib %t/libbar.dylib %t/test.o \
# RUN:     -o %t/test -no_fixup_chains
# RUN: llvm-nm -m %t/test | FileCheck --check-prefix=STUB %s

## Test dynamic lookup of dyld_stub_binder.
# RUN: %no-lsystem-lld -arch arm64 %t/libfoo.dylib %t/libbar.dylib %t/test.o \
# RUN:     -o %t/test -undefined dynamic_lookup -no_fixup_chains
# RUN: llvm-nm -m %t/test | FileCheck --check-prefix=DYNSTUB %s
# RUN: %no-lsystem-lld -arch arm64 %t/libfoo.dylib %t/libbar.dylib %t/test.o \
# RUN:     -o %t/test -U dyld_stub_binder -no_fixup_chains
# RUN: llvm-nm -m %t/test | FileCheck --check-prefix=DYNSTUB %s

# MISSINGSTUB:      error: undefined symbol: dyld_stub_binder
# MISSINGSTUB-NEXT: >>> referenced by lazy binding (normally in libSystem.dylib)

# NOSTUB-NOT: dyld_stub_binder
# STUB: (undefined) external dyld_stub_binder (from libSystem)
# DYNSTUB: (undefined) external dyld_stub_binder (dynamically looked up)

#--- foo.s
.globl _foo
_foo:

#--- bar.s
.text
.globl _bar
_bar:
  bl _foo
  ret

#--- test.s
.text
.globl _main

.p2align 2
_main:
  bl _foo
  bl _bar
  ret
