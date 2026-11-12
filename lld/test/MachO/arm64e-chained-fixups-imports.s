# REQUIRES: aarch64
## DYLD_CHAINED_PTR_ARM64E binds have 16-bit import ordinals, so an image in
## that format can have at most 65535 imports, while the 24-bit ordinals of
## DYLD_CHAINED_PTR_ARM64E_USERLAND24 have room for more.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/lib.o %t/lib.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/max.o %t/max.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/over.o %t/over.s
# RUN: %lld -arch arm64e -platform_version macos 11.0 11.0 -dylib \
# RUN:   -install_name @executable_path/libmany.dylib %t/lib.o -o %t/libmany.dylib

# RUN: %lld -arch arm64e -platform_version macos 11.0 11.0 -dylib %t/max.o \
# RUN:   %t/libmany.dylib -o %t/max.dylib
# RUN: llvm-objdump --macho --chained-fixups %t/max.dylib | \
# RUN:   FileCheck %s --check-prefix=MAX
# RUN: not %lld -arch arm64e -platform_version macos 11.0 11.0 -dylib %t/over.o \
# RUN:   %t/libmany.dylib -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 -dylib %t/over.o \
# RUN:   %t/libmany.dylib -o %t/userland24.dylib
# RUN: llvm-objdump --macho --chained-fixups %t/userland24.dylib | \
# RUN:   FileCheck %s --check-prefix=USERLAND24

# MAX:        imports_count  = 65535
# MAX:        pointer_format = 1 (DYLD_CHAINED_PTR_ARM64E)
# ERR:        error: chained fixups, imports_count (65536) exceeds max of 65535
# USERLAND24: imports_count  = 65536
# USERLAND24: pointer_format = 12 (DYLD_CHAINED_PTR_ARM64E_USERLAND24)

#--- lib.s
.macro def
.globl _s\@
_s\@:
.endm
.rept 65536
def
.endr
  ret

#--- max.s
.macro ref
.quad _s\@
.endm
.data
.p2align 3
.rept 65535
ref
.endr

#--- over.s
.macro ref
.quad _s\@
.endm
.data
.p2align 3
.rept 65536
ref
.endr
