# REQUIRES: aarch64
## Authenticated pointers can only be encoded as chained fixups, so arm64e
## output always uses them: also below the deployment targets at which lld
## switches other architectures to chained fixups, and when -no_fixup_chains is
## given. In the latter case lld warns and, like ld64, uses
## DYLD_CHAINED_PTR_ARM64E even where DYLD_CHAINED_PTR_ARM64E_USERLAND24 is the
## default, and keeps __mod_init_func.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/foo.o %t/foo.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/test.o %t/test.s
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o %t/arm64.o %t/arm64.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/local.o %t/local.s
# RUN: %lld -arch arm64e -platform_version macos 11.0 11.0 -dylib \
# RUN:   -install_name @executable_path/libfoo.dylib %t/foo.o -o %t/libfoo.dylib

## macOS 11.0 is below the deployment target at which lld uses chained fixups
## for arm64.
# RUN: %lld -arch arm64e -platform_version macos 11.0 11.0 -dylib %t/test.o \
# RUN:   %t/libfoo.dylib -o %t/macos11.dylib
# RUN: llvm-objdump --macho --all-headers %t/macos11.dylib | \
# RUN:   FileCheck %s --check-prefixes=INIT-OFFSETS,CHAINED \
# RUN:   --implicit-check-not=LC_DYLD_INFO --implicit-check-not=__mod_init_func
# RUN: llvm-objdump --macho --chained-fixups %t/macos11.dylib | \
# RUN:   FileCheck %s --check-prefix=ARM64E
# RUN: llvm-objdump --macho --dyld-info %t/macos11.dylib | \
# RUN:   FileCheck %s --check-prefix=PTRS

# RUN: %no-fatal-warnings-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -dylib -no_fixup_chains %t/test.o %t/libfoo.dylib \
# RUN:   -o %t/no-fixup-chains.dylib 2>&1 | FileCheck %s --check-prefix=WARN
# RUN: llvm-objdump --macho --all-headers %t/no-fixup-chains.dylib | \
# RUN:   FileCheck %s --check-prefixes=MOD-INIT,CHAINED \
# RUN:   --implicit-check-not=LC_DYLD_INFO --implicit-check-not=__init_offsets
# RUN: llvm-objdump --macho --chained-fixups %t/no-fixup-chains.dylib | \
# RUN:   FileCheck %s --check-prefix=ARM64E
# RUN: llvm-objdump --macho --dyld-info %t/no-fixup-chains.dylib | \
# RUN:   FileCheck %s --check-prefix=PTRS

## arm64e chains pointers with an 8-byte stride; don't suggest the ignored
## -no_fixup_chains when two pointers are 4 bytes out of step.
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/unaligned.o \
# RUN:   %t/unaligned.s
# RUN: not %lld -arch arm64e -platform_version macos 13.0 13.0 -dylib \
# RUN:   %t/unaligned.o -o /dev/null 2>&1 | FileCheck %s \
# RUN:   --check-prefix=UNALIGNED --implicit-check-not=-no_fixup_chains

## arm64 still honors -no_fixup_chains.
# RUN: %lld -arch arm64 -platform_version macos 13.0 13.0 -dylib \
# RUN:   -no_fixup_chains %t/arm64.o -o %t/arm64.dylib
# RUN: llvm-objdump --macho --all-headers %t/arm64.dylib | \
# RUN:   FileCheck %s --check-prefix=OPCODES --implicit-check-not=LC_DYLD_CHAINED

## The releases from which arm64e uses DYLD_CHAINED_PTR_ARM64E_USERLAND24.
# DEFINE: %{link} = %no-arg-lld -arch arm64e -fatal_warnings -dylib \
# DEFINE:   %t/local.o -o %t/format.dylib
# DEFINE: %{format} = llvm-objdump --macho --chained-fixups %t/format.dylib
# RUN: %{link} -platform_version macos 11.0 11.0
# RUN: %{format} | FileCheck %s --check-prefix=ARM64E
# RUN: %{link} -platform_version macos 12.0 12.0
# RUN: %{format} | FileCheck %s --check-prefix=USERLAND24
# RUN: %{link} -platform_version mac-catalyst 14.0 14.0
# RUN: %{format} | FileCheck %s --check-prefix=ARM64E
# RUN: %{link} -platform_version mac-catalyst 15.0 15.0
# RUN: %{format} | FileCheck %s --check-prefix=USERLAND24
# RUN: %{link} -platform_version ios 14.0 14.0
# RUN: %{format} | FileCheck %s --check-prefix=ARM64E
# RUN: %{link} -platform_version ios 15.0 15.0
# RUN: %{format} | FileCheck %s --check-prefix=USERLAND24
# RUN: %{link} -platform_version ios-simulator 16.0 16.0
# RUN: %{format} | FileCheck %s --check-prefix=USERLAND24
# RUN: %{link} -platform_version tvos 14.0 14.0
# RUN: %{format} | FileCheck %s --check-prefix=ARM64E
# RUN: %{link} -platform_version tvos 15.0 15.0
# RUN: %{format} | FileCheck %s --check-prefix=USERLAND24
# RUN: %{link} -platform_version watchos 7.0 7.0
# RUN: %{format} | FileCheck %s --check-prefix=ARM64E
# RUN: %{link} -platform_version watchos 8.0 8.0
# RUN: %{format} | FileCheck %s --check-prefix=USERLAND24
# RUN: %{link} -platform_version xros 1.0 1.0
# RUN: %{format} | FileCheck %s --check-prefix=USERLAND24
# RUN: %{link} -platform_version driverkit 20.0 20.0
# RUN: %{format} | FileCheck %s --check-prefix=ARM64E
# RUN: %{link} -platform_version driverkit 21.0 21.0
# RUN: %{format} | FileCheck %s --check-prefix=USERLAND24

# WARN: warning: bind opcodes are no longer supported with arm64e, switching to chained fixups
# UNALIGNED: error: __DATA,__data, offset 12: fixups are unaligned (offset 12 is not a multiple of the stride)

# INIT-OFFSETS: sectname __init_offsets
# MOD-INIT:     sectname __mod_init_func
# CHAINED:      cmd LC_DYLD_CHAINED_FIXUPS
# OPCODES:      cmd LC_DYLD_INFO_ONLY

# ARM64E:     pointer_format = 1 (DYLD_CHAINED_PTR_ARM64E)
# USERLAND24: pointer_format = 12 (DYLD_CHAINED_PTR_ARM64E_USERLAND24)

## _local_ptr is a rebase, and _foo_ptr a bind of _foo.
# PTRS:      __DATA __data 0x{{[0-9A-F]+}} 0x{{[0-9A-F]+}} rebase
# PTRS-NEXT: __DATA __data 0x{{[0-9A-F]+}} 0x{{[0-9A-F]+}} bind 0x0 libfoo _foo

#--- foo.s
.globl _foo
_foo:
  ret

#--- test.s
.p2align 2
_init:
  ret

.globl _local
_local:
  ret

.data
.p2align 3
.globl _local_ptr, _foo_ptr
_local_ptr:
  .quad _local
_foo_ptr:
  .quad _foo

.section __DATA,__mod_init_func,mod_init_funcs
.p2align 3
  .quad _init

#--- arm64.s
.data
.p2align 3
_ptr:
  .quad _ptr

#--- local.s
.data
.p2align 3
_ptr:
  .quad _ptr

#--- unaligned.s
.data
.p2align 3
_a:
  .quad _a
  .long 0
  .quad _a
