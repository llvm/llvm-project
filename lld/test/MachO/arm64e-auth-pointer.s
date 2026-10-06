# REQUIRES: aarch64
## Authenticated pointers become authenticated chained rebases and binds that
## keep their key, diversity, address diversity and addend, next to
## unauthenticated ones. The words and imports below are the ones Xcode 27's ld
## writes for the same object.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/foo.o %t/foo.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/test.o %t/test.s
# RUN: %lld -arch arm64e -platform_version macos 11.0 11.0 -dylib \
# RUN:   -install_name @executable_path/libfoo.dylib %t/foo.o -o %t/libfoo.dylib

# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 %t/test.o \
# RUN:   %t/libfoo.dylib -o %t/userland24
# RUN: llvm-objdump --macho --chained-fixups --syms %t/userland24 > %t/out
# RUN: llvm-objdump --macho --dyld-info %t/userland24 >> %t/out
# RUN: FileCheck %s --check-prefixes=CHECK,USERLAND24 -D#%x,BASE=0x100000000 \
# RUN:   < %t/out

# RUN: %lld -arch arm64e -platform_version macos 11.0 11.0 %t/test.o \
# RUN:   %t/libfoo.dylib -o %t/arm64e
# RUN: llvm-objdump --macho --chained-fixups --syms %t/arm64e > %t/out
# RUN: llvm-objdump --macho --dyld-info %t/arm64e >> %t/out
# RUN: FileCheck %s --check-prefixes=CHECK,ARM64E -D#%x,BASE=0x100000000 \
# RUN:   < %t/out

## An authenticated rebase stores its target as a 32-bit offset from the image
## base, which the target below doesn't fit in.
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/far.o %t/far.s
# RUN: not %lld -arch arm64e -platform_version macos 13.0 13.0 %t/far.o \
# RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=FAR
# FAR: error: vmOffset (0x100008000) cannot fit in fixup at __DATA+0x0

## Authenticated binds have no addend field, so their addends go in the
## imports table. The unauthenticated bind shares import 0 and keeps its
## addend inline.
# CHECK:      imports_format = 2 (DYLD_CHAINED_IMPORT_ADDEND)
# USERLAND24: pointer_format = 12 (DYLD_CHAINED_PTR_ARM64E_USERLAND24)
# ARM64E:     pointer_format = 1 (DYLD_CHAINED_PTR_ARM64E)
# CHECK:      dyld chained import addend[0]
# CHECK:        name_offset = {{[0-9]+}} (_foo)
# CHECK-NEXT:   addend      = 0
# CHECK:      dyld chained import addend[1]
# CHECK:        name_offset = {{[0-9]+}} (_foo)
# CHECK-NEXT:   addend      = -4
# CHECK:      dyld chained import addend[2]
# CHECK:        name_offset = {{[0-9]+}} (_foo)
# CHECK-NEXT:   addend      = 16
# CHECK:      [[#%x,LOCAL:]] l F __TEXT,__text _local

## An authenticated rebase stores its target as an offset from the image base
## in both formats, and an authenticated bind its import's ordinal, with no room
## for an addend. Only the unauthenticated rebase differs between the formats:
## USERLAND24 stores the offset, ARM64E the address.
# CHECK-LABEL:     dyld information:
# CHECK-NEXT:      segment section address pointer type div addr key addend dylib symbol/vm address
# CHECK-NEXT:      __DATA __data 0x[[#%X,BASE+0x4000]] 0x[[#%.16X,0x8008000000000000+LOCAL-BASE]] rebase 0x0000 false IA 0x[[#%X,LOCAL]]
# CHECK-NEXT:      __DATA __data 0x[[#%X,BASE+0x4008]] 0x[[#%.16X,0x800B123400000000+LOCAL+8-BASE]] rebase 0x1234 true IB 0x[[#%X,LOCAL+8]]
# CHECK-NEXT:      __DATA __data 0x[[#%X,BASE+0x4010]] 0xC00C002A00000000 bind 0x002A false DA 0x0 libfoo _foo
# CHECK-NEXT:      __DATA __data 0x[[#%X,BASE+0x4018]] 0xC00FFFFF00000002 bind 0xFFFF true DB 0x10 libfoo _foo
# CHECK-NEXT:      __DATA __data 0x[[#%X,BASE+0x4020]] 0xC008000100000001 bind 0x0001 false IA 0xFFFFFFFFFFFFFFFC libfoo _foo
# USERLAND24-NEXT: __DATA __data 0x[[#%X,BASE+0x4028]] 0x[[#%.16X,0x8000000000000+LOCAL-BASE]] rebase 0x[[#%X,LOCAL]]
# ARM64E-NEXT:     __DATA __data 0x[[#%X,BASE+0x4028]] 0x[[#%.16X,0x8000000000000+LOCAL]] rebase 0x[[#%X,LOCAL]]
# CHECK-NEXT:      __DATA __data 0x[[#%X,BASE+0x4030]] 0x4000000800000000 bind 0x8 libfoo _foo

#--- foo.s
.globl _foo
_foo:
  ret

#--- test.s
.globl _main
.p2align 2
_main:
  ret
_local:
  ret

.data
.p2align 3
  .quad _local@AUTH(ia,0)
  .quad (_local + 8)@AUTH(ib,0x1234,addr)
  .quad _foo@AUTH(da,42)
  .quad (_foo + 16)@AUTH(db,0xffff,addr)
  .quad (_foo - 4)@AUTH(ia,1)
  .quad _local
  .quad _foo + 8

#--- far.s
.globl _main
.p2align 2
_main:
  ret

.data
.p2align 3
  .quad _far@AUTH(da,0)

.zerofill __DATA,__bss,_big,0x100000000

.section __FAR,__data
.p2align 3
_far:
  .quad 0
