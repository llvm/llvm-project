# REQUIRES: aarch64
## The top byte of a rebased pointer goes in the high8 field. In an executable,
## DYLD_CHAINED_PTR_ARM64E_USERLAND24 stores the rest of the target as an
## offset from the image base, and DYLD_CHAINED_PTR_ARM64E as an address. A
## tagged target below the image base is rejected in both formats, as ld64
## does, even though DYLD_CHAINED_PTR_ARM64E could encode its address.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/tagged.o %t/tagged.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/below.o %t/below.s

# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 %t/tagged.o \
# RUN:   -o %t/userland24
# RUN: llvm-objdump --macho --syms --chained-fixups %t/userland24 > %t/userland24.txt
# RUN: llvm-objdump --macho --dyld-info %t/userland24 >> %t/userland24.txt
# RUN: FileCheck %s --check-prefix=USERLAND24 -D#%x,BASE=0x100000000 \
# RUN:   < %t/userland24.txt

# RUN: %lld -arch arm64e -platform_version macos 11.0 11.0 %t/tagged.o \
# RUN:   -o %t/arm64e
# RUN: llvm-objdump --macho --syms --chained-fixups %t/arm64e > %t/arm64e.txt
# RUN: llvm-objdump --macho --dyld-info %t/arm64e >> %t/arm64e.txt
# RUN: FileCheck %s --check-prefix=ARM64E -D#%x,BASE=0x100000000 \
# RUN:   < %t/arm64e.txt

# RUN: not %lld -arch arm64e -platform_version macos 13.0 13.0 %t/below.o \
# RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=BELOW
# RUN: not %lld -arch arm64e -platform_version macos 11.0 11.0 %t/below.o \
# RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=BELOW-ARM64E

## `_target + 0x1200000000000000`: target:43 is _target's offset from the image
## base (USERLAND24) or its address (ARM64E), and high8:8 is 0x12.
# USERLAND24:  pointer_format = 12 (DYLD_CHAINED_PTR_ARM64E_USERLAND24)
# USERLAND24:  [[#%x,TARGET:]] l O __DATA,__data _target
# USERLAND24:  dyld information:
# USERLAND24:  __DATA __data 0x[[#%X,TARGET+8]] 0x[[#%.16X,0x900000000000+TARGET-BASE]] rebase 0x[[#%X,TARGET+0x1200000000000000]]

# ARM64E:      pointer_format = 1 (DYLD_CHAINED_PTR_ARM64E)
# ARM64E:      [[#%x,TARGET:]] l O __DATA,__data _target
# ARM64E:      dyld information:
# ARM64E:      __DATA __data 0x[[#%X,TARGET+8]] 0x[[#%.16X,0x900000000000+TARGET]] rebase 0x[[#%X,TARGET+0x1200000000000000]]

# BELOW: error: vmOffset (0x11FFFFFF{{[0-9A-F]+}}) cannot fit in fixup at __DATA+0x0
# BELOW-ARM64E: error: vmAddress (0x12000000{{[0-9A-F]+}}) cannot fit in fixup at __DATA+0x0

#--- tagged.s
.globl _main
.p2align 2
_main:
  ret

.data
.p2align 3
_target:
  .quad 0
_tagged:
  .quad _target + 0x1200000000000000

#--- below.s
.globl _main
.p2align 2
_main:
  ret

.data
.p2align 3
_below:
  .quad _main + 0x11ffffff00000000
