# REQUIRES: aarch64
## Only the plain pointers in __thread_vars are offsets into the thread-local
## data. An authenticated pointer there is a signed pointer to its target, which
## ld64 emits as an authenticated rebase.

# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos %s -o %t.o
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 -dylib %t.o \
# RUN:   -o %t.dylib
# RUN: llvm-objdump --macho --section-headers %t.dylib > %t.out
# RUN: llvm-objdump --macho --dyld-info %t.dylib >> %t.out
# RUN: llvm-objdump --macho -s --section __DATA,__thread_vars %t.dylib >> %t.out
# RUN: FileCheck %s < %t.out

## Without chained fixups, which -preload turns off, the pointer cannot be
## encoded.
# RUN: not %no-fatal-warnings-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -dylib -preload %t.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# CHECK:      __thread_vars 00000018 {{0*}}[[#%x,VARS:]] DATA
# CHECK-NEXT: __thread_data 00000008 {{0*}}[[#%x,DATA:]] DATA

## The first word is an authenticated rebase of _tls with key DA. The third,
## the offset of _tls in the thread-local data, is not a fixup.
# CHECK-LABEL: dyld information:
# CHECK:       __DATA __thread_vars 0x[[#%X,VARS]] 0x{{[0-9A-F]+}} rebase 0x0000 false DA 0x[[#%X,DATA]]
# CHECK-NOT:   __thread_vars
# CHECK-LABEL: Contents of (__DATA,__thread_vars) section
# CHECK:       {{0*}}[[#VARS+16]] 00000000 00000000

# ERR: error: {{.*}}.o:(symbol _desc+0x0): authenticated pointer requires chained fixups

.section __DATA,__thread_data,thread_local_regular
.p2align 3
_tls:
  .quad 42

.section __DATA,__thread_vars,thread_local_variables
.globl _desc
_desc:
  .quad _tls@AUTH(da,0)
  .quad 0
  .quad _tls
