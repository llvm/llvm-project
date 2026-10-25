# REQUIRES: aarch64
## On arm64e, the thunk pointer of a thread-local variable descriptor, a bind to
## __tlv_bootstrap, is an authenticated bind with the IA key and no diversity,
## because TLV accesses call it with blraaz. On arm64 it is a plain bind.

# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t-arm64e.o %s
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 -dylib %t-arm64e.o \
# RUN:   -o %t-arm64e.dylib
# RUN: llvm-objdump --macho --dyld-info %t-arm64e.dylib | \
# RUN:   FileCheck %s --check-prefix=ARM64E
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o %t-arm64.o %s
# RUN: %lld -arch arm64 -platform_version macos 13.0 13.0 -dylib %t-arm64.o \
# RUN:   -o %t-arm64.dylib
# RUN: llvm-objdump --macho --dyld-info %t-arm64.dylib | \
# RUN:   FileCheck %s --check-prefix=ARM64

# ARM64E: __DATA __thread_vars 0x{{[0-9A-F]+}} 0x{{[0-9A-F]+}} bind 0x0000 false IA 0x0 libSystem __tlv_bootstrap
# ARM64:  __DATA __thread_vars 0x{{[0-9A-F]+}} 0x{{[0-9A-F]+}} bind 0x0 libSystem __tlv_bootstrap

.section __DATA,__thread_data,thread_local_regular
.p2align 2
_tv$tlv$init:
  .long 5

.section __DATA,__thread_vars,thread_local_variables
.globl _tv
_tv:
  .quad __tlv_bootstrap
  .quad 0
  .quad _tv$tlv$init
