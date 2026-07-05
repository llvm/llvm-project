# REQUIRES: x86, aarch64
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-iossimulator %s -o %t.o
## This should succeed even though libsystem_kernel.dylib has a mismatched platform.
# RUN: %no-arg-lld -lSystem -arch x86_64 -platform_version ios-simulator 14.0 15.0 \
# RUN:   -syslibroot %S/Inputs/iPhoneSimulator.sdk %t.o -o %t
# RUN: llvm-objdump --macho --bind %t | FileCheck %s
# CHECK: __DATA_CONST __got  0x100001000 pointer  0 libSystem  dyld_stub_binder

.globl _main
_main:
  callq ___fsync
  ret
