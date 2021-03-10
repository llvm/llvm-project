# REQUIRES: x86, aarch64
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/x86_64-test.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t/arm64-test.o
# RUN: %lld -arch x86_64 -platform_version macos 10.5.0 11.0 -o %t/x86-64-executable %t/x86_64-test.o
# RUN: %lld -arch arm64 -o %t/arm64-executable %t/arm64-test.o
# RUN: %lld -arch x86_64 -dylib -o %t/x86-64-dylib %t/x86_64-test.o
# RUN: %lld -arch arm64  -dylib -o %t/arm64-dylib %t/arm64-test.o

# RUN: llvm-objdump --macho --private-header %t/x86-64-executable | FileCheck %s -DCAPS=LIB64
# RUN: llvm-objdump --macho --private-header %t/arm64-executable | FileCheck %s -DCAPS=0x00
# RUN: llvm-objdump --macho --private-header %t/x86-64-dylib | FileCheck %s -DCAPS=0x00
# RUN: llvm-objdump --macho --private-header %t/arm64-dylib | FileCheck %s -DCAPS=0x00

# CHECK:      magic        cputype cpusubtype  caps     filetype {{.*}} flags
# CHECK-NEXT: MH_MAGIC_64  {{.*}}         ALL  [[CAPS]] {{.*}}          NOUNDEFS {{.*}} TWOLEVEL

.globl _main
_main:
  ret
