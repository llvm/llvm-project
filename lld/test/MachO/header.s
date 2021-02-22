# REQUIRES: x86, arm
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: %lld -arch x86_64 -platform_version macos 10.5.0 11.0 -o %t/x86-64-executable %t/test.o
# RUN: %lld -arch arm64 -o %t/arm64-executable %t/test.o
# RUN: %lld -arch x86_64 -dylib -o %t/x86-64-dylib %t/test.o
# RUN: %lld -arch arm64  -dylib -o %t/arm64-dylib %t/test.o

# RUN: llvm-objdump --macho --all-headers %t/x86-64-executable | FileCheck %s -DCAPS=LIB64
# RUN: llvm-objdump --macho --all-headers %t/arm64-executable | FileCheck %s -DCAPS=0x00
# RUN: llvm-objdump --macho --all-headers %t/x86-64-dylib | FileCheck %s -DCAPS=0x00
# RUN: llvm-objdump --macho --all-headers %t/arm64-dylib | FileCheck %s -DCAPS=0x00

# CHECK:      magic        cputype cpusubtype  caps    filetype
# CHECK-NEXT: MH_MAGIC_64  {{.*}}         ALL  [[CAPS]]   {{.*}}

.globl _main
_main:
  ret
