# REQUIRES: x86, aarch64
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/x86-64-test.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t/arm64-test.o
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos %s -o %t/arm64-32-test.o
# RUN: %lld -arch x86_64 -platform_version macos 10.5.0 11.0 -o %t/x86-64-executable %t/x86-64-test.o
# RUN: %lld -arch arm64 -o %t/arm64-executable %t/arm64-test.o
# RUN: %lld-watchos -o %t/arm64-32-executable %t/arm64-32-test.o
# RUN: %lld -arch x86_64 -dylib -o %t/x86-64-dylib %t/x86-64-test.o
# RUN: %lld -arch arm64  -dylib -o %t/arm64-dylib %t/arm64-test.o
# RUN: %lld-watchos -dylib -o %t/arm64-32-dylib %t/arm64-32-test.o

# RUN: llvm-objdump --macho --private-header %t/x86-64-executable | FileCheck %s -DCPU=X86_64 -DCAPS=LIB64
# RUN: llvm-objdump --macho --private-header %t/arm64-executable | FileCheck %s -DCPU=ARM64 -DCAPS=0x00
# RUN: llvm-objdump --macho --private-header %t/arm64-32-executable | FileCheck %s --check-prefix=ARM64-32
# RUN: llvm-objdump --macho --private-header %t/x86-64-dylib | FileCheck %s -DCPU=X86_64 -DCAPS=0x00
# RUN: llvm-objdump --macho --private-header %t/arm64-dylib | FileCheck %s -DCPU=ARM64 -DCAPS=0x00
# RUN: llvm-objdump --macho --private-header %t/arm64-32-dylib | FileCheck %s --check-prefix=ARM64-32

# CHECK:      magic        cputype  cpusubtype  caps     filetype {{.*}} flags
# CHECK-NEXT: MH_MAGIC_64  [[CPU]]         ALL  [[CAPS]] {{.*}}          NOUNDEFS {{.*}} TWOLEVEL

# ARM64-32:      magic     cputype  cpusubtype  caps  filetype {{.*}} flags
# ARM64-32-NEXT: MH_MAGIC  ARM64_32         V8  0x00  {{.*}}          NOUNDEFS {{.*}} TWOLEVEL

.globl _main
_main:
  ret
