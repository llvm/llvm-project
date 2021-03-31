# REQUIRES: aarch64, x86

# RUN: rm -rf %t && mkdir -p %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t/test.o
# RUN: not %lld -arch x86_64 -lSystem %t/test.o -o /dev/null 2>&1 | FileCheck %s -DFILE=%t/test.o
# CHECK: error: {{.*}}[[FILE]] has architecture arm64 which is incompatible with target architecture x86_64

# RUN: %lld -dylib  -arch arm64 -platform_version macOS 9.0 11.0 -o %t/out.dylib %t/test.o

# RUN: not %lld -dylib -arch arm64 -platform_version iOS 9.0 11.0  %t/out.dylib  \
# RUN:  -o /dev/null 2>&1 | FileCheck %s --check-prefix=DYLIB-PLAT
# DYLIB-PLAT: {{.*}}out.dylib has platform macOS, which is different from target platform iOS

# RUN: not %lld -dylib -arch arm64 -platform_version macOS 14.0 15.0  %t/out.dylib  \
# RUN:  -o /dev/null 2>&1 | FileCheck %s --check-prefix=DYLIB-VERSION
# DYLIB-VERSION: {{.*}}out.dylib has version 9.0.0, which is incompatible with target version of 14.0

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos10.15.0 %s -o %t/test_x86.o

# RUN: not %lld %t/test_x86.o -lSystem -arch x86_64 -platform_version iOS 10.0 15.0 \
# RUN:  -o /dev/null 2>&1 | FileCheck %s --check-prefix=OBJ-PLAT
# OBJ-PLAT: {{.*}}test_x86.o has platform macOS, which is different from target platform iOS

# RUN: not %lld %t/test_x86.o -lSystem -arch x86_64 -platform_version macOS 14.0 15.0 \
# RUN:  -o /dev/null 2>&1 | FileCheck %s --check-prefix=OBJ-VERSION
# OBJ-VERSION: {{.*}}test_x86.o has version 10.15.0, which is incompatible with target version of 14.0

.globl _main
_main:
  ret
