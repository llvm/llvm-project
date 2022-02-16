# REQUIRES: aarch64, x86

# RUN: rm -rf %t && mkdir -p %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/native.o
# RUN: not %no-fatal-warnings-lld -arch x86_64 -lSystem %t/test.o -o /dev/null -arch_errors_fatal 2>&1 | FileCheck %s -DFILE=%t/test.o --check-prefix=CHECK-ERROR
# RUN: %no-fatal-warnings-lld -arch x86_64 -lSystem %t/test.o %t/native.o -o /dev/null 2>&1 | FileCheck %s -DFILE=%t/test.o --check-prefix=CHECK-WARNING
# RUN: %lld -arch arm64 -lSystem %t/test.o -arch_errors_fatal -o /dev/null
# CHECK-ERROR: error: {{.*}}[[FILE]] has architecture arm64 which is incompatible with target architecture x86_64
# CHECK-WARNING: warning: {{.*}}[[FILE]] has architecture arm64 which is incompatible with target architecture x86_64

# RUN: %lld -dylib -arch arm64 -platform_version macOS 10.14 10.15 -o %t/out.dylib %t/test.o

# RUN: not %lld -dylib -arch arm64 -platform_version iOS 9.0 11.0 %t/out.dylib \
# RUN:  -o /dev/null 2>&1 | FileCheck %s --check-prefix=DYLIB-PLAT
# DYLIB-PLAT: {{.*}}out.dylib has platform macOS, which is different from target platform iOS

# RUN: %lld -lSystem -dylib -arch arm64 -platform_version macOS 10.14.0 10.15.0 %t/out.dylib -o /dev/null

# RUN: %no-fatal-warnings-lld -lSystem -dylib -arch arm64 -platform_version macOS 10.13.0 10.15.0 %t/out.dylib \
# RUN:  -o /dev/null 2>&1 | FileCheck %s --check-prefix=DYLIB-VERSION
# DYLIB-VERSION: warning: {{.*}}out.dylib has version 10.14.0, which is newer than target minimum of 10.13.0

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos10.15.0 %s -o %t/test_x86.o

# RUN: not %lld %t/test_x86.o -lSystem -arch x86_64 -platform_version iOS 10.0 15.0 \
# RUN:  -o /dev/null 2>&1 | FileCheck %s --check-prefix=OBJ-PLAT
# OBJ-PLAT: {{.*}}test_x86.o has platform macOS, which is different from target platform iOS

# RUN: %lld %t/test_x86.o -lSystem -arch x86_64 -platform_version macOS 10.15.0 10.15.0 -o /dev/null

# RUN: %no-fatal-warnings-lld %t/test_x86.o -lSystem -arch x86_64 -platform_version macOS 10.14.0 10.15.0 \
# RUN:  -o /dev/null 2>&1 | FileCheck %s --check-prefix=OBJ-VERSION
# OBJ-VERSION: warning: {{.*}}test_x86.o has version 10.15.0, which is newer than target minimum of 10.14.0

## Test that simulators platforms are compat with their simulatees.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-ios14.0 %s -o %t/test_x86_ios.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-ios14.0-simulator %s -o %t/test_x86_ios_sim.o

# RUN: %lld -dylib -platform_version ios-simulator 14.0.0 14.0.0 %t/test_x86_ios.o -o /dev/null
# RUN: %lld -dylib -platform_version ios 14.0.0 14.0.0 %t/test_x86_ios_sim.o -o /dev/null

# RUN: not %lld -dylib  -platform_version watchos-simulator 14.0.0 14.0.0 %t/test_x86_ios.o \
# RUN:	-o /dev/null 2>&1 | FileCheck %s --check-prefix=CROSS-SIM
# CROSS-SIM: {{.*}}test_x86_ios.o has platform iOS, which is different from target platform watchOS Simulator
# RUN: not %lld -dylib  -platform_version watchos-simulator 14.0.0 14.0.0 %t/test_x86_ios_sim.o \
# RUN:	-o /dev/null 2>&1 | FileCheck %s --check-prefix=CROSS-SIM2
# CROSS-SIM2: {{.*}}test_x86_ios_sim.o has platform iOS Simulator, which is different from target platform watchOS Simulator

.globl _main
_main:
  ret
