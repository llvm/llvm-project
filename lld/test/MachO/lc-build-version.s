# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

# RUN: %lld -platform_version macos 10.14 10.15 -o %t.macos-10-14 %t.o
# RUN: llvm-objdump --macho --all-headers %t.macos-10-14 | FileCheck %s --check-prefix=MACOS-10-14

# MACOS-10-14: cmd LC_BUILD_VERSION
# MACOS-10-14-NEXT: cmdsize 32
# MACOS-10-14-NEXT: platform macos
# MACOS-10-14-NEXT: sdk 10.15
# MACOS-10-14-NEXT: minos 10.14
# MACOS-10-14-NEXT: ntools 1
# MACOS-10-14-NEXT: tool lld
# MACOS-10-14-NEXT: version {{[0-9\.]+}}

# RUN: %lld -platform_version macos 10.13 10.15 -o %t.macos-10-13 %t.o
# RUN: llvm-objdump --macho --all-headers %t.macos-10-13 | FileCheck %s --check-prefix=MACOS-10-13

# MACOS-10-13: cmd LC_VERSION_MIN_MACOSX
# MACOS-10-13-NEXT: cmdsize 16
# MACOS-10-13-NEXT: version 10.13
# MACOS-10-13-NEXT: sdk 10.15

# RUN: %no-arg-lld -arch x86_64 -platform_version ios 12.0 10.15 -o %t.ios-12-0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.ios-12-0 | FileCheck %s --check-prefix=IOS-12-0
# RUN: %no-arg-lld -arch x86_64 -platform_version ios-simulator 13.0 10.15 -o %t.ios-sim-13-0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.ios-sim-13-0 | FileCheck %s --check-prefix=IOS-12-0

# IOS-12-0: cmd LC_BUILD_VERSION

# RUN: %no-arg-lld -arch x86_64 -platform_version ios 11.0 10.15 -o %t.ios-11-0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.ios-11-0 | FileCheck %s --check-prefix=IOS-11-0
# RUN: %no-arg-lld -arch x86_64 -platform_version ios-simulator 12.0 10.15 -o %t.ios-sim-12-0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.ios-sim-12-0 | FileCheck %s --check-prefix=IOS-11-0

# IOS-11-0: cmd LC_VERSION_MIN_IPHONEOS

# RUN: %no-arg-lld -arch x86_64 -platform_version tvos 12.0 10.15 -o %t.tvos-12-0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.tvos-12-0 | FileCheck %s --check-prefix=TVOS-12-0
# RUN: %no-arg-lld -arch x86_64 -platform_version tvos-simulator 13.0 10.15 -o %t.tvos-sim-13-0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.tvos-sim-13-0 | FileCheck %s --check-prefix=TVOS-12-0

# TVOS-12-0: cmd LC_BUILD_VERSION

# RUN: %no-arg-lld -arch x86_64 -platform_version tvos 11.0 10.15 -o %t.tvos-11-0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.tvos-11-0 | FileCheck %s --check-prefix=TVOS-11-0
# RUN: %no-arg-lld -arch x86_64 -platform_version tvos-simulator 12.0 10.15 -o %t.tvos-sim-12-0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.tvos-sim-12-0 | FileCheck %s --check-prefix=TVOS-11-0

# TVOS-11-0: cmd LC_VERSION_MIN_TVOS

# RUN: %no-arg-lld -arch x86_64 -platform_version watchos 5.0 10.15 -o %t.watchos-5-0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.watchos-5-0 | FileCheck %s --check-prefix=WATCHOS-5-0
# RUN: %no-arg-lld -arch x86_64 -platform_version watchos-simulator 6.0 10.15 -o %t.watchos-sim-6-0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.watchos-sim-6-0 | FileCheck %s --check-prefix=WATCHOS-5-0

# WATCHOS-5-0: cmd LC_BUILD_VERSION

# RUN: %no-arg-lld -arch x86_64 -platform_version watchos 4.0 10.15 -o %t.watchos-4-0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.watchos-4-0 | FileCheck %s --check-prefix=WATCHOS-4-0
# RUN: %no-arg-lld -arch x86_64 -platform_version watchos-simulator 5.0 10.15 -o %t.watchos-sim-5-0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.watchos-sim-5-0 | FileCheck %s --check-prefix=WATCHOS-4-0

# WATCHOS-4-0: cmd LC_VERSION_MIN_WATCHOS

# RUN: %no-arg-lld -arch x86_64 -platform_version xros 1.0 1.1 -o %t.xros-1-0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.xros-1-0 | FileCheck %s --check-prefix=XROS-1-0
# RUN: %no-arg-lld -arch x86_64 -platform_version xros-simulator 1.0 1.1 -o %t.xros-sim-1-0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.xros-sim-1-0 | FileCheck %s --check-prefix=XROS-1-0

# XROS-1-0: cmd LC_BUILD_VERSION

.text
.global _main
_main:
  mov $0, %eax
  ret
