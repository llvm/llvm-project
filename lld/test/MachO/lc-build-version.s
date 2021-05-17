# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

# RUN: %lld -platform_version macos 10.14 10.15 -o %t.macos_10_14 %t.o
# RUN: llvm-objdump --macho --all-headers %t.macos_10_14 | FileCheck %s --check-prefix=MACOS_10_14

# MACOS_10_14: cmd LC_BUILD_VERSION
# MACOS_10_14-NEXT: cmdsize 32
# MACOS_10_14-NEXT: platform macos
# MACOS_10_14-NEXT: sdk 10.15
# MACOS_10_14-NEXT: minos 10.14
# MACOS_10_14-NEXT: ntools 1
# MACOS_10_14-NEXT: tool ld
# MACOS_10_14-NEXT: version {{[0-9\.]+}}

# RUN: %lld -platform_version macos 10.13 10.15 -o %t.macos_10_13 %t.o
# RUN: llvm-objdump --macho --all-headers %t.macos_10_13 | FileCheck %s --check-prefix=MACOS_10_13

# MACOS_10_13: cmd LC_VERSION_MIN_MACOSX
# MACOS_10_13-NEXT: cmdsize 16
# MACOS_10_13-NEXT: version 10.13
# MACOS_10_13-NEXT: sdk 10.15

# RUN: %lld -platform_version ios 12.0 10.15 -o %t.ios_12_0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.ios_12_0 | FileCheck %s --check-prefix=IOS_12_0
# RUN: %lld -platform_version ios-simulator 13.0 10.15 -o %t.ios_sim_13_0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.ios_sim_13_0 | FileCheck %s --check-prefix=IOS_12_0

# IOS_12_0: cmd LC_BUILD_VERSION

# RUN: %lld -platform_version ios 11.0 10.15 -o %t.ios_11_0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.ios_11_0 | FileCheck %s --check-prefix=IOS_11_0
# RUN: %lld -platform_version ios-simulator 12.0 10.15 -o %t.ios_sim_12_0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.ios_sim_12_0 | FileCheck %s --check-prefix=IOS_11_0

# IOS_11_0: cmd LC_VERSION_MIN_IPHONEOS

# RUN: %lld -platform_version tvos 12.0 10.15 -o %t.tvos_12_0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.tvos_12_0 | FileCheck %s --check-prefix=TVOS_12_0
# RUN: %lld -platform_version tvos-simulator 13.0 10.15 -o %t.tvos_sim_13_0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.tvos_sim_13_0 | FileCheck %s --check-prefix=TVOS_12_0

# TVOS_12_0: cmd LC_BUILD_VERSION

# RUN: %lld -platform_version tvos 11.0 10.15 -o %t.tvos_11_0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.tvos_11_0 | FileCheck %s --check-prefix=TVOS_11_0
# RUN: %lld -platform_version tvos-simulator 12.0 10.15 -o %t.tvos_sim_12_0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.tvos_sim_12_0 | FileCheck %s --check-prefix=TVOS_11_0

# TVOS_11_0: cmd LC_VERSION_MIN_TVOS

# RUN: %lld -platform_version watchos 5.0 10.15 -o %t.watchos_5_0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.watchos_5_0 | FileCheck %s --check-prefix=WATCHOS_5_0
# RUN: %lld -platform_version watchos-simulator 6.0 10.15 -o %t.watchos_sim_6_0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.watchos_sim_6_0 | FileCheck %s --check-prefix=WATCHOS_5_0

# WATCHOS_5_0: cmd LC_BUILD_VERSION

# RUN: %lld -platform_version watchos 4.0 10.15 -o %t.watchos_4_0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.watchos_4_0 | FileCheck %s --check-prefix=WATCHOS_4_0
# RUN: %lld -platform_version watchos-simulator 5.0 10.15 -o %t.watchos_sim_5_0 %t.o
# RUN: llvm-objdump --macho --all-headers %t.watchos_sim_5_0 | FileCheck %s --check-prefix=WATCHOS_4_0

# WATCHOS_4_0: cmd LC_VERSION_MIN_WATCHOS

.text
.global _main
_main:
  mov $0, %eax
  ret
