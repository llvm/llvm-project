! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Just check some of the macros - if they are printed
!
! Check amd64-linux architecture
! RUN: %flang -dM -E -target x86_64-unknown-linux-gnu %s | FileCheck %s \
! RUN: --check-prefixes=CHECK-MACROS-AMD64,CHECK-MACROS-LINUX,CHECK-MACROS-COMMON-ACCPP
!
! Check amd64-linux architecture but from FPP module
! RUN: %flang -Hx,123,0x4000000 -dM -E -target x86_64-unknown-linux-gnu %s | FileCheck %s \
! RUN: --check-prefixes=CHECK-MACROS-AMD64,CHECK-MACROS-LINUX,CHECK-MACROS-COMMON-FPP
!
! Check aarch64-linux architecture
! RUN: %flang -dM -E -target aarch64-unknown-linux-gnu %s | FileCheck %s \
! RUN: --check-prefixes=CHECK-MACROS-AARCH64,CHECK-MACROS-LINUX,CHECK-MACROS-COMMON-ACCPP
!
! Check aarch64-linux architecture but from FPP module
! RUN: %flang -Hx,123,0x4000000 -dM -E -target aarch64-unknown-linux-gnu %s | FileCheck %s \
! RUN: --check-prefixes=CHECK-MACROS-AARCH64,CHECK-MACROS-LINUX,CHECK-MACROS-COMMON-FPP
!
! Check aarch64-windows architecture
! RUN: %flang -dM -E -target aarch64-unknown-windows-gnu %s | FileCheck %s \
! RUN: --check-prefixes=CHECK-MACROS-AARCH64,CHECK-MACROS-WINDOWS,CHECK-MACROS-COMMON-ACCPP
!
! Check aarch64-windows architecture but from FPP module
! RUN: %flang -Hx,123,0x4000000 -dM -E -target aarch64-unknown-windows-gnu %s | FileCheck %s \
! RUN: --check-prefixes=CHECK-MACROS-AARCH64,CHECK-MACROS-WINDOWS,CHECK-MACROS-COMMON-FPP
!
! Check amd64-windows architecture
! RUN: %flang -dM -E -target x86_64-unknown-windows-gnu %s | FileCheck %s \
! RUN: --check-prefixes=CHECK-MACROS-AMD64,CHECK-MACROS-WINDOWS,CHECK-MACROS-COMMON-ACCPP
!
! Check amd64-windows architecture but from FPP module
! RUN: %flang -Hx,123,0x4000000 -dM -E -target x86_64-unknown-windows-gnu %s | FileCheck %s \
! RUN: --check-prefixes=CHECK-MACROS-AMD64,CHECK-MACROS-WINDOWS,CHECK-MACROS-COMMON-FPP
!
! Check i386-windows architecture
! RUN: %flang -dM -E -target i386-unknown-windows-gnu %s | FileCheck %s \
! RUN: --check-prefixes=CHECK-MACROS-I386,CHECK-MACROS-WINDOWS,CHECK-MACROS-COMMON-ACCPP
!
! Check i386-windows architecture but from FPP module
! RUN: %flang -Hx,123,0x4000000 -dM -E -target i386-unknown-windows-gnu %s | FileCheck %s \
! RUN: --check-prefixes=CHECK-MACROS-I386,CHECK-MACROS-WINDOWS,CHECK-MACROS-COMMON-FPP
!
! CHECK-MACROS-AMD64: __amd64__
! CHECK-MACROS-AMD64: __x86_64__

! CHECK-MACROS-AARCH64: __AARCH64EL__ 
! CHECK-MACROS-AARCH64: __aarch64__ 
! CHECK-MACROS-AARCH64: __ARM_ARCH
! CHECK-MACROS-AARCH64: __ARM_ARCH_PROFILE
! CHECK-MACROS-AARCH64: __ARM_64BIT_STATE

! CHECK-MACROS-I386: __i386__

! CHECK-MACROS-LINUX: __linux
! CHECK-MACROS-LINUX: __gnu_linux__

! CHECK-MACROS-WINDOWS: _WIN32
! CHECK-MACROS-WINDOWS: _MSVCRT__
! CHECK-MACROS-WINDOWS: __declspec
! CHECK-MACROS-WINDOWS: __stdcall
! CHECK-MACROS-WINDOWS: __thiscall
! CHECK-MACROS-WINDOWS: __pascal

! CHECK-MACROS-COMMON-ACCPP: __FLANG
! CHECK-MACROS-COMMON-ACCPP: __FLANG_MAJOR__ 
! CHECK-MACROS-COMMON-ACCPP: __FLANG_MINOR__
! CHECK-MACROS-COMMON-ACCPP: __FLANG_PATCHLEVEL__

! CHECK-MACROS-COMMON-FPP: __PGIC__
! CHECK-MACROS-COMMON-FPP: __PGIC_MINOR__ 
! CHECK-MACROS-COMMON-FPP: __PGIF90__
! CHECK-MACROS-COMMON-FPP: __PGIF90_MINOR__
! CHECK-MACROS-COMMON-FPP: __PGIC_PATCHLEVEL__
! CHECK-MACROS-COMMON-FPP: __PGIF90_PATCHLEVEL__

program hello
  print *, "hello world"
end program hello
