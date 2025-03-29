! RUN: %flang -### --target=aarch64-windows-msvc %S/Inputs/hello.f90 -v 2>&1 | FileCheck %s --check-prefixes=MSVC,MSVC-AARCH64
! RUN: %flang -### --target=i386-windows-msvc %S/Inputs/hello.f90 -v 2>&1 | FileCheck %s --check-prefixes=MSVC,MSVC-X86_32
! RUN: %flang -### --target=x86_64-windows-msvc %S/Inputs/hello.f90 -v 2>&1 | FileCheck %s --check-prefixes=MSVC,MSVC-X86_64

! MSVC: -fc1
! MSVC-SAME: -D_MSC_VER={{[0-9]*}}
! MSVC-SAME: -D_MSC_FULL_VER={{[0-9]*}}
! MSVC-AARCH64-SAME: -D_M_ARM64=1
! MSVC-X86_32-SAME: -D_M_IX86=600
! MSVC-X86_64-SAME: -D_M_X64=100
