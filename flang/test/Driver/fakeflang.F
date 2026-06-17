! This is how CMake probes the compiler for CMAKE_Fortran_COMPILER_ID etc.

! RUN: rm -rf %t
! RUN: mkdir -p %t
! RUN: cd %t

! RUN: fakeflang -v -c --target=x86_64-unknown-linux-gnu %s
! RUN: FileCheck %s --input-file=a.out --check-prefixes=CHECK,GNU
! RUN: rm a.out

! RUN: fakeflang -v -c --target=x86_64-pc-windows-msvc %s
! RUN: FileCheck %s --input-file=a.out --check-prefixes=CHECK,MSVC --match-full-lines
! RUN: rm a.out


! CHECK: CMAKE_Fortran_COMPILER_ID=1
CMAKE_Fortran_COMPILER_ID=__flang__

! CHECK: CMAKE_Fortran_COMPILER_VERSION={{[0-9]+}} . {{[0-9]+}} . {{[0-9]+}}
CMAKE_Fortran_COMPILER_VERSION=__flang_major__.__flang_minor__.__flang_patchlevel__

! GNU: CMAKE_Fortran_SIMULATE_ID=_MSC_VER
! MSVC: CMAKE_Fortran_SIMULATE_ID={{[0-9]+}}
CMAKE_Fortran_SIMULATE_ID=_MSC_VER

! GNU: CMAKE_Fortran_PLATFORM_ID=_WIN32
! MSVC: CMAKE_Fortran_PLATFORM_ID=1
CMAKE_Fortran_PLATFORM_ID=_WIN32
