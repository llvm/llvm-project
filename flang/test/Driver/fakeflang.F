! This is how CMake probes the compiler for CMAKE_Fortran_COMPILER_ID etc.

! RUN: rm -rf %t
! RUN: mkdir -p %t
! RUN: cp %s %t/CMakeFortranCompilerId.F
! RUN: cp %s %t/compiletest.F
! RUN: cd %t

! RUN: fakeflang -v -c -cpp -E --target=x86_64-unknown-linux-gnu CMakeFortranCompilerId.F 2>&1 | FileCheck %s --check-prefix=FAKEWARNING
! RUN: FileCheck %s --input-file=a.out --check-prefixes=CHECK,GNU
! RUN: rm a.out

! RUN: fakeflang -v -c -cpp -E --target=aarch64-linux-gnu CMakeFortranCompilerId.F 2>&1 | FileCheck %s --check-prefix=FAKEWARNING
! RUN: FileCheck %s --input-file=a.out --check-prefixes=CHECK,GNU
! RUN: rm a.out

! RUN: fakeflang -v -c -cpp -E --target=x86_64-pc-windows-msvc CMakeFortranCompilerId.F 2>&1 | FileCheck %s --check-prefix=FAKEWARNING
! RUN: FileCheck %s --input-file=a.out --check-prefixes=CHECK,MSVC --match-full-lines
! RUN: rm a.out

! FLANG_BOOTSTRAP_PROBE forces probe mode
! RUN: env FLANG_BOOTSTRAP_PROBE=1 fakeflang -v -c --target=x86_64-unknown-linux-gnu compiletest.F 2>&1 | FileCheck %s --check-prefix=FAKEWARNING
! RUN: FileCheck %s --input-file=a.out --check-prefixes=CHECK,GNU
! RUN: rm a.out

! When not probing, compile everything normally
! RUN: fakeflang -c compiletest.F -o compiletest.F.o 2>&1 | FileCheck %s --check-prefix=NOFAKE --allow-empty
! RUN: test -e compiletest.F.o


! FAKEWARNING: fakeflang: warning: Emulating flang for compiler probing
! NOFAKE-NOT: fakeflang

#ifdef __fakeflang_probe__
! CHECK: dispatcher=1
dispatcher=__fakeflang_probe__

! CHECK: CMAKE_Fortran_COMPILER_ID=1
CMAKE_Fortran_COMPILER_ID=__flang__

! __GNUC__ is not defined
! CHECK: __GNUC__
__GNUC__

! CHECK: CMAKE_Fortran_COMPILER_VERSION={{[0-9]+}} . {{[0-9]+}} . {{[0-9]+}}
CMAKE_Fortran_COMPILER_VERSION=__flang_major__.__flang_minor__.__flang_patchlevel__

! GNU: CMAKE_Fortran_SIMULATE_ID=_MSC_VER
! MSVC: CMAKE_Fortran_SIMULATE_ID={{[0-9]+}}
CMAKE_Fortran_SIMULATE_ID=_MSC_VER

! GNU: CMAKE_Fortran_PLATFORM_ID=_WIN32
! MSVC: CMAKE_Fortran_PLATFORM_ID=1
CMAKE_Fortran_PLATFORM_ID=_WIN32
#else
      program helloworld
      end program helloworld
#endif
