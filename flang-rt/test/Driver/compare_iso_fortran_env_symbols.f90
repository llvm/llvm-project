! UNSUPPORTED: offload-cuda, system-windows
! REQUIRES: fortran-modules

! RUN: %flang -c -funsigned %S/../../lib/runtime/iso_fortran_env_impl.f90 -o %t.f90.o

! Extract defined symbol names and sizes from the Fortran object and the
! already-built runtime library (which was compiled with the correct CMake flags).
! RUN: llvm-nm --defined-only --format=posix --print-size %t.f90.o \
! RUN:   | grep '_QMiso_fortran_env_impl' | cut -d' ' -f1,4 | sort -u > %t.f90.syms
! RUN: llvm-nm --defined-only --format=posix --print-size "%libdir"/libflang_rt.runtime.* \
! RUN:   | grep '_QMiso_fortran_env_impl' | cut -d' ' -f1,4 | sort -u > %t.cpp.syms

! The two symbol lists must be identical.
! RUN: diff %t.f90.syms %t.cpp.syms
