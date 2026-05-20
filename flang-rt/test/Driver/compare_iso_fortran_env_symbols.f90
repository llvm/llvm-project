! UNSUPPORTED: offload-cuda

! RUN: %flang -c -funsigned %include/../module/iso_fortran_env_impl.f90 -o %t.f90.o
! RUN: %clang -x c++ -std=c++17 -c -I "%include" %S/../../lib/runtime/iso_fortran_env_impl.cpp -o %t.cpp.o

! Extract defined symbol names and sizes from both objects.
! RUN: llvm-nm --defined-only --format=posix --print-size %t.f90.o \
! RUN:   | grep '_QMiso_fortran_env_impl' | cut -d' ' -f1,4 | sort > %t.f90.syms
! RUN: llvm-nm --defined-only --format=posix --print-size %t.cpp.o \
! RUN:   | grep '_QMiso_fortran_env_impl' | cut -d' ' -f1,4 | sort > %t.cpp.syms

! The two symbol lists must be identical.
! RUN: diff %t.f90.syms %t.cpp.syms
