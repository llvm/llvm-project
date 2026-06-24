! Ensure the frontend lists files brought in by an INCLUDE line or by a
! preprocessor #include directive in the generated dependency file.

! RUN: rm -rf %t && split-file %s %t

! INCLUDE statement.
! RUN: %flang_fc1 -fsyntax-only %t/use-include.f90 -dependency-file %t/inc.d -MT custom.o
! RUN: FileCheck %s --input-file=%t/inc.d --check-prefix=INCLUDE
! INCLUDE: custom.o:
! INCLUDE: use-include.f90
! INCLUDE: header.h

! Preprocessor #include directive.
! RUN: %flang_fc1 -fsyntax-only -cpp %t/use-cpp.F90 -dependency-file %t/cpp.d -MT custom.o
! RUN: FileCheck %s --input-file=%t/cpp.d --check-prefix=CPP
! CPP: custom.o:
! CPP: use-cpp.F90
! CPP: header.h

!--- header.h
      integer :: x

!--- use-include.f90
program test
  include 'header.h'
end program test

!--- use-cpp.F90
program test
#include "header.h"
end program test
