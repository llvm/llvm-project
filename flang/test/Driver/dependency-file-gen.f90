! Frontend (-fc1) dependency-file generation: list INCLUDE and #include files,
! derive a default target, and support -Eonly (dependencies only, no source).

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

! No -MT (and no -o): the target is derived from the input file name.
! RUN: %flang_fc1 -fsyntax-only %t/use-include.f90 -dependency-file %t/def.d
! RUN: FileCheck %s --input-file=%t/def.d --check-prefix=DEFAULT
! DEFAULT: use-include.o:
! DEFAULT: use-include.f90

! -Eonly: run the prescanner only, writing the dependencies to stdout with no
! preprocessed source.
! RUN: %flang_fc1 -Eonly %t/use-include.f90 -dependency-file - -MT custom.o 2>&1 \
! RUN:   | FileCheck %s --check-prefix=EONLY
! EONLY: custom.o:
! EONLY: use-include.f90

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
