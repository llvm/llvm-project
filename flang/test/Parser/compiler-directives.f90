! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

! Test that compiler directives can appear in various places.

#define PROC(KIND) \
  interface; integer(KIND) function foo(a); \
    integer(KIND), intent(in) :: a; \
    !dir$ ignore_tkr a; \
  end; end interface

!dir$ integer
module m
  !dir$ integer
  use iso_fortran_env
  !dir$ integer
  implicit integer(a-z)
  !dir$ integer
  !dir$ integer=64
  !dir$ integer = 64
  !dir$  integer = 64
  PROC(4)
  !dir$ optimize:1
  !dir$ optimize : 1
  !dir$ loop count (10000)
  !dir$ loop count (1, 500, 5000, 10000)
  type stuff
     real(8), allocatable :: d(:)
     !dir$  align : 1024 :: d
  end type stuff
end

subroutine vector_always
  !dir$ vector always
  ! CHECK: !DIR$ VECTOR ALWAYS
  do i=1,10
  enddo
end subroutine

subroutine unroll
  !dir$ unroll
  ! CHECK: !DIR$ UNROLL
  do i=1,10
  enddo
  !dir$ unroll 2
  ! CHECK: !DIR$ UNROLL 2
  do i=1,10
  enddo
end subroutine

subroutine unroll_and_jam
  !dir$ unroll_and_jam
  ! CHECK: !DIR$ UNROLL_AND_JAM
  do i=1,10
  enddo
  !dir$ unroll_and_jam 2
  ! CHECK: !DIR$ UNROLL_AND_JAM 2
  do i=1,10
  enddo
end subroutine
