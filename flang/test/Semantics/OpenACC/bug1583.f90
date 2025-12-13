! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenacc
!DEF: /m Module
module m
 !DEF: /m/t PUBLIC DerivedType
 type :: t
  !DEF: /m/t/c ALLOCATABLE ObjectEntity REAL(4)
  real, allocatable :: c(:)
 end type
contains
 !DEF: /m/sub PUBLIC (Subroutine) Subprogram
 !DEF: /m/sub/v ObjectEntity TYPE(t)
 subroutine sub (v)
  !REF: /m/t
  !REF: /m/sub/v
  type(t) :: v
!$acc host_data use_device(v%c)
  !DEF: /foo EXTERNAL (Subroutine) ProcEntity
  !REF: /m/sub/v
  !REF: /m/t/c
  call foo(v%c)
!$acc end host_data
 end subroutine
end module
