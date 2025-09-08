! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
program main
  type t
    integer, allocatable :: ia(:)
  end type
  type(t) x
  integer, allocatable :: ja(:)
  allocate(ja(2:2))
  ja(2) = 2
  !CHECK: x=t(ia=(ja))
  x = t(+ja)            ! must be t(ia=(ja)), not t(ia=ja)
  print *, lbound(x%ia) ! must be 1, not 2
end
