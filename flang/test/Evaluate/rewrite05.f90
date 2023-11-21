! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
program main
  type t
    integer, allocatable :: component(:)
  end type
  type(t) :: x
  call init(10)
  !CHECK: PRINT *, [INTEGER(4)::int(lbound(x%component,dim=1,kind=8),kind=4)]
  print *, lbound(x%component)
  !CHECK: PRINT *, [INTEGER(4)::int(size(x%component,dim=1,kind=8)+lbound(x%component,dim=1,kind=8)-1_8,kind=4)]
  print *, ubound(x%component)
  !CHECK: PRINT *, int(size(x%component,dim=1,kind=8),kind=4)
  print *, size(x%component)
  !CHECK: PRINT *, 4_8*size(x%component,dim=1,kind=8)
  print *, sizeof(x%component)
  !CHECK: PRINT *, 1_4
  print *, lbound(iota(10), 1)
  !CHECK: PRINT *, ubound(iota(10_4),1_4)
  print *, ubound(iota(10), 1)
  !CHECK: PRINT *, size(iota(10_4))
  print *, size(iota(10))
  !CHECK: PRINT *, sizeof(iota(10_4))
  print *, sizeof(iota(10))
 contains
  function iota(n) result(result)
    integer, intent(in) :: n
    integer, allocatable :: result(:)
    result = [(j,j=1,n)]
  end
  subroutine init(n)
    integer, intent(in) :: n
    allocate(x%component(0:n-1))
  end
end
