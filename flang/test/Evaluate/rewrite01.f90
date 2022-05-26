! Test expression rewrites, in case where the expression cannot be
! folded to constant values.
! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

! Test rewrites of inquiry intrinsics with arguments whose shape depends
! on a function reference with non constant shape. The function reference
! must be retained.
module some_mod
contains
function returns_array(n, m)
  integer :: returns_array(10:n+10,10:m+10)
  returns_array = 0
end function

function returns_array_2(n)
  integer, intent(in) :: n
  integer :: returns_array_2(n)
  returns_array_2 = 0
end function

function returns_array_3()
  integer :: returns_array_3(7:46+2)
  returns_array_3 = 0
end function

subroutine ubound_test(x, n, m)
  integer :: x(n, m)
  integer :: y(0:n, 0:m) ! UBOUND could be 0 if n or m are < 0
  !CHECK: PRINT *, [INTEGER(4)::int(size(x,dim=1,kind=8),kind=4),int(size(x,dim=2,kind=8),kind=4)]
  print *, ubound(x)
  !CHECK: PRINT *, ubound(returns_array(n,m))
  print *, ubound(returns_array(n, m))
  !CHECK: PRINT *, ubound(returns_array(n,m),dim=1_4)
  print *, ubound(returns_array(n, m), dim=1)
  !CHECK: PRINT *, ubound(returns_array_2(m))
  print *, ubound(returns_array_2(m))
  !CHECK: PRINT *, 42_8
  print *, ubound(returns_array_3(), dim=1, kind=8)
  !CHECK: PRINT *, ubound(y)
  print *, ubound(y)
  !CHECK: PRINT *, ubound(y,1_4)
  print *, ubound(y, 1)
end subroutine

subroutine size_test(x, n, m)
  integer :: x(n, m)
  !CHECK: PRINT *, int(size(x,dim=1,kind=8)*size(x,dim=2,kind=8),kind=4)
  print *, size(x)
  !CHECK: PRINT *, size(returns_array(n,m))
  print *, size(returns_array(n, m))
  !CHECK: PRINT *, size(returns_array(n,m),dim=1_4)
  print *, size(returns_array(n, m), dim=1)
  !CHECK: PRINT *, size(returns_array_2(m))
  print *, size(returns_array_2(m))
  !CHECK: PRINT *, 42_8
  print *, size(returns_array_3(), kind=8)
end subroutine

subroutine shape_test(x, n, m)
  abstract interface
    function foo(n)
      integer, intent(in) :: n
      real, pointer :: foo(:,:)
    end function
  end interface
  procedure(foo), pointer :: pf
  integer :: x(n, m)
  !CHECK: PRINT *, [INTEGER(4)::int(size(x,dim=1,kind=8),kind=4),int(size(x,dim=2,kind=8),kind=4)]
  print *, shape(x)
  !CHECK: PRINT *, shape(returns_array(n,m))
  print *, shape(returns_array(n, m))
  !CHECK: PRINT *, shape(returns_array_2(m))
  print *, shape(returns_array_2(m))
  !CHECK: PRINT *, [INTEGER(8)::42_8]
  print *, shape(returns_array_3(), kind=8)
  !CHECK: PRINT *, 2_4
  print *, rank(pf(1))
end subroutine

subroutine lbound_test(x, n, m)
  integer :: x(n, m)
  integer :: y(0:n, 0:m) ! LBOUND could be 1 if n or m are < 0
  type t
    real, pointer :: p(:, :)
  end type
  type(t) :: a(10)
  !CHECK: PRINT *, [INTEGER(4)::1_4,1_4]
  print *, lbound(x)
  !CHECK: PRINT *, [INTEGER(4)::1_4,1_4]
  print *, lbound(returns_array(n, m))
  !CHECK: PRINT *, 1_4
  print *, lbound(returns_array(n, m), dim=1)
  !CHECK: PRINT *, 1_4
  print *, lbound(returns_array_2(m), dim=1)
  !CHECK: PRINT *, 1_4
  print *, lbound(returns_array_3(), dim=1)
  !CHECK: PRINT *, lbound(y)
  print *, lbound(y)
  !CHECK: PRINT *, lbound(y,1_4)
  print *, lbound(y, 1)
  !CHECK: PRINT *, lbound(a(1_8)%p,dim=1,kind=8)
  print *, lbound(a(1)%p, 1, kind=8)
end subroutine

!CHECK: len_test
subroutine len_test(a,b, c, d, e, n, m)
  character(*), intent(in) :: a
  character(10) :: b
  external b
  character(10), intent(in) :: c
  integer, intent(in) :: n, m
  character(n), intent(in) :: e

  !CHECK: PRINT *, int(a%len,kind=8)
  print *, len(a, kind=8)
  !CHECK: PRINT *, 5_4
  print *, len(a(1:5))
  !CHECK: PRINT *, 10_4
  print *, len(b(a))
  !CHECK: PRINT *, int(10_8+int(a%len,kind=8),kind=4)
  print *, len(b(a) // a)
  !CHECK: PRINT *, 10_4
  print *, len(c)
  !CHECK: PRINT *, len(c(int(i,kind=8):int(j,kind=8)))
  print *, len(c(i:j))
  !CHECK: PRINT *, 5_4
  print *, len(c(1:5))
  !CHECK: PRINT *, 10_4
  print *, len(b(c))
  !CHECK: PRINT *, 20_4
  print *, len(b(c) // c)
  !CHECK: PRINT *, 0_4
  print *, len(a(10:4))
  !CHECK: PRINT *, int(max(0_8,int(m,kind=8)-int(n,kind=8)+1_8),kind=4)
  print *, len(a(n:m))
  !CHECK: PRINT *, 10_4
  print *, len(b(a(n:m)))
  !CHECK: PRINT *, int(max(0_8,max(0_8,int(n,kind=8))-4_8+1_8),kind=4)
  print *, len(e(4:))
end subroutine len_test

end module
