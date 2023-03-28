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
  character(*) :: b
  external b
  character(10), intent(in) :: c
  character(10) :: d
  external d
  integer, intent(in) :: n, m
  character(n), intent(in) :: e
  character(5), parameter :: cparam = "abc  "
  interface
     function fun1(L)
       character(L) :: fun1
       integer :: L
     end function fun1
  end interface
  interface
     function mofun(L)
       character(L) :: mofun
       integer, intent(in) :: L
     end function mofun
  end interface

  !CHECK: PRINT *, int(int(a%len,kind=8),kind=4)
  print *, len(a)
  !CHECK: PRINT *, 5_4
  print *, len(a(1:5))
  !CHECK: PRINT *, len(b(a))
  print *, len(b(a))
  !CHECK: PRINT *, len(b(a)//a)
  print *, len(b(a) // a)
  !CHECK: PRINT *, 10_4
  print *, len(c)
  !CHECK: PRINT *, len(c(int(i,kind=8):int(j,kind=8)))
  print *, len(c(i:j))
  !CHECK: PRINT *, 5_4
  print *, len(c(1:5))
  !CHECK: PRINT *, 10_4
  print *, len(d(c))
  !CHECK: PRINT *, 20_4
  print *, len(d(c) // c)
  !CHECK: PRINT *, 0_4
  print *, len(a(10:4))
  !CHECK: PRINT *, int(max(0_8,int(m,kind=8)-int(n,kind=8)+1_8),kind=4)
  print *, len(a(n:m))
  !CHECK: PRINT *, len(b(a(int(n,kind=8):int(m,kind=8))))
  print *, len(b(a(n:m)))
  !CHECK: PRINT *, int(max(0_8,max(0_8,int(n,kind=8))-4_8+1_8),kind=4)
  print *, len(e(4:))
  !CHECK: PRINT *, len(fun1(n-m))
  print *, len(fun1(n-m))
  !CHECK: PRINT *, len(mofun(m+1_4))
  print *, len(mofun(m+1))
  !CHECK: PRINT *, 3_4
  print *, len(trim(cparam))
  !CHECK: PRINT *, len(trim(c))
  print *, len(trim(c))
  !CHECK: PRINT *, 40_4
  print *, len(repeat(c, 4))
  !CHECK: PRINT *, len(repeat(c,int(i,kind=8)))
  print *, len(repeat(c, i))
end subroutine len_test

!CHECK-LABEL: associate_tests
subroutine associate_tests(p)
  real, pointer :: p(:)
  real :: a(10:20)
  interface
    subroutine may_change_p_bounds(p)
      real, pointer :: p(:)
    end subroutine
  end interface
  associate(x => p)
    call may_change_p_bounds(p)
    !CHECK: PRINT *, lbound(x,dim=1,kind=8), size(x,dim=1,kind=8)+lbound(x,dim=1,kind=8)-1_8, size(x,dim=1,kind=8)
    print *, lbound(x, 1, kind=8), ubound(x, 1, kind=8), size(x, 1, kind=8)
  end associate
  associate(x => p+1)
    call may_change_p_bounds(p)
    !CHECK: PRINT *, 1_8, size(x,dim=1,kind=8), size(x,dim=1,kind=8)
    print *, lbound(x, 1, kind=8), ubound(x, 1, kind=8), size(x, 1, kind=8)
  end associate
  associate(x => a)
    !CHECK: PRINT *, 10_8, 20_8, 11_8
    print *, lbound(x, 1, kind=8), ubound(x, 1, kind=8), size(x, 1, kind=8)
  end associate
  associate(x => a+42.)
    !CHECK: PRINT *, 1_8, 11_8, 11_8
    print *, lbound(x, 1, kind=8), ubound(x, 1, kind=8), size(x, 1, kind=8)
  end associate
end subroutine

!CHECK-LABEL: array_constructor
subroutine array_constructor(a, u, v, w, x, y, z)
  real :: a(4)
  integer :: u(:), v(1), w(2), x(4), y(4), z(2, 2)
  interface
    function return_allocatable()
     real, allocatable :: return_allocatable(:)
    end function
  end interface
  !CHECK: PRINT *, size([REAL(4)::return_allocatable(),return_allocatable()])
  print *, size([return_allocatable(), return_allocatable()])
  !CHECK: PRINT *, [INTEGER(4)::x+y]
  print *, (/x/) + (/y/)
  !CHECK: PRINT *, [INTEGER(4)::x]+[INTEGER(4)::z]
  print *, (/x/) + (/z/)
  !CHECK: PRINT *, [INTEGER(4)::x+y,x+y]
  print *, (/x, x/) + (/y, y/)
  !CHECK: PRINT *, [INTEGER(4)::x,x]+[INTEGER(4)::x,z]
  print *, (/x, x/) + (/x, z/)
  !CHECK: PRINT *, [INTEGER(4)::x,w,w]+[INTEGER(4)::w,w,x]
  print *, (/x, w, w/) + (/w, w, x/)
  !CHECK: PRINT *, [INTEGER(4)::x]+[INTEGER(4)::1_4,2_4,3_4,4_4]
  print *, (/x/) + (/1, 2, 3, 4/)
  !CHECK: PRINT *, [INTEGER(4)::v]+[INTEGER(4)::1_4]
  print *, (/v/) + (/1/)
  !CHECK: PRINT *, [INTEGER(4)::x]+[INTEGER(4)::u]
  print *, (/x/) + (/u/)
  !CHECK: PRINT *, [INTEGER(4)::u]+[INTEGER(4)::u]
  print *, (/u/) + (/u/)
  !CHECK: PRINT *, [REAL(4)::a**x]
  print *, (/a/) ** (/x/)
  !CHECK: PRINT *, [REAL(4)::a]**[INTEGER(4)::z]
  print *, (/a/) ** (/z/)
end subroutine

!CHECK-LABEL: array_ctor_implied_do_index
subroutine array_ctor_implied_do_index(x, j)
  integer :: x(:)
  integer(8) :: j
  !CHECK: PRINT *, size([INTEGER(4)::(x(1_8:i:1_8),INTEGER(8)::i=1_8,2_8,1_8)])
  print *, size([(x(1:i), integer(8)::i=1,2)])
  !CHECK: PRINT *, int(0_8+2_8*(0_8+max((j-1_8+1_8)/1_8,0_8)),kind=4)
  print *, size([(x(1:j), integer(8)::i=1,2)])
end subroutine

end module
