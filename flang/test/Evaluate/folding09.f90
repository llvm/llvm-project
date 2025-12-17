! RUN: %python %S/test_folding.py %s %flang_fc1
! Test folding of IS_CONTIGUOUS

module m
  real, target :: hosted(2)
  integer, parameter :: cst(2,2) = reshape([1, 2, 3, 4], shape(cst))
  integer, parameter :: empty_cst(2,0) = reshape([1], shape(empty_cst))
  integer :: n, m
  logical, parameter :: test_param1 = is_contiguous(cst(:,1))
  logical, parameter :: test_param2 = is_contiguous(cst(1,:))
  logical, parameter :: test_param3 = is_contiguous(cst(:,n))
  logical, parameter :: test_param4 = is_contiguous(cst(n,:))
  logical, parameter :: test_param5 = is_contiguous(empty_cst(n,-1:n:2))
 contains
  function f()
    real, pointer, contiguous :: f(:)
    f => hosted
  end function
  subroutine test(arr1, arr2, arr3, mat, alloc, alch)
    real, intent(in) :: arr1(:), arr2(10), mat(10, 10)
    real, intent(in), contiguous :: arr3(:)
    real, allocatable :: alloc(:)
    real :: scalar
    character(5) charr(5)
    character(1) char1(5)
    character(0) char0(5)
    character(*) alch(5)
    integer(kind=merge(1,-1,       is_contiguous(0)))               t01
    integer(kind=merge(1,-1,       is_contiguous(scalar)))          t02
    integer(kind=merge(1,-1,       is_contiguous(scalar + scalar))) t03
    integer(kind=merge(1,-1,       is_contiguous([0, 1, 2])))       t04
    integer(kind=merge(1,-1,       is_contiguous(arr1 + 1.0)))      t05
    integer(kind=merge(1,-1,       is_contiguous(arr2)))            t06
    integer(kind=merge(1,-1,       is_contiguous(mat)))             t07
    integer(kind=merge(1,-1,       is_contiguous(mat(1:10,1))))     t08
    integer(kind=merge(1,-1,       is_contiguous(arr2(1:10:1))))    t09
    integer(kind=merge(1,-1, .not. is_contiguous(arr2(1:10:2))))    t10
    integer(kind=merge(1,-1,       is_contiguous(arr3)))            t11
    integer(kind=merge(1,-1, .not. is_contiguous(arr3(1:10:2))))    t12
    integer(kind=merge(1,-1,       is_contiguous(f())))             t13
    integer(kind=merge(1,-1,       is_contiguous(alloc)))           t14
    integer(kind=merge(1,-1,       is_contiguous(charr(:)(:))))     t15
    integer(kind=merge(1,-1,       is_contiguous(charr(1)(2:3))))   t16
    integer(kind=merge(1,-1,       is_contiguous(charr(:)(1:))))    t17
    integer(kind=merge(1,-1,       is_contiguous(charr(:)(3:2))))   t18
    integer(kind=merge(1,-1,       is_contiguous(charr(:)(1:5))))   t19
    integer(kind=merge(1,-1, .not. is_contiguous(charr(:)(1:4))))   t20
    integer(kind=merge(1,-1,       is_contiguous(char1(:)(n:m))))   t21
    integer(kind=merge(1,-1, .not. is_contiguous(char1(1:5:2)(n:m)))) t22
    integer(kind=merge(1,-1,       is_contiguous(char0(:)(n:m))))   t23
    integer(kind=merge(1,-1,       is_contiguous(char0(1:5:2)(n:m)))) t24
    integer(kind=merge(1,-1,       is_contiguous(alch(:)(:))))      t25
    associate (x => arr2)
      block
        integer(kind=merge(1,-1,is_contiguous(x))) n
      end block
    end associate
    associate (x => arr2(1:10:2))
      block
        integer(kind=merge(1,-1,.not. is_contiguous(x))) n
      end block
    end associate
    associate (x => arr3)
      block
        integer(kind=merge(1,-1,is_contiguous(x))) n
      end block
    end associate
    associate (x => arr3(1:10:2))
      block
        integer(kind=merge(1,-1,.not. is_contiguous(x))) n
      end block
    end associate
  end subroutine
  subroutine test2(x, vec)
    type t
      integer :: i
    end type
    type(t) :: x(100)
    integer(8) :: vec(10)
    integer(kind=merge(1,-1, .not. is_contiguous(x(1:50:2)%i)))    t01
    integer(kind=merge(1,-1, .not. is_contiguous(x(vec)%i)))       t02
  end subroutine
end module
