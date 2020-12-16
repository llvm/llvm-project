! RUN: %S/test_folding.sh %s %t %f18
! Test implementations of STORAGE_SIZE() and SIZEOF() as expression rewrites
module m1
  type :: t1
    real :: a(2,3)
    character*5 :: c(3)
  end type
  type :: t2(k)
    integer, kind :: k
    type(t1) :: a(k)
  end type
  type(t2(2)) :: a(2)
  integer, parameter :: ss1 = storage_size(a(1)%a(1)%a)
  integer, parameter :: sz1 = sizeof(a(1)%a(1)%a)
  integer, parameter :: ss2 = storage_size(a(1)%a(1)%c)
  integer, parameter :: sz2 = sizeof(a(1)%a(1)%c)
  integer, parameter :: ss3 = storage_size(a(1)%a)
  integer, parameter :: sz3 = sizeof(a(1)%a)
  integer, parameter :: ss4 = storage_size(a)
  integer, parameter :: sz4 = sizeof(a)
  logical, parameter :: test_ss = all([ss1,ss2,ss3,ss4]==[32, 40, 320, 640])
  logical, parameter :: test_sz = all([sz1,sz2,sz3,sz4]==[24, 15, 80, 160])
end module
