!RUN: %python %S/test_errors.py %s %flang_fc1 -Werror -Wpass-global-variable
module explicit_test_mod
  implicit none (type, external)
  integer :: i1
  integer :: i2(1)
  integer :: i3(3)
  integer, allocatable :: ia(:)

  real :: x1, y1
  real :: x2, y2
  real :: z, z2
  common /xy1/ x1, y1(1)
  common /xy2/ x2(1), y2
  common /fm/ z(1)
  common /fm_bad/ z2(5)
contains
  subroutine pass_int(i)
    integer, intent(inout) :: i
  end subroutine pass_int
  subroutine pass_int_1d(i)
    integer, intent(inout) :: i(*)
  end subroutine pass_int_1d
  subroutine pass_real(r)
    real, intent(inout) :: r
  end subroutine pass_real
  subroutine pass_real_1d(r)
    real, intent(inout) :: r(*)
  end subroutine pass_real_1d
  subroutine explicit_test(n)
    integer, intent(in) :: n

    !WARNING: Passing global variable 'i1' from MODULE 'explicit_test_mod' as function argument [-Wpass-global-variable]
    call pass_int(i1)               !< warn:    basic type
    call pass_int(i2(1))            !< ok:      shape == [1]
    call pass_int(i2(n))            !< ok:      shape == [1]
    !WARNING: Passing global variable 'i3' from MODULE 'explicit_test_mod' as function argument [-Wpass-global-variable]
    call pass_int(i3(1))            !< warn:    shape /= [1]
    !WARNING: Passing global variable 'i3' from MODULE 'explicit_test_mod' as function argument [-Wpass-global-variable]
    call pass_int(i3(n))            !< warn:    shape /= [1]
    !WARNING: Passing global variable 'i2' from MODULE 'explicit_test_mod' as function argument [-Wpass-global-variable]
    call pass_int_1d(i2)            !< warn:    whole array is passed
    call pass_int_1d(i2(n:n+3))     !< ok:      subrange of array
    !WARNING: Passing global variable 'i3' from MODULE 'explicit_test_mod' as function argument [-Wpass-global-variable]
    call pass_int_1d(i3)            !< warn:    shape /= [1]
    !WARNING: Passing global variable 'i3' from MODULE 'explicit_test_mod' as function argument [-Wpass-global-variable]
    call pass_int_1d(i3(n:n+3))     !< warn:    shape /= [1]
    call pass_int(ia(1))            !< ok:      allocatable
    call pass_int(ia(n))            !< ok:      allocatable
    call pass_int_1d(ia)            !< ok:      allocatable
    call pass_int_1d(ia(n:n+3))     !< ok:      allocatable

    !WARNING: Passing global variable 'x1' from COMMON 'xy1' as function argument [-Wpass-global-variable]
    call pass_real(x1)              !< warn:    x1 from common
    !WARNING: Passing global variable 'y1' from COMMON 'xy1' as function argument [-Wpass-global-variable]
    call pass_real_1d(y1)           !< warn:    y1 from common or offset /= 0
    !WARNING: Passing global variable 'y1' from COMMON 'xy1' as function argument [-Wpass-global-variable]
    call pass_real(y1(1))           !< warn:    offset /= 0
    !WARNING: Passing global variable 'y1' from COMMON 'xy1' as function argument [-Wpass-global-variable]
    call pass_real(y1(n))           !< warn:    offset /= 0
    !WARNING: Passing global variable 'y1' from COMMON 'xy1' as function argument [-Wpass-global-variable]
    call pass_real_1d(y1(n:n+3))    !< warn:    offset /= 0

    !WARNING: Passing global variable 'y2' from COMMON 'xy2' as function argument [-Wpass-global-variable]
    call pass_real(y2)              !< warn:    offset /= 0
    !WARNING: Passing global variable 'x2' from COMMON 'xy2' as function argument [-Wpass-global-variable]
    call pass_real_1d(x2)           !< warn:    more than one variable in common block
    !WARNING: Passing global variable 'x2' from COMMON 'xy2' as function argument [-Wpass-global-variable]
    call pass_real(x2(1))           !< warn:    more than one variable in common block
    !WARNING: Passing global variable 'x2' from COMMON 'xy2' as function argument [-Wpass-global-variable]
    call pass_real(x2(n))           !< warn:    more than one variable in common block
    !WARNING: Passing global variable 'x2' from COMMON 'xy2' as function argument [-Wpass-global-variable]
    call pass_real_1d(x2(n:n+3))    !< warn:    more than one variable in common block

    !WARNING: Passing global variable 'z' from COMMON 'fm' as function argument [-Wpass-global-variable]
    call pass_real_1d(z)            !< warn:    z from common
    call pass_real(z(1))            !< ok:      single element/begin of mem block
    call pass_real(z(n))            !< ok:      single element/begin of mem block
    call pass_real_1d(z(n:n+3))     !< ok:      mem block

    !WARNING: Passing global variable 'z2' from COMMON 'fm_bad' as function argument [-Wpass-global-variable]
    call pass_real_1d(z2)           !< warn:    shape /= [1]
    !WARNING: Passing global variable 'z2' from COMMON 'fm_bad' as function argument [-Wpass-global-variable]
    call pass_real(z2(1))           !< warn:    shape /= [1]
    !WARNING: Passing global variable 'z2' from COMMON 'fm_bad' as function argument [-Wpass-global-variable]
    call pass_real(z2(n))           !< warn:    shape /= [1]
    !WARNING: Passing global variable 'z2' from COMMON 'fm_bad' as function argument [-Wpass-global-variable]
    call pass_real_1d(z2(n:n+3))    !< warn:    shape /= [1]
  end subroutine explicit_test
end module explicit_test_mod

subroutine module_test(n)
  use explicit_test_mod, only: i1, i2, i3, ia
  implicit none (type, external)
  integer, intent(in) :: n

  external :: imp_pass_int, imp_pass_int_1d

  !WARNING: Passing global variable 'i1' from MODULE 'explicit_test_mod' as function argument [-Wpass-global-variable]
  call imp_pass_int(i1)              !< warn:    i1 from common
  call imp_pass_int(i2(1))           !< ok:      single element/begin of mem block
  call imp_pass_int(i2(n))           !< ok:      single element/begin of mem block
  !WARNING: Passing global variable 'i3' from MODULE 'explicit_test_mod' as function argument [-Wpass-global-variable]
  call imp_pass_int(i3(1))           !< warn:    shape /= [1]
  !WARNING: Passing global variable 'i3' from MODULE 'explicit_test_mod' as function argument [-Wpass-global-variable]
  call imp_pass_int(i3(n))           !< warn:    shape /= [1]
  call imp_pass_int(ia(1))           !< ok:      allocatable
  call imp_pass_int(ia(n))           !< ok:      allocatable

  !WARNING: Passing global variable 'i2' from MODULE 'explicit_test_mod' as function argument [-Wpass-global-variable]
  call imp_pass_int_1d(i2)           !< warn:    i2 from module
  call imp_pass_int_1d(i2(n:n+3))    !< ok:      mem block
  !WARNING: Passing global variable 'i3' from MODULE 'explicit_test_mod' as function argument [-Wpass-global-variable]
  call imp_pass_int_1d(i3)           !< warn:    i3 from module & shape /= [1]
  !WARNING: Passing global variable 'i3' from MODULE 'explicit_test_mod' as function argument [-Wpass-global-variable]
  call imp_pass_int_1d(i3(n:n+3))    !< warn:    shape /= [1]
  call imp_pass_int_1d(ia)           !< ok:      allocatable
  call imp_pass_int_1d(ia(n:n+3))    !< ok:      allocatable
end subroutine module_test

subroutine implicit_test(n)
  implicit none (type, external)
  integer, intent(in) :: n
  real :: x1, y1
  real :: x2, y2
  real :: z, z2
  common /xy1/ x1, y1(1)
  common /xy2/ x2(1), y2
  common /fm/ z(1)
  common /fm_bad/ z2(5)

  external :: imp_pass_real, imp_pass_real_1d

  !WARNING: Passing global variable 'x1' from COMMON 'xy1' as function argument [-Wpass-global-variable]
  call imp_pass_real(x1)             !< warn:    x1 from common
  !WARNING: Passing global variable 'y1' from COMMON 'xy1' as function argument [-Wpass-global-variable]
  call imp_pass_real_1d(y1)          !< warn:    y1 from common and offset /= 0
  !WARNING: Passing global variable 'y1' from COMMON 'xy1' as function argument [-Wpass-global-variable]
  call imp_pass_real(y1(1))          !< warn:    offset /= 0
  !WARNING: Passing global variable 'y1' from COMMON 'xy1' as function argument [-Wpass-global-variable]
  call imp_pass_real(y1(n))          !< warn:    offset /= 0
  !WARNING: Passing global variable 'y1' from COMMON 'xy1' as function argument [-Wpass-global-variable]
  call imp_pass_real_1d(y1(n:n+3))   !< warn:    offset /= 0

  !WARNING: Passing global variable 'y2' from COMMON 'xy2' as function argument [-Wpass-global-variable]
  call imp_pass_real(y2)             !< warn:    y2 from common and offset /= 0
  !WARNING: Passing global variable 'x2' from COMMON 'xy2' as function argument [-Wpass-global-variable]
  call imp_pass_real_1d(x2)          !< warn:    x2 from common
  !WARNING: Passing global variable 'x2' from COMMON 'xy2' as function argument [-Wpass-global-variable]
  call imp_pass_real(x2(1))          !< warn:    more than one variable in common
  !WARNING: Passing global variable 'x2' from COMMON 'xy2' as function argument [-Wpass-global-variable]
  call imp_pass_real(x2(n))          !< warn:    more than one variable in common
  !WARNING: Passing global variable 'x2' from COMMON 'xy2' as function argument [-Wpass-global-variable]
  call imp_pass_real_1d(x2(n:n+3))   !< warn:    more than one variable in common

  !WARNING: Passing global variable 'z' from COMMON 'fm' as function argument [-Wpass-global-variable]
  call imp_pass_real_1d(z)           !< warn:    z from common
  call imp_pass_real(z(1))           !< ok:      single element/begin of mem block
  call imp_pass_real(z(n))           !< ok:      single element/begin of mem block
  call imp_pass_real_1d(z(n:n+3))    !< ok:      mem block

  !WARNING: Passing global variable 'z2' from COMMON 'fm_bad' as function argument [-Wpass-global-variable]
  call imp_pass_real_1d(z2)          !< warn:    z2 from common, shape /= [1]
  !WARNING: Passing global variable 'z2' from COMMON 'fm_bad' as function argument [-Wpass-global-variable]
  call imp_pass_real(z2(1))          !< warn:    shape /= [1]
  !WARNING: Passing global variable 'z2' from COMMON 'fm_bad' as function argument [-Wpass-global-variable]
  call imp_pass_real(z2(n))          !< warn:    shape /= [1]
  !WARNING: Passing global variable 'z2' from COMMON 'fm_bad' as function argument [-Wpass-global-variable]
  call imp_pass_real_1d(z2(n:n+3))   !< warn:    shape /= [1]
end subroutine implicit_test
