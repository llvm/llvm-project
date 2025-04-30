! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! c_associated call variants

logical function f1() ! scalars
  use, intrinsic :: iso_c_binding, only: c_associated, c_loc, c_null_ptr, c_ptr
  implicit none

  interface
    type(c_ptr) function copy(pp)
      import
      type(c_ptr) :: pp
    end function copy
  end interface

  integer, target :: a(5), b(5)
  type(c_ptr) :: p0, pa, pb, px
  logical :: T1, T2, T3, T4, T5, T6, T7, T8

  p0 = c_null_ptr
  pa = c_loc(a)
  pb = c_loc(b)
  px = pa

  T1 = .not. c_associated(p0)
  T2 =       c_associated(pa)
  T3 =       c_associated(pa, px)
  T4 = .not. c_associated(pa, pb)

  T5 = .not. c_associated(copy(p0))
  T6 =       c_associated(copy(pa))
  T7 =       c_associated(copy(pa), copy(px))
  T8 = .not. c_associated(copy(pa), copy(pb))

  print*, 'f1:  ', T1, T2, T3, T4, ' ', T5, T6, T7, T8
  f1 = all([T1, T2, T3, T4, T5, T6, T7, T8])
end function f1

logical function f2() ! elements
  use, intrinsic :: iso_c_binding, only: c_associated, c_loc, c_ptr
  implicit none

  interface
    type(c_ptr) function copy(pp)
      import
      type(c_ptr) :: pp
    end function copy
  end interface

  integer, pointer :: a(:), b(:), x(:)
  logical :: T1, T2, T3, T4, T5, T6, T7, T8

  allocate(x(5))
  a => x
  b => x(2:5)

  T1 =       c_associated(c_loc(a(5)))
  T2 = .not. c_associated(c_loc(a(1)), c_loc(b(1)))
  T3 =       c_associated(c_loc(a(2)), c_loc(b))
  T4 =       c_associated(c_loc(a(2)), c_loc(b(1)))

  T5 =       c_associated(copy(c_loc(a(5))))
  T6 = .not. c_associated(copy(c_loc(a(1))), copy(c_loc(b(1))))
  T7 =       c_associated(copy(c_loc(a(2))), copy(c_loc(b)))
  T8 =       c_associated(copy(c_loc(a(2))), copy(c_loc(b(1))))

  print*, 'f2:  ', T1, T2, T3, T4, ' ', T5, T6, T7, T8
  f2 = all([T1, T2, T3, T4, T5, T6, T7, T8])
end function f2

logical function f3() ! components
  use, intrinsic :: iso_c_binding, only: c_associated, c_loc, c_null_ptr, c_ptr
  implicit none

  interface
    type(c_ptr) function copy(pp)
      import
      type(c_ptr) :: pp
    end function copy
  end interface

  type tt
    type(c_ptr) :: a, b, z
  end type tt

  type(tt) :: v
  integer, target :: x(5)
  logical :: T1, T2, T3, T4, T5, T6, T7, T8

  v%a = c_loc(x)
  v%b = c_loc(x(2))
  v%z = c_null_ptr

  T1 =       c_associated(v%a)
  T2 = .not. c_associated(v%a, v%b)
  T3 =       c_associated(v%b, c_loc(x(2)))
  T4 = .not. c_associated(v%z)

  T5 =       c_associated(copy(v%a))
  T6 = .not. c_associated(copy(v%a), copy(v%b))
  T7 =       c_associated(copy(v%b), copy(c_loc(x(2))))
  T8 = .not. c_associated(copy(v%z))

  print*, 'f3:  ', T1, T2, T3, T4, ' ', T5, T6, T7, T8
  f3 = all([T1, T2, T3, T4, T5, T6, T7, T8])
end function f3

type(c_ptr) function copy(pp)
  use, intrinsic :: iso_c_binding, only: c_ptr
  implicit none
  type(c_ptr) :: pp
  copy = pp
end function copy

  implicit none
  logical :: LL, f1, f2, f3

  LL = f1()
  LL = LL .and. f2()
  LL = LL .and. f3()

  if (.not. LL) print*, 'FAIL'
  if (      LL) print*, 'PASS'
end
