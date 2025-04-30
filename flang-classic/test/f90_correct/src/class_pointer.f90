! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! The dummy argument is class pointer and the real argument is type.

module mod1
  type, public :: mytype1
    integer, private :: key = 0
  end type mytype1

  type, public :: mytype2
    class(*), pointer :: p_key
  end type mytype2
contains

  function test(curr, key) result(rst)
    class(mytype2) :: curr
    class(*), target, intent(in) :: key
    integer :: rst
    integer :: ier = 0, debug = 1
    ! The key%sd will be generated bacause of the statement of
    ! allocate(curr%p_key, source = key) in parser stage, in which case, flang
    ! will make RTE_init_unl_poly_desc and get the wrong argument of type
    ! descriptor in check_pointer_type. If the statement is removed, flang will
    ! not generate RTE_init_unl_poly_desc and process the statement of
    ! ptr2_assign(curr%p_key,key) later.
    if (debug > ier) then
      curr%p_key => key
    else
      allocate(curr%p_key, source = key)
    endif
    rst = cmp(curr%p_key)
  end function

  function cmp(ty) result(res)
    class(*) :: ty
    select type(ty)
      class is(mytype1)
        res = 1
      class default
        res = 0
    end select
  end function
end module

program example
  use mod1
  integer :: rst(1)
  integer :: expect(1)
  integer, parameter :: n = 1
  type(mytype1), target :: lnk
  type(mytype2) :: currp
  expect = 1
  rst = test(currp, lnk)
  call check(rst, expect, n)
end
