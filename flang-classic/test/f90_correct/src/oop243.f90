! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module a
  implicit none
  type a_t
  logical :: result = .false.
   contains
     procedure :: say_hello
  end type a_t
  private
  public :: a_t
contains
  subroutine say_hello (this)
    class(a_t), intent(inout) :: this
    print *,'Hello from a'
    this%result = .true.
  end subroutine say_hello
end module a

module b
  use a
  implicit none
  type b_t
     type(a_t) :: a
   contains
     procedure :: say_hello
  end type b_t
contains
  subroutine say_hello (this)
    class(b_t), intent(inout) :: this
    call this%a%say_hello()
  end subroutine 
end module b

program p
USE CHECK_MOD
   use b
   logical results(1)
   logical expect(1)
   type(b_t) :: bt
   results = .false.
   expect = .true.
   call bt%say_hello()
   results(1) = bt%a%result
   call check(results,expect,1)
end
