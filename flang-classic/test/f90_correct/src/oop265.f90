! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod

type base_t
logical result
contains
procedure, pass(this)   :: baseproc_pass => baseproc
procedure, nopass :: baseproc_nopass => baseproc
generic           :: some_proc => baseproc_pass, baseproc_nopass
end type

type, extends(base_t) :: ext_t
end type

contains

logical function baseproc(this,v)
class(base_t) :: this
logical v
select type(this)
type is(base_t)
baseproc = v
type is (ext_t)
baseproc = .not. v
class default
stop 'baseproc: unexepected type for this'
end select
end function

end module

program p
USE CHECK_MOD
use mod
logical results(2)
logical expect(2)
type(base_t) :: t
type(ext_t) :: t2

  results = .false.
  expect = .true.
  t%result = .false.
  t2%result = .true.
  results(1) = t%some_proc(.true.)
  results(2) = t%some_proc(t2,.false.)

  call check(results,expect,2)

end program 
