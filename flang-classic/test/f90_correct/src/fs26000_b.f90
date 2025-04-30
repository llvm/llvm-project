! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Variation from fs26000_a (integer case)
! This is real case.

! Related to a test by Neil Carlson from Los Alamos National Laboratory
! related to Flang github issue #243

type foo
  class(*), allocatable :: val
end type
type(foo) :: x
x = foo(42.5)

select type (val => x%val)
type is (real)
  if (val /= 42.5) then
    print *, "FAIL 1"
  else
    print *, "PASS"
  endif
class default
  print *, "FAIL 2"
end select
end
