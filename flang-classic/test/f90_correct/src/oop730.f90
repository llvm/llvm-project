! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! complex type bound function call

module mmm
  complex, parameter :: bits(3) = [(1.0,2.0), (-3.0,-4.0), (5.0,6.0)]
  type :: vector
    complex :: data(3) = bits
  contains
    procedure :: get_value
  end type vector

contains
  complex function get_value(vec, elem)
    class(vector) :: vec
    integer       :: elem
    get_value = vec%data(elem)
  end function get_value
end

program test
  use mmm
  type(vector) :: vec
  integer      :: elem
  complex      :: ccc(3)
  do elem = 1, 3
    ccc(elem) = vec%get_value(elem)
  end do
  print*, sum(ccc), sum(bits)
  if (sum(ccc) .eq. sum(bits)) print*, 'PASS'
  if (sum(ccc) .ne. sum(bits)) print*, 'FAIL'
end
