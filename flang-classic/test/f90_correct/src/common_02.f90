! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for derived type in COMMON.

block data
  integer :: x = 10
  common /comm6/ x
end block data

program test
  use iso_c_binding
  implicit none

  type type1
    integer :: t1
  end type type1

  type type2
    sequence
    integer, allocatable, dimension(:) :: t2
  end type type2

  type type3
    sequence
    type(type2) :: t3
  end type type3

  type type4
    sequence
    integer, allocatable :: t4
  end type type4

  type type5
    sequence
    type(type4) :: t5
  end type type5

  type, bind(c) :: type6
    integer :: i = 10
  end type type6

  type :: type7(k)
    sequence
    integer, kind :: k = 4
    integer(kind=k) :: i = 10
  end type type7

  !{error "PGF90-S-0155-Derived type shall have the BIND attribute or the SEQUENCE attribute in COMMON - a_type1"}
  common /comm1/ a_type1
  !{error "PGF90-S-0155-Derived type cannot have allocatable attribute in COMMON - b_type2"}
  common /comm2/ b_type2
  !{error "PGF90-S-0155-Derived type cannot have allocatable attribute in COMMON - c_type3"}
  common /comm3/ c_type3
  !{error "PGF90-S-0155-Derived type cannot have allocatable attribute in COMMON - d_type4"}
  common /comm4/ d_type4
  !{error "PGF90-S-0155-Derived type cannot have allocatable attribute in COMMON - array_type5"}
  common /comm5/ array_type5
  !{error "PGF90-S-0155-Derived type cannot have default initialization in COMMON - e_type6"}
  common /comm6/ e_type6
  !{error "PGF90-S-0155-Derived type cannot have default initialization in COMMON - f_type7"}
  common /comm7/ f_type7

  type(type1) :: a_type1
  type(type2) :: b_type2
  type(type3) :: c_type3
  type(type4) :: d_type4
  type(type5) :: array_type5(5)
  type(type6) :: e_type6
  type(type7(4)) :: f_type7
end
