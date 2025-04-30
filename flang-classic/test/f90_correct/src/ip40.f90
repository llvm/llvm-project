!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! accessing a host dummy in an allocate statement from within
! the internal procedure may not sufficient for the dummy to be homed
! in host.  The homing will not be done if the GSCOPE flag of the dummy
! is not set
!
module f19924
  type :: vertex_table_entry_t
     integer :: pdg1 = 0, pdg2 = 0
     integer :: n = 0
     integer, dimension(:), allocatable :: pdg3
  end type vertex_table_entry_t
  type :: vertex_table_t
     type(vertex_table_entry_t), dimension(:), allocatable :: entry
     integer :: n_collisions = 0
     integer :: mask
  end type vertex_table_t
  integer*4 :: expect(2) = (/42*4,42/)
  integer*4 :: result(2)
end module f19924
subroutine vertex_table_match (vt, pdg3)
    use f19924
    type(vertex_table_t), intent(in) :: vt
    integer, dimension(:), allocatable, intent(out) :: pdg3
    integer :: i
    call match(i)
contains
    subroutine match (hashval)
      integer, intent(in) :: hashval
!!    print *,size(vt%entry(1)%pdg3)
      allocate(pdg3(size(vt%entry(1)%pdg3)))
!     print *,size(pdg3)
      result(1) = size(pdg3)*4
    end subroutine match
  end subroutine vertex_table_match
program p
 use f19924
 interface 
  subroutine vertex_table_match (vt, pdg3)
    use f19924
    type(vertex_table_t), intent(in) :: vt
    integer, dimension(:), allocatable, intent(out) :: pdg3
  end subroutine
 end interface
 type(vertex_table_t) :: pvt
 integer, dimension(:), allocatable :: alloc_arr
 allocate(pvt%entry(1:1))
 allocate(pvt%entry(1)%pdg3(1:42))
 call vertex_table_match(pvt, alloc_arr)
!print *,size(alloc_arr)
 result(2) = size(alloc_arr)
 call check(result, expect, 2);
end
