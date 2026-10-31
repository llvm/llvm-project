! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! COMMON-block members participate in clause conflicts and containment.

subroutine block_private_member_firstprivate()
  integer :: a, b
  common /blk/ a, b
  !ERROR: 'a' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel private(/blk/) firstprivate(a)
    a = b
  !$acc end parallel
end subroutine

subroutine member_firstprivate_block_private()
  integer :: a, b
  common /blk/ a, b
  !ERROR: 'blk' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel firstprivate(a) private(/blk/)
    a = b
  !$acc end parallel
end subroutine

subroutine block_firstprivate_member_private()
  integer :: a, b
  common /blk/ a, b
  !ERROR: 'a' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel firstprivate(/blk/) private(a)
    a = b
  !$acc end parallel
end subroutine

subroutine member_private_block_firstprivate()
  integer :: a, b
  common /blk/ a, b
  !ERROR: 'blk' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel private(a) firstprivate(/blk/)
    a = b
  !$acc end parallel
end subroutine

subroutine block_private_member_private()
  integer :: a, b
  common /blk/ a, b
  !WARNING: 'a' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
  !$acc parallel private(/blk/) private(a)
    a = b
  !$acc end parallel
end subroutine

subroutine member_private_block_private()
  integer :: a, b
  common /blk/ a, b
  !WARNING: 'a' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
  !$acc parallel private(a) private(/blk/)
    a = b
  !$acc end parallel
end subroutine

subroutine block_firstprivate_member_firstprivate()
  integer :: a, b
  common /blk/ a, b
  !WARNING: 'a' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
  !$acc parallel firstprivate(/blk/) firstprivate(a)
    a = b
  !$acc end parallel
end subroutine

subroutine member_firstprivate_block_firstprivate()
  integer :: a, b
  common /blk/ a, b
  !WARNING: 'a' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
  !$acc parallel firstprivate(a) firstprivate(/blk/)
    a = b
  !$acc end parallel
end subroutine

subroutine all_members_before_block()
  integer :: a, b
  common /blk/ a, b
  !WARNING: 'a' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
  !WARNING: 'b' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
  !$acc parallel private(a, b) private(/blk/)
    a = b
  !$acc end parallel
end subroutine

subroutine all_members_after_block()
  integer :: a, b
  common /blk/ a, b
  !WARNING: 'a' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
  !WARNING: 'b' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
  !$acc parallel private(/blk/) private(a, b)
    a = b
  !$acc end parallel
end subroutine

subroutine common_firstprivate_loop_index()
  integer :: a, b
  common /blk/ a, b
  !$acc parallel loop firstprivate(/blk/)
  !ERROR: 'a' appears in more than one data-sharing clause on the same OpenACC directive
  do a = 1, 10
    b = a
  end do
end subroutine

subroutine common_private_loop_index()
  integer :: a, b
  common /blk/ a, b
  !$acc parallel loop private(/blk/)
  do a = 1, 10
    b = a
  end do
end subroutine

subroutine common_member_section_conflict()
  integer :: a(10), b
  common /blk/ a, b
  !ERROR: 'a(1:5)' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel private(/blk/) firstprivate(a(1:5))
    a(1) = b
  !$acc end parallel
end subroutine

subroutine distinct_common_members()
  integer :: a, b
  common /blk/ a, b
  !$acc parallel private(a) firstprivate(b)
    a = b
  !$acc end parallel
end subroutine

subroutine distinct_common_blocks()
  integer :: a, b
  common /blk1/ a
  common /blk2/ b
  !$acc parallel private(/blk1/) firstprivate(/blk2/)
    a = b
  !$acc end parallel
end subroutine

subroutine common_data_action_and_dsa()
  integer :: a, b
  common /blk/ a, b
  !$acc parallel copy(/blk/) private(a)
    a = b
  !$acc end parallel
end subroutine

subroutine repeated_whole_common()
  integer :: a, b
  common /blk/ a, b
  !WARNING: 'blk' appears more than once in the same kind of data-sharing clause on an OpenACC directive; duplicate ignored [-Wopenacc-usage]
  !$acc parallel private(/blk/) private(/blk/)
    a = b
  !$acc end parallel
end subroutine

subroutine common_container_last(i, j)
  integer :: a(10), b, i, j
  common /blk/ a, b
  !WARNING: 'a(i)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
  !WARNING: 'a(j)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
  !$acc parallel private(a(i), a(j)) private(/blk/)
    a(9) = b
  !$acc end parallel
end subroutine

subroutine common_component_conflict()
  type :: pair
    sequence
    integer :: a, b
  end type
  type(pair) :: p
  common /blk/ p
  !ERROR: 'p%a' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel private(/blk/) firstprivate(p%a)
    p%a = p%b
  !$acc end parallel
end subroutine

module common_host_association
  integer :: a, b
  common /host_blk/ a, b
contains
  subroutine host_associated_member_conflict()
    !ERROR: 'a' appears in more than one data-sharing clause on the same OpenACC directive
    !$acc parallel private(/host_blk/) firstprivate(a)
      a = b
    !$acc end parallel
  end subroutine
end module
