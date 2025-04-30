! @@name:	taskgroup.1f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
module tree_type_mod
  integer, parameter :: max_steps=100
  type tree_type
    type(tree_type), pointer :: left, right
  end type
  contains
    subroutine compute_something(tree)
      type(tree_type), pointer :: tree
! some computation
    end subroutine
    recursive subroutine compute_tree(tree)
      type(tree_type), pointer :: tree
      if (associated(tree%left)) then
!$omp task
        call compute_tree(tree%left)
!$omp end task
      endif
      if (associated(tree%right)) then
!$omp task
        call compute_tree(tree%right)
!$omp end task
      endif
!$omp task
      call compute_something(tree)
!$omp end task
    end subroutine
end module
program main
  use tree_type_mod
  type(tree_type), pointer :: tree
  call init_tree(tree);
!$omp parallel
!$omp single
!$omp task
  call start_background_work()
!$omp end task
  do i=1, max_steps
!$omp taskgroup
!$omp task
    call compute_tree(tree)
!$omp end task
!$omp end taskgroup ! wait on tree traversal in this step
    call check_step()
  enddo
!$omp end single
!$omp end parallel    ! only now is background work required to be complete
  call print_results()
end program
