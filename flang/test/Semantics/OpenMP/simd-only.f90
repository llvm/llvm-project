! RUN: %flang_fc1 -fopenmp-simd -fdebug-dump-parse-tree %s 2>&1 | FileCheck %s

! Test that non-SIMD OpenMPConstructs are removed on the parse tree level
! when -fopenmp-simd is specified.
! Tests the logic in lib/Semantics/rewrite-parse-tree.cpp

! CHECK-LABEL: Name = 'test_simd'
subroutine test_simd()
  integer :: i

  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
  ! CHECK: OmpLoopDirective -> llvm::omp::Directive = simd
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
  !$omp simd
  do i = 1, 100
  end do
end subroutine

! CHECK-LABEL: Name = 'test_do_simd'
subroutine test_do_simd()
  integer :: i

  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
  ! CHECK: OmpLoopDirective -> llvm::omp::Directive = do simd
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
  !$omp do simd
  do i = 1, 100
  end do
end subroutine


! CHECK-LABEL: Name = 'test_parallel_do_simd'
subroutine test_parallel_do_simd()
  integer :: i

  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
  ! CHECK: OmpLoopDirective -> llvm::omp::Directive = parallel do simd
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
  !$omp parallel do simd
  do i = 1, 100
  end do
end subroutine

! CHECK-LABEL: Name = 'test_simd_scan'
subroutine test_simd_scan()
  integer :: i
  real :: sum

  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
  ! CHECK: OmpLoopDirective -> llvm::omp::Directive = simd
  !$omp simd reduction(inscan,+:sum)
  do i = 1, N
    sum = sum + a(i)
    ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct -> OmpDirectiveSpecification
    ! CHECK: OmpDirectiveName -> llvm::omp::Directive = scan
    !$omp scan inclusive(sum)
    sum       = sum + a(i)
  end do

end subroutine

! CHECK-LABEL: Name = 'test_simd_atomic'
subroutine test_simd_atomic()
  integer :: i, x

  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
  ! CHECK: OmpLoopDirective -> llvm::omp::Directive = simd
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
  !$omp simd
  do i = 1, 100
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAtomicConstruct
  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=i'
  !$omp atomic write
  x = i
  end do
end subroutine

! CHECK-LABEL: Name = 'test_do'
subroutine test_do()
  integer :: i

  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
  ! CHECK-NOT: OmpLoopDirective -> llvm::omp::Directive = do
  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
  !$omp parallel do
  do i = 1, 100
  end do
end subroutine

! CHECK-LABEL: Name = 'test_do_nested'
subroutine test_do_nested()
  integer :: i

  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
  ! CHECK-NOT: OmpLoopDirective -> llvm::omp::Directive = parallel do
  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
  !$omp parallel do
  do i = 1, 100
    do j = 1, 100
    end do
  end do
end subroutine

! CHECK-LABEL: Name = 'test_target'
subroutine test_target()
  integer :: i

  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPBlockct
  ! CHECK-NOT: OmpLoopDirective -> llvm::omp::Directive = target
  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
  !$omp target
  do i = 1, 100
  end do
  !$omp end target
end subroutine

! CHECK-LABEL: Name = 'test_target_teams_distribute'
subroutine test_target_teams_distribute()
  integer :: i

  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
  ! CHECK-NOT: OmpLoopDirective -> llvm::omp::Directive = target teams distribute
  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
  !$omp target teams distribute
  do i = 1, 100
  end do
  !$omp end target teams distribute
end subroutine


! CHECK-LABEL: Name = 'test_target_data'
subroutine test_target_data()
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPBlockConstruct
  ! CHECK-NOT: OmpLoopDirective -> llvm::omp::Directive = target data
  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
  !$omp target data map(to: A) map(tofrom: B)
  do i = 1, 100
  end do
  !$omp end target data
end subroutine

! CHECK-LABEL: Name = 'test_loop'
subroutine test_loop()
  integer :: i

  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
  ! CHECK-NOT: OmpLoopDirective -> llvm::omp::Directive = loop
  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
  !$omp loop bind(thread)
  do i = 1, 100
  end do
end subroutine

! CHECK-LABEL: Name = 'test_unroll'
subroutine test_unroll()
  integer :: i

  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
  ! CHECK-NOT: OmpLoopDirective -> llvm::omp::Directive = unroll
  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
  !$omp unroll
  do i = 1, 100
  end do
end subroutine

! CHECK-LABEL: Name = 'test_do_ordered'
subroutine test_do_ordered()
  integer :: i, x
  x = 0

  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
  ! CHECK-NOT: OmpLoopDirective -> llvm::omp::Directive = do
  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
  !$omp do ordered
  do i = 1, 100
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPBlockConstruct
  ! CHECK-NOT: OmpLoopDirective -> llvm::omp::Directive = ordered
  !$omp ordered
  x = x + 1
  !$omp end ordered
  end do
end subroutine

! CHECK-LABEL: Name = 'test_cancel'
subroutine test_cancel()
  integer :: i, x
  x = 0

  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
  ! CHECK-NOT: OmpLoopDirective -> llvm::omp::Directive = parallel do
  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
  !$omp parallel do
  do i = 1, 100
  if (i == 10) then
    ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPCancelConstruct -> OmpDirectiveSpecification
    ! CHECK-NOT: OmpLoopDirective -> llvm::omp::Directive = cancel
    !$omp cancel do
  end if
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPCancellationPointConstruct -> OmpDirectiveSpecification
  ! CHECK-NOT: OmpLoopDirective -> llvm::omp::Directive = cancellation point
  !$omp cancellation point do
  end do
end subroutine

! CHECK-LABEL: Name = 'test_scan'
subroutine test_scan()
  integer :: i, sum

  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
  ! CHECK-NOT: OmpLoopDirective -> llvm::omp::Directive = parallel do
  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
  !$omp parallel do reduction(inscan, +: sum)
  do i = 1, n
    sum = sum + i
    ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct -> OmpDirectiveSpecification
    ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = scan
    !$omp scan inclusive(sum)
  end do
  !$omp end parallel do
end subroutine

! CHECK-LABEL: Name = 'test_target_map'
subroutine test_target_map()
  integer :: array(10)

  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPBlockConstruct
  ! CHECK-NOT: OmpLoopDirective -> llvm::omp::Directive = target
  !$omp target map(tofrom: array(2:10))
    array(2) = array(2) * 2
  !$omp end target
end subroutine

! CHECK-LABEL: Name = 'test_sections'
subroutine test_sections()
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPSectionsConstruct
  !$omp sections
  ! CHECK-NOT: OpenMPConstruct -> OpenMPSectionConstruct
  !$omp section
  ! CHECK-NOT: OpenMPConstruct -> OpenMPSectionConstruct
  !$omp section
  !$omp end sections
end subroutine

! CHECK-LABEL: Name = 'test_threadprivate_mod'
module test_threadprivate_mod
  implicit none
  ! CHECK: DeclarationConstruct -> SpecificationConstruct -> TypeDeclarationStmt
  ! CHECK: Name = 'x'
  ! CHECK: Name = 'y'
  integer :: x, y
  ! CHECK: DeclarationConstruct -> SpecificationConstruct -> OtherSpecificationStmt -> CommonStmt
  ! CHECK: Name = 'x'
  ! CHECK: Name = 'y'
  common /vars/ x, y
  ! CHECK-NOT: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPThreadprivate
  !$omp threadprivate(/vars/)
end module

! CHECK-LABEL: Name = 'test_atomic'
subroutine test_atomic()
  real :: z, x, y
  !$omp parallel private(tid, z)
    ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAtomicConstruct
    ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=y'
    !$omp atomic write
      x = y
    ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAtomicConstruct
    ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'z=x'
    !$omp atomic read
      z = x
    ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAtomicConstruct
    ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=x+1._4'
    !$omp atomic update
      x = x + 1
    ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAtomicConstruct
    ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'z=x'
    !$omp atomic read
      z = x
    ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAtomicConstruct
    ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=x+y'
    !$omp atomic capture
      x   = x + y
    !$omp end atomic
  !$omp end parallel
end subroutine

! CHECK-LABEL: Name = 'test_task_single_taskwait'
subroutine test_task_single_taskwait()
  integer :: x
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPBlockConstruct
  ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = parallel
  !$omp parallel
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPBlockConstruct
  ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = single
  !$omp single
    ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
    do i = 1, 5
      ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPBlockConstruct
      ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = task
      ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=i'
      !$omp task
      x = i
      !$omp end task
    end do
    ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct -> OmpDirectiveSpecification
    ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = taskwait
    !$omp taskwait
  !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: Name = 'test_task_taskyield_flush_barrier'
subroutine test_task_taskyield_flush_barrier()
  integer :: x, i
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPBlockConstruct
  ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = parallel
  !$omp parallel
    ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct -> OmpDirectiveSpecification
    ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = barrier
    !$omp barrier
    ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPBlockConstruct
    ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = single
    !$omp single
      ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPBlockConstruct
      ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = task
      !$omp task
        ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct -> OmpDirectiveSpecification
        ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = taskyield
        !$omp taskyield
        ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=i'
        x = i
        ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPFlushConstruct -> OmpDirectiveSpecification
        !$omp flush
      !$omp end task
      ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPBlockConstruct
      ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = task
      !$omp task
        ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPFlushConstruct -> OmpDirectiveSpecification
        !$omp flush
      !$omp end task
      ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct -> OmpDirectiveSpecification
      ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = taskwait
      !$omp taskwait
    !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: Name = 'test_master_masked'
subroutine test_master_masked()
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPBlockConstruct
  ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = parallel
  !$omp parallel private(tid)
    ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPBlockConstruct
    ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = masked
    !$omp masked
    ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=y'
    x = y
    !$omp end masked
    ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPBlockConstruct
    ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = master
    !$omp master
    ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'y=x'
    y = x
    !$omp end master
  !$omp end parallel
end subroutine

! CHECK-LABEL: Name = 'test_critical'
subroutine test_critical()
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPBlockConstruct
  ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = parallel
  !$omp parallel do private(i)
  do i = 1, 4
    !$omp critical(mylock)
    ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=y'
    x = y
    !$omp end critical(mylock)
  end do
  !$omp end parallel do
end subroutine

! CHECK-LABEL: Name = 'test_target_enter_exit_update_data'
subroutine test_target_enter_exit_update_data()
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct -> OmpDirectiveSpecification
  ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = target enter data
  !$omp target enter data map(to: A)
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPBlockConstruct
  ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = target teams distribute parallel do
  !$omp target teams distribute parallel do
  ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
  do i = 1, n
    ! CHECK: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=y'
    x = y
  end do
  !$omp end target teams distribute parallel do
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct -> OmpDirectiveSpecification
  ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = target update
  !$omp target update from(A)
  ! CHECK-NOT: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPStandaloneConstruct -> OpenMPSimpleStandaloneConstruct -> OmpDirectiveSpecification
  ! CHECK-NOT: OmpDirectiveName -> llvm::omp::Directive = target exit data
  !$omp target exit data map(from: A)
end subroutine

! CHECK-LABEL: Name = 'test_declare_mapper'
module test_declare_mapper
  implicit none

  type :: myvec_t
    integer               :: len
    real, allocatable     :: data(:)
  end type myvec_t

  ! CHECK-NOT: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareMapperConstruct
  !$omp declare mapper(myvec_t :: v) map(v, v%data(1:v%len))
end module
