! RUN: split-file %s %t
! RUN: %python %S/../test_errors.py %t/declarative.f90 %flang -fopenmp -fopenmp-version=52
! RUN: %python %S/../test_errors.py %t/pure_assumes.f90 %flang -fopenmp -fopenmp-version=51
! RUN: %python %S/../test_errors.py %t/assumes.f90 %flang -fopenmp -fopenmp-version=52
! RUN: %python %S/../test_errors.py %t/conditional.f90 %flang -fopenmp -fopenmp-version=60
! RUN: %python %S/../test_errors.py %t/placement.f90 %flang -fopenmp -fopenmp-version=52
! RUN: %python %S/../test_errors.py %t/placement.f90 %flang -fopenmp -fopenmp-version=60
! RUN: %python %S/../test_errors.py %t/execution.f90 %flang -fopenmp -fopenmp-version=52
! RUN: %python %S/../test_errors.py %t/execution.f90 %flang -fopenmp -fopenmp-version=60

!--- declarative.f90
! Check OpenMP declarative directives

! 2.4 requires

subroutine requires_1(a)
  real(8), intent(inout) :: a
  !WARNING: REVERSE_OFFLOAD clause is not supported and will be ignored
  !$omp requires reverse_offload, unified_shared_memory, atomic_default_mem_order(relaxed)
  a = a + 0.01
end subroutine requires_1

subroutine requires_2(a)
  real(8), intent(inout) :: a
  !$omp requires unified_address
  a = a + 0.01
end subroutine requires_2

! 2.8.2 declare-simd

subroutine declare_simd_1(a, b)
  real(8), intent(inout) :: a, b
  !WARNING: DECLARE SIMD directive is not implemented yet and will be ignored
  !ERROR: 'a' in ALIGNED clause must be of type C_PTR, POINTER or ALLOCATABLE
  !$omp declare simd(declare_simd_1) aligned(a)
  a = 3.14 + b
end subroutine declare_simd_1

module m1
  abstract interface
     subroutine sub(x,y)
       integer, intent(in)::x
       integer, intent(in)::y
     end subroutine sub
  end interface
end module m1

subroutine declare_simd_2
  use m1
  procedure (sub) sub1
  !WARNING: DECLARE SIMD directive is not implemented yet and will be ignored
  !ERROR: INBRANCH and NOTINBRANCH clauses are mutually exclusive as members of 'branch' clause group
  !$omp declare simd(sub1) inbranch notinbranch
  procedure (sub), pointer::p
  p=>sub1
  call p(5,10)
end subroutine declare_simd_2

subroutine sub1 (x,y)
  integer, intent(in)::x, y
  print *, x+y
end subroutine sub1

! 2.10.6 declare-target
! 2.15.2 threadprivate

module m2
contains
  subroutine foo
    !$omp declare target
    !WARNING: The entity with PARAMETER attribute is used in a DECLARE TARGET directive [-Wopenmp-usage]
    !WARNING: The entity with PARAMETER attribute is used in a DECLARE TARGET directive [-Wopenmp-usage]
    !$omp declare target (foo, N, M)
    !WARNING: TO clause is no longer allowed on DECLARE TARGET directive since OpenMP v5.2 [-Wopenmp-deprecated]
    !WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead. [-Wopenmp-deprecated]
    !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !$omp declare target to(Q, S) link(R)
    !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !$omp declare target enter(Q, S) link(R)
    !WARNING: TO clause is no longer allowed on DECLARE TARGET directive since OpenMP v5.2 [-Wopenmp-deprecated]
    !WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead. [-Wopenmp-deprecated]
    !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !ERROR: MAP clause is not allowed on DECLARE TARGET directive
    !$omp declare target to(Q) map(from:Q)
    !ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !ERROR: MAP clause is not allowed on DECLARE TARGET directive
    !$omp declare target enter(Q) map(from:Q)
    integer, parameter :: N=10000, M=1024
    integer :: i
    real :: Q(N, N), R(N,M), S(M,M)
    !ERROR: A variable that appears in a THREADPRIVATE directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
    !$omp threadprivate(i)
  end subroutine foo
end module m2

! 2.16 declare-reduction

subroutine declare_red_1()
  integer :: my_var
  !$omp declare reduction (my_add_red : integer : omp_out = omp_out + omp_in) initializer (omp_priv=0)
  my_var = 0
  !$omp parallel reduction (my_add_red : my_var) num_threads(4)
  my_var = 1
  !$omp end parallel
  print *, "sum of thread numbers is ", my_var
end subroutine declare_red_1

!--- assumes.f90
module assumes_no_openmp_module
  implicit none
  !$omp assumes no_openmp
contains
  subroutine set_value(value)
    integer :: value
    value = 7
  end subroutine set_value
end module assumes_no_openmp_module

subroutine assumes_no_openmp_procedure(value)
  use assumes_no_openmp_module, only: set_value
  implicit none
  !$omp assumes no_openmp
  integer :: value
  call set_value(value)
end subroutine assumes_no_openmp_procedure

subroutine assumes_no_openmp_duplicate
  !ERROR: At most one NO_OPENMP clause can appear on ASSUMES directive
  !$omp assumes no_openmp no_openmp
end subroutine assumes_no_openmp_duplicate

module assumes_no_parallelism_module
  implicit none
  !$omp assumes no_parallelism
contains
  pure subroutine work()
    !$omp assumes no_openmp no_parallelism
  end subroutine work
end module assumes_no_parallelism_module

subroutine assumes_no_parallelism_procedure
  use assumes_no_parallelism_module, only: work
  implicit none
  !$omp assumes no_parallelism no_openmp
  call work()
end subroutine assumes_no_parallelism_procedure

subroutine assumes_no_parallelism_duplicate
  !ERROR: At most one NO_PARALLELISM clause can appear on ASSUMES directive
  !$omp assumes no_parallelism no_parallelism
end subroutine assumes_no_parallelism_duplicate

module assumes_absent_module
  implicit none
  !$omp assumes absent(parallel, barrier)
contains
  subroutine local_work()
    !$omp assumes absent(critical)
  end subroutine local_work
end module assumes_absent_module

subroutine assumes_absent_procedure()
  use assumes_absent_module
  implicit none
  !$omp assumes absent(target enter data)
  !$omp assumes absent(parallel)
end subroutine assumes_absent_procedure

subroutine assumes_absent_combined()
  !$omp assumes absent(barrier) no_openmp no_openmp_routines no_parallelism holds(.true.)
end subroutine assumes_absent_combined

subroutine assumes_absent_contains()
  !$omp assumes absent(barrier) contains(parallel)
  !$omp parallel
  !$omp end parallel
end subroutine assumes_absent_contains

subroutine assumes_absent_duplicate()
  !ERROR: At most one ABSENT clause can appear on ASSUMES directive
  !$omp assumes absent(parallel) absent(barrier)
end subroutine assumes_absent_duplicate

subroutine conditional_version()
  !WARNING: An argument to NO_PARALLELISM is an OpenMP v6.0 feature, try -fopenmp-version=60
  !$omp assumes no_parallelism(.true.)
end subroutine

!--- pure_assumes.f90
pure subroutine work()
  !ERROR: The OpenMP directive 'ASSUMES' is not allowed in a PURE procedure in OpenMP v5.1, try -fopenmp-version=52
  !$omp assumes no_openmp no_parallelism
end subroutine work

!--- conditional.f90
pure subroutine conditional_valid()
  logical, parameter :: enabled = .true., disabled = .false.
  !$omp assumes no_parallelism
  !$omp assumes no_parallelism(enabled .and. .not. disabled) absent(barrier)
  !$omp assumes no_parallelism(.false.) no_openmp
end subroutine

subroutine conditional_parameter_statement_later()
  logical :: enabled
  !$omp assumes no_parallelism(enabled)
  parameter (enabled = .true.)
end subroutine

subroutine conditional_parameter_declaration_later()
  !$omp assumes no_parallelism(enabled)
  logical, parameter :: enabled = .true.
end subroutine

subroutine conditional_host_shadowing()
  logical, parameter :: enabled = .true.
contains
  subroutine local_condition()
    !$omp assumes no_parallelism(enabled)
    logical, parameter :: enabled = .false.
  end subroutine
end subroutine

subroutine conditional_nonconstant(condition)
  !ERROR: Must be a constant value
  !$omp assumes no_parallelism(condition)
  logical :: condition
end subroutine

subroutine conditional_wrong_type()
  !ERROR: Must have LOGICAL type, but is INTEGER(4)
  !$omp assumes no_parallelism(1)
end subroutine

subroutine conditional_array()
  logical, parameter :: conditions(2) = [.true., .false.]
  !ERROR: Must be a scalar value, but is a rank-1 array
  !$omp assumes no_parallelism(conditions)
end subroutine

subroutine conditional_duplicate()
  !ERROR: At most one NO_PARALLELISM clause can appear on ASSUMES directive
  !$omp assumes no_parallelism(.true.) no_parallelism(.false.)
end subroutine

!--- placement.f90
module placement_module
  implicit none
  !$omp assumes no_openmp no_parallelism
  integer, parameter :: constant_value = 1
  interface
    module subroutine separate_work()
    end subroutine
  end interface
end module

submodule (placement_module) placement_submodule
  implicit none
  !ERROR: ASSUMES directive must appear in the specification part of a module or subprogram
  !$omp assumes no_parallelism
contains
  module procedure separate_work
  end procedure
end submodule

subroutine valid_placement()
  use placement_module, only: constant_value
  implicit none
  !$omp assumes no_parallelism
  integer :: value
  value = constant_value
contains
  subroutine after_import()
    import, only: constant_value
    implicit none
    !$omp assumes no_parallelism
  end subroutine
end subroutine

program main_assumes
  implicit none
  !ERROR: ASSUMES directive must appear in the specification part of a module or subprogram
  !$omp assumes no_parallelism
end program

block data block_data_assumes
  implicit none
  !ERROR: ASSUMES directive must appear in the specification part of a module or subprogram
  !$omp assumes no_openmp
end block data

subroutine block_assumes()
  block
    !ERROR: ASSUMES directive must appear in the specification part of a module or subprogram
    !$omp assumes absent(parallel)
  end block
end subroutine

subroutine before_use()
  !ERROR: ASSUMES directive must appear after all USE, IMPORT, and IMPLICIT statements
  !$omp assumes no_parallelism
  use placement_module, only: constant_value
end subroutine

subroutine import_host()
contains
  subroutine before_import()
    !ERROR: ASSUMES directive must appear after all USE, IMPORT, and IMPLICIT statements
    !$omp assumes no_openmp
    import, none
  end subroutine
end subroutine

subroutine before_implicit()
  !ERROR: ASSUMES directive must appear after all USE, IMPORT, and IMPLICIT statements
  !$omp assumes no_parallelism
  !ERROR: ASSUMES directive must appear after all USE, IMPORT, and IMPLICIT statements
  !$omp assumes absent(barrier)
  implicit none
end subroutine

!--- execution.f90
subroutine after_execution()
  integer :: value
  value = 0
  !ERROR: misplaced declaration in the execution part
  !$omp assumes no_parallelism
end subroutine
