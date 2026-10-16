! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

! The bounded-recipe-effects assumption is selected using the types of all
! reduction and privatized variables.
module recipe_memory_effects
  implicit none
  type :: record
    sequence
    integer :: value
  end type
contains
  ! Scalar and boxed intrinsic reductions and privatization are eligible.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPintrinsic_clauses(
  ! CHECK: omp.sections reduction(
  ! CHECK: } <assume_bounded_recipe_effects>
  subroutine intrinsic_clauses(x, y, p, f, l)
    integer :: x, p, f, l
    real :: y
    !$omp sections reduction(+:x,y) private(p) firstprivate(f) lastprivate(l)
      p = f
      l = p
      x = x + p
      y = y + 1.0
    !$omp end sections
  end subroutine

  ! A derived private variable prevents the assumption.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPderived_private(
  ! CHECK: omp.sections {
  ! CHECK-NOT: assume_bounded_recipe_effects
  ! CHECK: return
  subroutine derived_private()
    integer :: i
    type(record) :: z
    !$omp sections private(i, z)
      i = 1
      z%value = i
    !$omp end sections
  end subroutine

  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPintrinsic_private(
  ! CHECK: omp.sections {
  ! CHECK: } <assume_bounded_recipe_effects>
  subroutine intrinsic_private()
    integer :: x
    real :: y
    !$omp sections private(x, y)
      x = 1
      y = 1.0
    !$omp end sections
  end subroutine

  ! The generated conditional-lastprivate record is eligible.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPconditional_lastprivate(
  ! CHECK: omp.sections reduction(
  ! CHECK: } <assume_bounded_recipe_effects>
  subroutine conditional_lastprivate(x, y)
    integer :: x, y
    !$omp sections lastprivate(conditional:x) firstprivate(x) reduction(+:y)
      x = x + 1
      y = y + x
    !$omp end sections
  end subroutine

  ! A user-derived reduction still prevents the assumption.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPconditional_lastprivate_derived_reduction(
  ! CHECK: omp.sections reduction(byref
  ! CHECK-NOT: assume_bounded_recipe_effects
  ! CHECK: return
  subroutine conditional_lastprivate_derived_reduction(x, z)
    integer :: x
    type(record) :: z
    !$omp declare reduction(record_sum : record : &
    !$omp & omp_out%value = omp_out%value + omp_in%value) &
    !$omp & initializer(omp_priv%value = 0)
    !$omp parallel
      !$omp sections lastprivate(conditional:x) reduction(record_sum:z)
        x = 1
        z%value = z%value + x
      !$omp end sections
    !$omp end parallel
  end subroutine

  ! A derived private variable still prevents the assumption.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPconditional_lastprivate_derived_private(
  ! CHECK: omp.sections reduction(byref
  ! CHECK-NOT: assume_bounded_recipe_effects
  ! CHECK: return
  subroutine conditional_lastprivate_derived_private(x, z)
    integer :: x
    type(record) :: z
    !$omp parallel
      !$omp sections lastprivate(conditional:x) private(z)
        z%value = 1
        x = z%value
      !$omp end sections
    !$omp end parallel
  end subroutine
end module
