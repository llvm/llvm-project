! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

! Each clause has its own bounded-recipe-effects assumption, which requires
! all variables in that clause to be non-derived and non-allocatable.
module recipe_memory_effects
  implicit none
  type :: record
    sequence
    integer :: value
  end type
contains
  ! Intrinsic private and reduction variables are eligible independently.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPintrinsic_clauses(
  ! CHECK: omp.sections reduction(
  ! CHECK: } <private_assume_bounded_recipe_effects, reduction_assume_bounded_recipe_effects>
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

  ! Non-derived private, firstprivate and lastprivate variables are eligible
  ! without a reduction clause.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPintrinsic_private(
  ! CHECK: omp.sections {
  ! CHECK: } <private_assume_bounded_recipe_effects>
  ! CHECK: return
  subroutine intrinsic_private(p, f, l)
    integer :: p, f, l
    !$omp sections private(p) firstprivate(f) lastprivate(l)
      p = f
      l = p
    !$omp end sections
  end subroutine

  ! An allocatable variable prevents the assumption for the whole private clause.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPboxed_private(
  ! CHECK: omp.sections {
  ! CHECK-NOT: assume_bounded_recipe_effects
  ! CHECK: return
  subroutine boxed_private(a, p)
    real, allocatable :: a(:)
    real, pointer :: p
    !$omp sections private(a, p)
      a = 1.0
      nullify(p)
    !$omp end sections
  end subroutine

  ! Scalar allocatables are also excluded, independently of the other clause.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPallocatable_private_scalar_reduction(
  ! CHECK: omp.sections reduction(
  ! CHECK-NOT: private_assume_bounded_recipe_effects
  ! CHECK: } <reduction_assume_bounded_recipe_effects>
  ! CHECK: return
  subroutine allocatable_private_scalar_reduction(a, x)
    real, allocatable :: a
    integer :: x
    !$omp sections private(a) reduction(+:x)
      x = x + 1
    !$omp end sections
  end subroutine

  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPallocatable_reduction_scalar_private(
  ! CHECK: omp.sections reduction(
  ! CHECK-NOT: reduction_assume_bounded_recipe_effects
  ! CHECK: } <private_assume_bounded_recipe_effects>
  ! CHECK: return
  subroutine allocatable_reduction_scalar_private(a, x)
    real, allocatable :: a
    integer :: x
    !$omp sections reduction(+:a) private(x)
      x = 1
      a = a + x
    !$omp end sections
  end subroutine

  ! A derived private variable prevents the assumption for the private clause.
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

  ! The generated conditional-lastprivate record is eligible.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPconditional_lastprivate(
  ! CHECK: omp.sections reduction(
  ! CHECK: } <reduction_assume_bounded_recipe_effects>
  subroutine conditional_lastprivate(x, y)
    integer :: x, y
    !$omp sections lastprivate(conditional:x) firstprivate(x) reduction(+:y)
      x = x + 1
      y = y + x
    !$omp end sections
  end subroutine

  ! A derived reduction prevents the assumption for the whole reduction clause,
  ! independently of the non-derived private variables.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPconditional_lastprivate_derived_reduction(
  ! CHECK: omp.sections reduction({{(byref )?}}@
  ! CHECK-NOT: reduction_assume_bounded_recipe_effects
  ! CHECK: } <private_assume_bounded_recipe_effects>
  ! CHECK-NOT: reduction_assume_bounded_recipe_effects
  ! CHECK: return
  subroutine conditional_lastprivate_derived_reduction(x, z)
    integer :: x, p
    type(record) :: z
    !$omp declare reduction(record_sum : record : &
    !$omp & omp_out%value = omp_out%value + omp_in%value) &
    !$omp & initializer(omp_priv%value = 0)
    !$omp parallel
      !$omp sections lastprivate(conditional:x) reduction(record_sum:z) private(p)
        p = 1
        x = p
        z%value = z%value + x
      !$omp end sections
    !$omp end parallel
  end subroutine

  ! A common block is eligible when all its members are non-derived.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPintrinsic_common_private(
  ! CHECK: omp.sections {
  ! CHECK: } <private_assume_bounded_recipe_effects>
  ! CHECK: return
  subroutine intrinsic_common_private()
    integer :: x
    real :: y
    common /intrinsic_block/ x, y
    !$omp sections private(/intrinsic_block/)
      x = 1
      y = 1.0
    !$omp end sections
  end subroutine

  ! A derived common-block member prevents the private-clause assumption.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPderived_common_private(
  ! CHECK: omp.sections {
  ! CHECK-NOT: assume_bounded_recipe_effects
  ! CHECK: return
  subroutine derived_common_private()
    integer :: x
    type(record) :: z
    common /derived_block/ x, z
    !$omp sections private(/derived_block/)
      x = 1
      z%value = x
    !$omp end sections
  end subroutine
end module
