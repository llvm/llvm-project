! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

! A user-defined reduction prevents the assumption of bounded reduction recipe
! effects, since they can access storage beyond the reduction operands even if
! those operands have intrinsic types.

module user_recipes
  interface operator(.combine.)
    integer function combine_integer(a, b)
      integer, intent(in) :: a, b
    end function
    real function combine_real(a, b)
      real, intent(in) :: a, b
    end function
  end interface
  !$omp declare reduction(sum : integer : omp_out = omp_out + omp_in) &
  !$omp & initializer(omp_priv = 0)
  !$omp declare reduction(.combine. : integer, real : omp_out = omp_out + omp_in) &
  !$omp & initializer(omp_priv = 0)
end module

! A renamed user reduction disables the assumption even when followed by
! a built-in reduction.
! CHECK-LABEL: func.func @_QPnamed_user_reduction(
! CHECK: omp.sections reduction(
! CHECK-NOT: reduction_assume_bounded_recipe_effects
! CHECK: } <private_assume_bounded_recipe_effects>
! CHECK-NOT: reduction_assume_bounded_recipe_effects
! CHECK: return
subroutine named_user_reduction(x, y, z)
  use user_recipes, only: renamed_sum => sum
  integer :: x, y, z, p
  !$omp parallel
    !$omp sections reduction(renamed_sum:x) reduction(+:y) &
    !$omp & lastprivate(conditional:z) private(p)
      p = 1
      x = x + p
      y = y + p
      z = p
    !$omp end sections
  !$omp end parallel
end subroutine

! A renamed user-defined operator also disables the assumption for both types.
! CHECK-LABEL: func.func @_QPoperator_user_reduction(
! CHECK: omp.sections reduction(
! CHECK-NOT: reduction_assume_bounded_recipe_effects
! CHECK: return
subroutine operator_user_reduction(i, r)
  use user_recipes, only: operator(.local.) => operator(.combine.)
  integer :: i
  real :: r
  !$omp sections reduction(.local.:i,r)
    i = i + 1
    r = r + 1.0
  !$omp end sections
end subroutine

! MAX is user-defined for INTEGER but built-in for REAL. Mixed types disable
! the assumption in either order; a subsequent REAL-only construct retains it.
! CHECK-LABEL: func.func @_QPshadowed_intrinsic_reduction(
! CHECK: omp.sections reduction(
! CHECK-NOT: reduction_assume_bounded_recipe_effects
! CHECK: omp.sections reduction(
! CHECK-NOT: reduction_assume_bounded_recipe_effects
! CHECK: omp.sections reduction(
! CHECK: } <reduction_assume_bounded_recipe_effects>
! CHECK: return
subroutine shadowed_intrinsic_reduction(i, r)
  integer :: i
  real :: r
  !$omp declare reduction(max : integer : omp_out = omp_out + omp_in) &
  !$omp & initializer(omp_priv = 0)

  !$omp sections reduction(max:r,i)
    i = i + 1
    r = max(r, 1.0)
  !$omp end sections

  !$omp sections reduction(max:i,r)
    i = i + 1
    r = max(r, 1.0)
  !$omp end sections

  !$omp sections reduction(max:r)
    r = max(r, 1.0)
  !$omp end sections
end subroutine

! A user-defined operator prevents the taskgroup assumption even when followed
! by a built-in reduction clause.
! CHECK-LABEL: func.func @_QPtaskgroup_operator_reduction(
! CHECK: omp.taskgroup task_reduction(
! CHECK-NOT: task_reduction_assume_bounded_recipe_effects
! CHECK: return
subroutine taskgroup_operator_reduction(i, r)
  use user_recipes, only: operator(.combine.)
  integer :: i
  real :: r
  !$omp taskgroup task_reduction(.combine.:i) task_reduction(+:r)
    i = i + 1
    r = r + 1.0
  !$omp end taskgroup
end subroutine

! Addition is user-defined for INTEGER but built-in for REAL. Mixed types
! prevent the assumption; a REAL-only taskgroup retains it.
! CHECK-LABEL: func.func @_QPtaskgroup_user_add_reduction(
! CHECK: omp.taskgroup task_reduction(
! CHECK-NOT: task_reduction_assume_bounded_recipe_effects
! CHECK: omp.taskgroup task_reduction(
! CHECK-NOT: task_reduction_assume_bounded_recipe_effects
! CHECK: omp.taskgroup task_reduction(
! CHECK: } <task_reduction_assume_bounded_recipe_effects>
! CHECK: return
subroutine taskgroup_user_add_reduction(i, r)
  integer :: i
  real :: r
  !$omp declare reduction(+ : integer : omp_out = omp_out + omp_in) &
  !$omp & initializer(omp_priv = 0)

  !$omp taskgroup task_reduction(+:r,i)
    i = i + 1
    r = r + 1.0
  !$omp end taskgroup

  !$omp taskgroup task_reduction(+:i,r)
    i = i + 1
    r = r + 1.0
  !$omp end taskgroup

  !$omp taskgroup task_reduction(+:r)
    r = r + 1.0
  !$omp end taskgroup
end subroutine
