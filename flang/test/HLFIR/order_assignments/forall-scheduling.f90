! Test forall scheduling analysis from lower-hlfir-ordered-assignments pass.
! The printed output is done via LLVM_DEBUG, hence the "asserts" requirement.
! This test test that conflicting actions are not scheduled to be evaluated
! in the same loops (same run id).

! RUN: bbc -hlfir -o - -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments)" --debug-only=flang-ordered-assignment -flang-dbg-order-assignment-schedule-only %s 2>&1 | FileCheck %s
! REQUIRES: asserts

subroutine no_conflict(x)
  real :: x(:)
  forall(i=1:10) x(i) = i
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPno_conflict ------------
!CHECK-NEXT: run 1 evaluate: forall/region_assign1

subroutine rhs_lhs_overlap(x)
  real :: x(:)
  forall(i=1:10) x(i) = x(11-i)
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPrhs_lhs_overlap ------------
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?xf32>>' at index: 0 W:<block argument> of type '!fir.box<!fir.array<?xf32>>' at index: 0
!CHECK-NEXT: run 1 save    : forall/region_assign1/rhs
!CHECK-NEXT: run 2 evaluate: forall/region_assign1

subroutine no_rhs_lhs_overlap(x, y)
  real :: x(:), y(:)
  forall(i=1:10) x(i) = y(i)
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPno_rhs_lhs_overlap ------------
!CHECK-NEXT: run 1 evaluate: forall/region_assign1

subroutine no_rhs_lhs_overlap_2(x)
  real :: x(:), y(10)
  forall(i=1:10) x(i) = y(i)
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPno_rhs_lhs_overlap_2 ------------
!CHECK-NEXT: run 1 evaluate: forall/region_assign1

subroutine no_rhs_lhs_overlap_3()
  real :: x(10), y(10)
  forall(i=1:10) x(i) = y(i)
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPno_rhs_lhs_overlap_3 ------------
!CHECK-NEXT: run 1 evaluate: forall/region_assign1

subroutine array_expr_rhs_lhs_overlap(x)
  real :: x(:, :)
  forall(i=1:10) x(i, :) = x(:, i)*2
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QParray_expr_rhs_lhs_overlap ------------
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?x?xf32>>' at index: 0 W:<block argument> of type '!fir.box<!fir.array<?x?xf32>>' at index: 0
!CHECK-NEXT: run 1 save    : forall/region_assign1/rhs
!CHECK-NEXT: run 2 evaluate: forall/region_assign1

subroutine array_expr_no_rhs_lhs_overlap(x, y, z)
  real :: x(:, :), y(:, :), z(:, :)
  forall(i=1:10) x(i, :) = y(:, i) + z(i, :)
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QParray_expr_no_rhs_lhs_overlap ------------
!CHECK-NEXT: run 1 evaluate: forall/region_assign1

subroutine rhs_lhs_overlap_2(x, y)
  real, target :: x(:), y(:)
  forall(i=1:10) x(i) = y(i)
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPrhs_lhs_overlap_2 ------------
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?xf32>>' at index: 1 W:<block argument> of type '!fir.box<!fir.array<?xf32>>' at index: 0
!CHECK-NEXT: run 1 save    : forall/region_assign1/rhs
!CHECK-NEXT: run 2 evaluate: forall/region_assign1

subroutine lhs_lhs_overlap(x)
  integer :: x(10)
  forall(i=1:10) x(x(i)) = i
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPlhs_lhs_overlap ------------
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.ref<!fir.array<10xi32>>' at index: 0 W:<block argument> of type '!fir.ref<!fir.array<10xi32>>' at index: 0
!CHECK-NEXT: run 1 save    : forall/region_assign1/lhs
!CHECK-NEXT: run 2 evaluate: forall/region_assign1

subroutine unknown_function_call(x)
  interface
    pure real function foo(x, i)
      integer, intent(in) :: i
      real, intent(in) :: x(10)
    end function
  end interface
  real :: x(10)
  forall(i=1:10) x(i) = foo(x, i)
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPunknown_function_call ------------
!CHECK-NEXT: unknown effect: {{.*}} fir.call @_QPfoo
!CHECK-NEXT: conflict: R/W: <unknown> W:<block argument> of type '!fir.ref<!fir.array<10xf32>>' at index: 0
!CHECK-NEXT: run 1 save    : forall/region_assign1/rhs
!CHECK-NEXT: run 2 evaluate: forall/region_assign1

subroutine unknown_function_call2(x)
  interface
    pure real function foo2(i)
      integer, value :: i
    end function
  end interface
  ! foo2 may read x since it is a target, even if it is pure,
  ! if the actual argument of x is a module variable accessible
  ! to foo via host association.
  real, target :: x(:)
  forall(i=1:10) x(i) = foo2(i)
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPunknown_function_call2 ------------
!CHECK-NEXT: unknown effect: {{.*}} fir.call @_QPfoo2(
!CHECK-NEXT: conflict: R/W: <unknown> W:<block argument> of type '!fir.box<!fir.array<?xf32>>' at index: 0
!CHECK-NEXT: run 1 save    : forall/region_assign1/rhs
!CHECK-NEXT: run 2 evaluate: forall/region_assign1

subroutine forall_mask_conflict(x)
  integer :: x(:)
  forall(i=1:10, x(11-i)>0) x(i) = 42
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPforall_mask_conflict ------------
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?xi32>>' at index: 0 W:<block argument> of type '!fir.box<!fir.array<?xi32>>' at index: 0
!CHECK-NEXT: run 1 save    : forall/forall_mask1/mask
!CHECK-NEXT: run 2 evaluate: forall/forall_mask1/region_assign1

subroutine forall_ub_conflict(x, y)
  integer :: x(:, :)
  forall(i=1:10)
    forall(j=1:x(i,i))
      x(i, j) = 42
    end forall
  end forall
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPforall_ub_conflict ------------
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 0 W:<block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 0
!CHECK-NEXT: run 1 save    : forall/forall1/ub
!CHECK-NEXT: run 2 evaluate: forall/forall1/region_assign1

subroutine sequential_assign(x, y)
  integer :: x(:), y(:)
  forall(i=1:10)
    x(i) = y(i)
    y(2*i) = x(i)
  end forall
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPsequential_assign ------------
!CHECK-NEXT: run 1 evaluate: forall/region_assign1
!CHECK-NEXT: run 2 evaluate: forall/region_assign2

subroutine loads_of_conlficts(x, y)
  integer, target :: x(:, :), y(:, :)
  forall(i=1:10)
    forall (j=1:y(i,i)) x(x(i, j), j) = y(i, j)
    forall (j=1:x(i,i), y(i,i)>0) y(x(i, j), j) = 0
  end forall
end subroutine
!CHECK-LABEL: ------------ scheduling forall in _QPloads_of_conlficts ------------
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 1 W:<block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 0
!CHECK-NEXT: run 1 save    : forall/forall1/ub
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 1 W:<block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 0
!CHECK-NEXT: run 1 save    : forall/forall1/region_assign1/rhs
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 0 W:<block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 0
!CHECK-NEXT: run 1 save    : forall/forall1/region_assign1/lhs
!CHECK-NEXT: run 2 evaluate: forall/forall1/region_assign1
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 0 W:<block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 1
!CHECK-NEXT: run 3 save    : forall/forall2/ub
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 1 W:<block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 1
!CHECK-NEXT: run 3 save    : forall/forall2/forall_mask1/mask
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 0 W:<block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 1
!CHECK-NEXT: run 3 save    : forall/forall2/forall_mask1/region_assign1/lhs
!CHECK-NEXT: run 4 evaluate: forall/forall2/forall_mask1/region_assign1
