! Test scheduling of WHERE in lower-hlfir-ordered-assignments pass.

! RUN: bbc -hlfir -o - -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments)" --debug-only=flang-ordered-assignment -flang-dbg-order-assignment-schedule-only %s 2>&1 | FileCheck %s
! REQUIRES: asserts

subroutine no_conflict(x, y)
  real :: x(:), y(:)
  where (y.gt.0) x = y
end subroutine

subroutine fake_conflict(x, y)
  ! The conflict here could be avoided because the read and write are
  ! aligned, so there would not be any read after write at the element
  ! level, but this will require a bit more work to detect this (like
  ! comparing the hlfir.designate operations).
  real :: x(:), y(:)
  where (x.gt.y) x = y
end subroutine

subroutine only_once(x, y, z)
  interface
    impure function call_me_only_once()
      logical :: call_me_only_once(10)
    end function
  end interface
  real :: x(:), y(:), z(:)
  where (call_me_only_once())
    x = y
    z = y
  end where
end subroutine

subroutine rhs_lhs_conflict(x, y)
  real :: x(:, :), y(:, :)
  where (y.gt.0.) x = transpose(x)
end subroutine

subroutine where_construct_no_conflict(x, y, z, mask1, mask2)
  real :: x(:), y(:), z(:)
  logical :: mask1(:), mask2(:)
  where (mask1)
    x = y
  elsewhere (mask2)
    z = y
  end where
end subroutine

subroutine where_construct_conflict(x, y)
  real :: x(:, :), y(:, :)
  where (y.gt.0.)
    x = y
  elsewhere (x.gt.0)
    y = x
  end where
end subroutine

subroutine where_construct_conflict_2(x, y)
  real :: x(:, :), y(:, :)
  where (x.gt.0.)
    x = y
  elsewhere (y.gt.0)
    y = x
  end where
end subroutine

subroutine where_vector_subscript_conflict_1(x, vec1)
  real :: x(10)
  integer :: vec1(10)
  where (x(vec1).lt.0.) x = 42.
end subroutine

subroutine where_vector_subscript_conflict_2(x, vec1)
  integer :: x(10)
  real :: y(10)
  where (y(x).lt.0.) x = 0
end subroutine

subroutine where_in_forall_conflict(x)
  real :: x(:, :)
  forall (i = 1:10)
    where (x(i, :).gt.0) x(:, i) = x(i, :)
  end forall
end subroutine

subroutine no_need_to_make_lhs_temp(x, y, i, j)
  integer :: j, i, x(:, :), y(:, :)
  call internal
contains
subroutine internal
  ! The internal procedure context currently gives a hard time to
  ! FIR alias analysis that flags the read of i,j and y as conflicting
  ! with the write to x. But this is not a reason to create a temporary
  ! storage for the LHS: the address is anyway fully computed in
  ! a descriptor (fir.box) before assigning any element of x.

  ! Note that the where mask is also saved while there is no real
  ! need to: it is addressing x elements in the same order as they
  ! are being assigned. But this will require more work in the
  ! conflict analysis to prove that the lowered DAG of `x(:, y(i, j))`
  ! are the same and that the access to this designator is done in the
  ! same ordered inside the mask and LHS.
  where (x(:, y(i, j)) == y(i, j))  x(:, y(i, j)) = 42
end subroutine
end subroutine

subroutine where_construct_unknown_conflict(x, mask)
  real :: x(:)
  logical :: mask(:)
  interface
     real function f()
     end function f
  end interface
  where (mask) x = f()
end subroutine

subroutine elsewhere_construct_unknown_conflict(x, y, mask1, mask2)
  real :: x(:), y(:)
  logical :: mask1(:), mask2(:)
  interface
     real function f()
     end function f
  end interface
  where (mask1)
     x = 1.0
  elsewhere (mask2)
     y = f()
  end where
end subroutine

!CHECK-LABEL: ------------ scheduling where in _QPno_conflict ------------
!CHECK-NEXT: run 1 evaluate: where/region_assign1
!CHECK-LABEL: ------------ scheduling where in _QPfake_conflict ------------
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?xf32>>' at index: 0 W:<block argument> of type '!fir.box<!fir.array<?xf32>>' at index: 0
!CHECK-NEXT: run 1 save    : where/mask
!CHECK-NEXT: run 2 evaluate: where/region_assign1
!CHECK-LABEL: ------------ scheduling where in _QPonly_once ------------
!CHECK-NEXT: unknown effect: %11 = fir.call @_QPcall_me_only_once() fastmath<contract> : () -> !fir.array<10x!fir.logical<4>>
!CHECK-NEXT: saving eval because write effect prevents re-evaluation
!CHECK-NEXT: run 1 save  (w): where/mask
!CHECK-NEXT: run 2 evaluate: where/region_assign1
!CHECK-NEXT: run 3 evaluate: where/region_assign2
!CHECK-LABEL: ------------ scheduling where in _QPrhs_lhs_conflict ------------
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?x?xf32>>' at index: 0 W:<block argument> of type '!fir.box<!fir.array<?x?xf32>>' at index: 0
!CHECK-NEXT: run 1 save    : where/region_assign1/rhs
!CHECK-NEXT: run 2 evaluate: where/region_assign1
!CHECK-LABEL: ------------ scheduling where in _QPwhere_construct_no_conflict ------------
!CHECK-NEXT: run 1 evaluate: where/region_assign1
!CHECK-NEXT: run 2 evaluate: where/elsewhere1/region_assign1
!CHECK-LABEL: ------------ scheduling where in _QPwhere_construct_conflict ------------
!CHECK-NEXT: run 1 evaluate: where/region_assign1
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?x?xf32>>' at index: 1 W:<block argument> of type '!fir.box<!fir.array<?x?xf32>>' at index: 1
!CHECK-NEXT: run 2 save    : where/mask
!CHECK-NEXT: run 3 evaluate: where/elsewhere1/region_assign1
!CHECK-LABEL: ------------ scheduling where in _QPwhere_construct_conflict_2 ------------
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?x?xf32>>' at index: 0 W:<block argument> of type '!fir.box<!fir.array<?x?xf32>>' at index: 0
!CHECK-NEXT: run 1 save    : where/mask
!CHECK-NEXT: run 2 evaluate: where/region_assign1
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?x?xf32>>' at index: 1 W:<block argument> of type '!fir.box<!fir.array<?x?xf32>>' at index: 1
!CHECK-NEXT: run 3 save    : where/elsewhere1/mask
!CHECK-NEXT: run 4 evaluate: where/elsewhere1/region_assign1
!CHECK-LABEL: ------------ scheduling where in _QPwhere_vector_subscript_conflict_1 ------------
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.ref<!fir.array<10xf32>>' at index: 0 W:<block argument> of type '!fir.ref<!fir.array<10xf32>>' at index: 0
!CHECK-NEXT: run 1 save    : where/mask
!CHECK-NEXT: run 2 evaluate: where/region_assign1
!CHECK-LABEL: ------------ scheduling where in _QPwhere_vector_subscript_conflict_2 ------------
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.ref<!fir.array<10xi32>>' at index: 0 W:<block argument> of type '!fir.ref<!fir.array<10xi32>>' at index: 0
!CHECK-NEXT: run 1 save    : where/mask
!CHECK-NEXT: run 2 evaluate: where/region_assign1
!CHECK-LABEL: ------------ scheduling forall in _QPwhere_in_forall_conflict ------------
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?x?xf32>>' at index: 0 W:<block argument> of type '!fir.box<!fir.array<?x?xf32>>' at index: 0
!CHECK-NEXT: run 1 save    : forall/where1/mask
!CHECK-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?x?xf32>>' at index: 0 W:<block argument> of type '!fir.box<!fir.array<?x?xf32>>' at index: 0
!CHECK-NEXT: run 1 save    : forall/where1/region_assign1/rhs
!CHECK-NEXT: run 2 evaluate: forall/where1/region_assign1
!CHECK-LABEL: ------------ scheduling where in _QFno_need_to_make_lhs_tempPinternal ------------
!CHECK-NEXT: conflict: R/W: %{{[0-9]+}} = fir.load %{{[0-9]+}} : !fir.llvm_ptr<!fir.ref<i32>> W:%{{[0-9]+}} = fir.load %{{[0-9]+}} : !fir.ref<!fir.box<!fir.array<?x?xi32>>>
!CHECK-NEXT: run 1 save    : where/mask
!CHECK-NEXT: run 2 evaluate: where/region_assign1
!CHECK-NEXT: ------------ scheduling where in _QPwhere_construct_unknown_conflict ------------
!CHECK-NEXT: unknown effect: %{{.*}} = fir.call @_QPf() fastmath<contract> : () -> f32
!CHECK-NEXT: conflict: R/W: %{{.*}} = hlfir.declare %{{.*}} {uniq_name = "_QFwhere_construct_unknown_conflictEmask"} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>) W:<unknown>
!CHECK-NEXT: run 1 save    : where/mask
!CHECK-NEXT: unknown effect: %{{.*}} = fir.call @_QPf() fastmath<contract> : () -> f32
!CHECK-NEXT: saving eval because write effect prevents re-evaluation
!CHECK-NEXT: run 2 save  (w): where/region_assign1/rhs
!CHECK-NEXT: run 3 evaluate: where/region_assign1
!CHECK-NEXT: ------------ scheduling where in _QPelsewhere_construct_unknown_conflict ------------
!CHECK-NEXT: run 1 evaluate: where/region_assign1
!CHECK-NEXT: unknown effect: %{{.*}} = fir.call @_QPf() fastmath<contract> : () -> f32
!CHECK-NEXT: conflict: R/W: %{{.*}} = hlfir.declare %{{.*}} {uniq_name = "_QFelsewhere_construct_unknown_conflictEmask1"} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>) W:<unknown>
!CHECK-NEXT: run 2 save    : where/mask
!CHECK-NEXT: conflict: R/W: %{{.*}} = hlfir.declare %{{.*}} {uniq_name = "_QFelsewhere_construct_unknown_conflictEmask2"} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>) W:<unknown>
!CHECK-NEXT: run 2 save    : where/elsewhere1/mask
!CHECK-NEXT: unknown effect: %{{.*}} = fir.call @_QPf() fastmath<contract> : () -> f32
!CHECK-NEXT: saving eval because write effect prevents re-evaluation
!CHECK-NEXT: run 3 save  (w): where/elsewhere1/region_assign1/rhs
!CHECK-NEXT: run 4 evaluate: where/elsewhere1/region_assign1
