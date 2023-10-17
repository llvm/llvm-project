! Test optional fusing of forall assignments in the scheduling analysis
! from lower-hlfir-ordered-assignments pass. Assignments are fused in the
! same loop nest if they are given the same run id.

! RUN: bbc -hlfir -o - -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments{fuse-assignments=false})" --debug-only=flang-ordered-assignment -flang-dbg-order-assignment-schedule-only %s 2>&1 | FileCheck %s --check-prefix NOFUSE

! RUN: bbc -hlfir -o - -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments{fuse-assignments=true})" --debug-only=flang-ordered-assignment -flang-dbg-order-assignment-schedule-only %s 2>&1 | FileCheck %s --check-prefix FUSE

! REQUIRES: asserts

subroutine fusable_assign_easy(x, y, z)
  integer :: x(:), y(:), z(:)
  forall(i=1:10)
    x(i) = 42
    z(i) = 42
  end forall
end subroutine
!NOFUSE-LABEL: ------------ scheduling forall in _QPfusable_assign_easy ------------
!NOFUSE-NEXT: run 1 evaluate: forall/region_assign1
!NOFUSE-NEXT: run 2 evaluate: forall/region_assign2

!FUSE-LABEL: ------------ scheduling forall in _QPfusable_assign_easy ------------
!FUSE-NEXT: run 1 evaluate: forall/region_assign1
!FUSE-NEXT: run 1 evaluate: forall/region_assign2

subroutine fusable_assign(x, y, z)
  integer :: x(:), y(:), z(:)
  forall(i=1:10)
    x(i) = y(i)
    z(i) = y(11-i)
  end forall
end subroutine
!NOFUSE-LABEL: ------------ scheduling forall in _QPfusable_assign ------------
!NOFUSE-NEXT: run 1 evaluate: forall/region_assign1
!NOFUSE-NEXT: run 2 evaluate: forall/region_assign2

!FUSE-LABEL: ------------ scheduling forall in _QPfusable_assign ------------
!FUSE-NEXT: run 1 evaluate: forall/region_assign1
!FUSE-NEXT: run 1 evaluate: forall/region_assign2

subroutine unfusable_assign_1(x, y, z)
  integer :: x(:), y(:), z(:)
  forall(i=1:10)
    x(i) = y(i)
    z(i) = x(11-i)
  end forall
end subroutine
!NOFUSE-LABEL: ------------ scheduling forall in _QPunfusable_assign_1 ------------
!NOFUSE-NEXT: run 1 evaluate: forall/region_assign1
!NOFUSE-NEXT: run 2 evaluate: forall/region_assign2

!FUSE-LABEL: ------------ scheduling forall in _QPunfusable_assign_1 ------------
!FUSE-NEXT: run 1 evaluate: forall/region_assign1
!FUSE-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?xi32>>' at index: 0 W:<block argument> of type '!fir.box<!fir.array<?xi32>>' at index: 0
!FUSE-NEXT: run 2 evaluate: forall/region_assign2

subroutine unfusable_assign_2(x, y)
  integer :: x(:), y(:)
  forall(i=1:10)
    x(i) = y(i)
    x(i+1) = y(i+1)
  end forall
end subroutine
!NOFUSE-LABEL: ------------ scheduling forall in _QPunfusable_assign_2 ------------
!NOFUSE-NEXT: run 1 evaluate: forall/region_assign1
!NOFUSE-NEXT: run 2 evaluate: forall/region_assign2

!FUSE-LABEL: ------------ scheduling forall in _QPunfusable_assign_2 ------------
!FUSE-NEXT: run 1 evaluate: forall/region_assign1
!FUSE-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?xi32>>' at index: 0 W:<block argument> of type '!fir.box<!fir.array<?xi32>>' at index: 0
!FUSE-NEXT: run 2 evaluate: forall/region_assign2

subroutine unfusable_assign_3(x, y, z)
  integer :: x(:, :), y(:, :), z(:, :)
  forall(i=1:10)
    forall(j=1:z(i, i)) x(i, j) = y(i, j)
    z(i, :) = y(i, :)
  end forall
end subroutine
!NOFUSE-LABEL: ------------ scheduling forall in _QPunfusable_assign_3 ------------
!NOFUSE-NEXT: run 1 evaluate: forall/forall1/region_assign1
!NOFUSE-NEXT: run 2 evaluate: forall/region_assign1

!FUSE-LABEL: ------------ scheduling forall in _QPunfusable_assign_3 ------------
!FUSE-NEXT: run 1 evaluate: forall/forall1/region_assign1
!FUSE-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 2 W:<block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 2
!FUSE-NEXT: run 2 evaluate: forall/region_assign1

subroutine unfusable_assign_4(x, y, z)
  integer :: x(:, :), y(:, :), z(:, :)
  forall(i=1:10)
    x(i, :) = y(i, :)
    forall(j=1:x(i, i)) z(i, j) = y(i, j)
  end forall
end subroutine
!NOFUSE-LABEL: ------------ scheduling forall in _QPunfusable_assign_4 ------------
!NOFUSE-NEXT: run 1 evaluate: forall/region_assign1
!NOFUSE-NEXT: run 2 evaluate: forall/forall1/region_assign1

!FUSE-LABEL: ------------ scheduling forall in _QPunfusable_assign_4 ------------
!FUSE-NEXT: run 1 evaluate: forall/region_assign1
!FUSE-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 0 W:<block argument> of type '!fir.box<!fir.array<?x?xi32>>' at index: 0
!FUSE-NEXT: run 2 evaluate: forall/forall1/region_assign1
