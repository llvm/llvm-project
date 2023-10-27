! Test scheduling of WHERE in lower-hlfir-ordered-assignments pass
! when fusing is enabled or disabled.

!RUN: bbc -hlfir -o - -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments{fuse-assignments=false})" --debug-only=flang-ordered-assignment -flang-dbg-order-assignment-schedule-only %s 2>&1 | FileCheck %s --check-prefix NOFUSE

!RUN: bbc -hlfir -o - -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments{fuse-assignments=true})" --debug-only=flang-ordered-assignment -flang-dbg-order-assignment-schedule-only %s 2>&1 | FileCheck %s --check-prefix FUSE

!REQUIRES: asserts

subroutine fusable(x, y, mask)
  real :: x(:), y(:)
  logical :: mask(:)
  where (mask)
    x = 41.
    y = 42.
  end where
end subroutine

subroutine unfusable(x, y, mask)
  real :: x(:), y(:)
  logical :: mask(:)
  where (mask)
    x(1:10) = y
    y = x(10:1:-1)
  end where
end subroutine

!NOFUSE-LABEL: ------------ scheduling where in _QPfusable ------------
!NOFUSE-NEXT: run 1 evaluate: where/region_assign1
!NOFUSE-NEXT: run 2 evaluate: where/region_assign2
!NOFUSE-LABEL: ------------ scheduling where in _QPunfusable ------------
!NOFUSE-NEXT: run 1 evaluate: where/region_assign1
!NOFUSE-NEXT: run 2 evaluate: where/region_assign2

!FUSE-LABEL: ------------ scheduling where in _QPfusable ------------
!FUSE-NEXT: run 1 evaluate: where/region_assign1
!FUSE-NEXT: run 1 evaluate: where/region_assign2
!FUSE-LABEL: ------------ scheduling where in _QPunfusable ------------
!FUSE-NEXT: run 1 evaluate: where/region_assign1
!FUSE-NEXT: conflict: R/W: <block argument> of type '!fir.box<!fir.array<?xf32>>' at index: 1 W:<block argument> of type '!fir.box<!fir.array<?xf32>>' at index: 1
!FUSE-NEXT: run 2 evaluate: where/region_assign2
