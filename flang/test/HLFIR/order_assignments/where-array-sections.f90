! Test scheduling of WHERE with aligned array sections.

!RUN: bbc -hlfir -o - -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments{fuse-assignments=false})" --debug-only=flang-ordered-assignment -flang-dbg-order-assignment-schedule-only %s 2>&1 | FileCheck %s --check-prefix NOFUSE

!RUN: bbc -hlfir -o - -pass-pipeline="builtin.module(lower-hlfir-ordered-assignments{fuse-assignments=true})" --debug-only=flang-ordered-assignment -flang-dbg-order-assignment-schedule-only %s 2>&1 | FileCheck %s --check-prefix FUSE

!REQUIRES: asserts

subroutine no_temps(var1, var2, var3)
  implicit none
  real, contiguous, dimension(:,:) :: var1, var2
  real, contiguous, dimension(:) :: var3

  where (var2(:,2) < 0.)
    var2(:,1) = var2(:,1) + var2(:,2)
    var2(:,1) = var2(:,2)
    var3(:) = var3(:) - var2(:,2)
    var2(:,2) = 0.
  end where
end

subroutine must_create_mask_temp_if_not_fused(var1, var2, var3)
  implicit none
  real, contiguous, dimension(:,:) :: var1, var2
  real, contiguous, dimension(:) :: var3

  where (var2(:,2) < 0.)
    var2(:,1) = var2(:,1) + var2(:,2)
    var2(:,2) = 0. ! -> modifies mask 1-1 
    var2(:,1) = var2(:,2)
    var3(:) = var3(:) - var2(:,2)
  end where
end

subroutine must_split_and_create_temps(var1, var2, var3)
  implicit none
  real, contiguous, dimension(:,:) :: var1, var2
  real, contiguous, dimension(:) :: var3

  where (var2(:,2) < 0.)
    var2(:,1) = var2(:,1) + var2(:,2)
    var2(:,2) = 0. ! -> modifies mask 1-1
    ! RHS/LHS overlap require saving RHS and splitting loops, which requires
    ! also saving the mask before the assignment above.
    var2(:,1) = var2(2,:) + var2(2,:)
    var3(:) = var3(:) - var2(:,2)
  end where
end

!NOFUSE-LABEL: ------------ scheduling where in _QPno_temps ------------
!NOFUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!NOFUSE-NEXT: run 1 evaluate: where/region_assign1
!NOFUSE-NEXT: run 2 evaluate: where/region_assign2
!NOFUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg2 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>> W:%{{.*}} = fir.box_addr %arg2 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
!NOFUSE-NEXT: run 3 evaluate: where/region_assign3
!NOFUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!NOFUSE-NEXT: run 4 evaluate: where/region_assign4
!NOFUSE-LABEL: ------------ scheduling where in _QPmust_create_mask_temp_if_not_fused ------------
!NOFUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!NOFUSE-NEXT: run 1 evaluate: where/region_assign1
!NOFUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!NOFUSE-NEXT: run 2 evaluate: where/region_assign2
!NOFUSE-NEXT: where/mask is modified in order by where/region_assign2 and is needed by where/region_assign3 that is scheduled in a later run
!NOFUSE-NEXT: run 0 save    : where/mask
!NOFUSE-NEXT: run 4 evaluate: where/region_assign3
!NOFUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg2 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>> W:%{{.*}} = fir.box_addr %arg2 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
!NOFUSE-NEXT: run 5 evaluate: where/region_assign4
!NOFUSE-LABEL: ------------ scheduling where in _QPmust_split_and_create_temps ------------
!NOFUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!NOFUSE-NEXT: run 1 evaluate: where/region_assign1
!NOFUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!NOFUSE-NEXT: run 2 evaluate: where/region_assign2
!NOFUSE-NEXT: conflicting arrays:%{{.*}} and %{{.*}}
!NOFUSE-NEXT: run 3 save    : where/region_assign3/rhs
!NOFUSE-NEXT: where/mask is modified in order by where/region_assign2 and is needed by where/region_assign3 that is scheduled in a later run
!NOFUSE-NEXT: run 0 save    : where/mask
!NOFUSE-NEXT: run 5 evaluate: where/region_assign3
!NOFUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg2 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>> W:%{{.*}} = fir.box_addr %arg2 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
!NOFUSE-NEXT: run 6 evaluate: where/region_assign4

!FUSE-LABEL: ------------ scheduling where in _QPno_temps ------------
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: run 1 evaluate: where/region_assign1
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: run 1 evaluate: where/region_assign2
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg2 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>> W:%{{.*}} = fir.box_addr %arg2 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
!FUSE-NEXT: run 1 evaluate: where/region_assign3
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: run 1 evaluate: where/region_assign4
!FUSE-LABEL: ------------ scheduling where in _QPmust_create_mask_temp_if_not_fused ------------
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: run 1 evaluate: where/region_assign1
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: run 1 evaluate: where/region_assign2
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: run 1 evaluate: where/region_assign3
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg2 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>> W:%{{.*}} = fir.box_addr %arg2 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: run 1 evaluate: where/region_assign4
!FUSE-LABEL: ------------ scheduling where in _QPmust_split_and_create_temps ------------
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: run 1 evaluate: where/region_assign1
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>> W:%{{.*}} = fir.box_addr %arg1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
!FUSE-NEXT: run 1 evaluate: where/region_assign2
!FUSE-NEXT: conflicting arrays:%{{.*}} and %{{.*}}
!FUSE-NEXT: run 2 save    : where/region_assign3/rhs
!FUSE-NEXT: where/mask is modified in order by where/region_assign1, where/region_assign2 and is needed by where/region_assign3 that is scheduled in a later run
!FUSE-NEXT: run 0 save    : where/mask
!FUSE-NEXT: run 4 evaluate: where/region_assign3
!FUSE-NEXT: conflict (aligned): R/W: %{{.*}} = fir.box_addr %arg2 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>> W:%{{.*}} = fir.box_addr %arg2 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
!FUSE-NEXT: run 4 evaluate: where/region_assign4
