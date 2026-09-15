! Test that the lower-hlfir-ordered-assignments pass falls back to the
! 1D HomogeneousScalarStack temporary (counter-based) when the FORALL loop
! nest is deeper than Fortran::common::maxRank (15), because fir.array can
! only hold up to maxRank dimensions.
!
! Below maxRank, the new ArrayTemp is used and there is no counter; here we
! verify the opposite: the counter (a fir.alloca index, fir.load/addi/store
! pattern) is restored when the loop nest has 16 levels.
!
! The test uses a rank-8 array of derived type with a rank-8 array component
! to spread 16 indexable dimensions across the FORALL header.
!
! RUN: bbc -emit-hlfir -o - %s | fir-opt --lower-hlfir-ordered-assignments | FileCheck %s

module many_forall_mod
  type :: t
    real :: c(2,2,2,2,2,2,2,2)
  end type
contains
  subroutine more_than_15_forall(a)
    type(t), intent(inout) :: a(2,2,2,2,2,2,2,2)
    forall (i1=1:2, i2=1:2, i3=1:2, i4=1:2, i5=1:2, i6=1:2, i7=1:2, i8=1:2, &
            j1=1:2, j2=1:2, j3=1:2, j4=1:2, j5=1:2, j6=1:2, j7=1:2, j8=1:2)
      a(i1,i2,i3,i4,i5,i6,i7,i8)%c(j1,j2,j3,j4,j5,j6,j7,j8) = &
        a(3-i1,3-i2,3-i3,3-i4,3-i5,3-i6,3-i7,3-i8)%c(3-j1,3-j2,3-j3,3-j4,3-j5,3-j6,3-j7,3-j8)
    end forall
  end subroutine
end module
! With 16 nested loops, the temporary must be the 1D counter-based form
! (HomogeneousScalarStack) instead of a 16D ArrayTemp, since fir.array is
! limited to Fortran::common::maxRank dimensions.
!
! CHECK-LABEL: func.func @_QMmany_forall_modPmore_than_15_forall(
! There must be a counter in memory (fir.alloca index).
! CHECK:         %[[CTR:.*]] = fir.alloca index
! The temporary is a 1D fir.array<?xf32>.
! CHECK:         %[[ALLOC:.*]] = fir.allocmem !fir.array<?xf32>, %{{.*}} {bindc_name = ".tmp.forall", uniq_name = ""}
! Plain fir.shape (no shift), since the temp is indexed by the counter.
! CHECK:         %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:         hlfir.declare %[[ALLOC]](%[[SHAPE]]) {uniq_name = ".tmp.forall"} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.heap<!fir.array<?xf32>>)
! Inside the loop nest the counter is incremented and the temp is indexed
! through the counter (not directly through the loop induction variables).
! CHECK:         fir.load %[[CTR]] : !fir.ref<index>
! CHECK:         arith.addi %{{.*}}, %{{.*}} : index
! CHECK:         fir.store %{{.*}} to %[[CTR]] : !fir.ref<index>
