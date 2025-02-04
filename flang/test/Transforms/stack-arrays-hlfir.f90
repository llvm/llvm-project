! Similar to stack-arrays.f90; i.e. both test the stack-arrays pass for different
! kinds of supported inputs. This one differs in that it takes the hlfir lowering
! path in flag rather than the fir one. For example, temp arrays are lowered
! differently in hlfir vs. fir and the IR that reaches the stack arrays pass looks
! quite different.


! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - \
! RUN: | fir-opt --lower-hlfir-ordered-assignments \
! RUN:           --bufferize-hlfir \
! RUN:           --convert-hlfir-to-fir \
! RUN:           --array-value-copy \
! RUN:           --stack-arrays \
! RUN: | FileCheck %s

subroutine temp_array
  implicit none
  integer (8) :: lV
  integer (8), dimension (2) :: iaVS

  lV = 202

  iaVS = [lV, lV]
end subroutine temp_array
! CHECK-LABEL: func.func @_QPtemp_array{{.*}} {
! CHECK-NOT:     fir.allocmem
! CHECK-NOT:     fir.freemem
! CHECK:         fir.alloca !fir.array<2xi64>
! CHECK-NOT:     fir.allocmem
! CHECK-NOT:     fir.freemem
! CHECK:         return
! CHECK-NEXT:  }

subroutine omp_temp_array
  implicit none
  integer (8) :: lV
  integer (8), dimension (2) :: iaVS

  lV = 202

  !$omp target
    iaVS = [lV, lV]
  !$omp end target
end subroutine omp_temp_array
! CHECK-LABEL: func.func @_QPomp_temp_array{{.*}} {
! CHECK:         omp.target {{.*}} {
! CHECK-NOT:       fir.allocmem
! CHECK-NOT:       fir.freemem
! CHECK:           fir.alloca !fir.array<2xi64>
! CHECK-NOT:       fir.allocmem
! CHECK-NOT:       fir.freemem
! CHECK:           omp.terminator
! CHECK-NEXT:    }
! CHECK:         return
! CHECK-NEXT:  }

subroutine omp_target_wsloop
  implicit none
  integer (8) :: lV, i
  integer (8), dimension (2) :: iaVS

  lV = 202

  !$omp target teams distribute
  do i = 1, 10
    iaVS = [lV, lV]
  end do
  !$omp end target teams distribute
end subroutine omp_target_wsloop
! CHECK-LABEL: func.func @_QPomp_target_wsloop{{.*}} {
! CHECK:         omp.target {{.*}} {
! CHECK-NOT:       fir.allocmem
! CHECK-NOT:       fir.freemem
! CHECK:           fir.alloca !fir.array<2xi64>
! CHECK:         omp.teams {
! CHECK:         omp.distribute {
! CHECK:         omp.loop_nest {{.*}} {
! CHECK-NOT:       fir.allocmem
! CHECK-NOT:       fir.freemem
! CHECK:           omp.yield
! CHECK-NEXT:    }
! CHECK:         return
! CHECK-NEXT:  }
