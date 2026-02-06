! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s

program wsloop_collapse_continue
  integer i, j

! CHECK: omp.wsloop {{.*}} {
! CHECK: omp.loop_nest ({{.*}}) : i32 = ({{.*}}) to ({{.*}}) inclusive step ({{.*}}) collapse(2) {
  !$omp do collapse(2)
  do 50 i = 1, 42
     do 51 j = 1, 84
! CHECK: fir.call @_FortranAioOutputInteger32(
        print *, i
! CHECK: fir.call @_FortranAioOutputInteger32(
        print *, j
     51 continue
  50 continue
  !$omp end do

end program wsloop_collapse_continue
