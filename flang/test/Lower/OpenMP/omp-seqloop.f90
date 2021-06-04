! This test checks lowering of OpenMP DO Directive(Worksharing).

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

!FIRDialect: func @_QQmain()
program wsloop
        integer :: i

!FIRDialect:  omp.parallel {
        !$OMP PARALLEL
!FIRDialect:    %[[PRIVATE_INDX:.*]] = fir.alloca i32 {{{.*}}uniq_name = "i"}
!FIRDialect:    %[[FINAL_INDX:.*]] = fir.do_loop %[[INDX:.*]] = {{.*}} {
        do i=1, 9
        print*, i
        end do
!FIRDialect:    }
!FIRDialect:    %[[FINAL_INDX_CVT:.*]] = fir.convert %[[FINAL_INDX]] : (index) -> i32
!FIRDialect:    fir.store %[[FINAL_INDX_CVT]] to %[[PRIVATE_INDX]] : !fir.ref<i32>
!FIRDialect:    omp.terminator
!FIRDialect:  }
        !$OMP END PARALLEL
end

!FIRDialect: func @_QPsub1() {
subroutine sub1
!FIRDialect:   {{.*}} = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsub1Ei"}
  integer :: i
  integer :: arr(10)
!FIRDialect:   omp.parallel {
!FIRDialect:     {{.*}} = fir.alloca i32 {uniq_name = "i"}
  !$OMP PARALLEL
  do i=1, 10
    arr(i) = i
  end do
!FIRDialect:     omp.terminator
!FIRDialect:   }
  !$OMP END PARALLEL
end subroutine

!FIRDialect: func @_QPsub2() {
subroutine sub2
!FIRDialect:   {{.*}} = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsub2Ei"}
  integer :: i
  integer :: arr(10)
!FIRDialect:   omp.parallel {
!FIRDialect:     {{.*}} = fir.alloca i32 {uniq_name = "i"}
  !$OMP PARALLEL
!FIRDialect:     omp.master  {
  !$OMP MASTER
  do i=1, 10
    arr(i) = i
  end do
!FIRDialect:       omp.terminator
!FIRDialect:     }
  !$OMP END MASTER
!FIRDialect:     omp.terminator
!FIRDialect:   }
  !$OMP END PARALLEL
end subroutine


!FIRDialect: func @_QPsub3() {
subroutine sub3
!FIRDialect:   {{.*}} = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsub3Ei"}
!FIRDialect:   {{.*}} = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFsub3Ej"}
  integer :: i,j
  integer :: arr(10)
!FIRDialect:   omp.parallel {
!FIRDialect:     {{.*}} = fir.alloca i32 {uniq_name = "i"}
  !$OMP PARALLEL
  do i=1, 10
    arr(i) = i
  end do
!FIRDialect:     omp.master  {
  !$OMP MASTER
!FIRDialect:       omp.parallel {
!FIRDialect:         {{.*}} = fir.alloca i32 {uniq_name = "j"}
  !$OMP PARALLEL
  do j=1, 10
    arr(j) = j
  end do
!FIRDialect:         omp.terminator
!FIRDialect:       }
  !$OMP END PARALLEL
!FIRDialect:       omp.terminator
!FIRDialect:     }
  !$OMP END MASTER
!FIRDialect:     omp.terminator
!FIRDialect:   }
  !$OMP END PARALLEL
end subroutine
