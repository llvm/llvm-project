! RUN: %flang_fc1 -fopenmp -fdebug-dump-symbols -o - %s 2>&1 | FileCheck %s
! Check scan reduction

! CHECK: MainProgram scope: omp_reduction
program omp_reduction
  ! CHECK: i size=4 offset=0: ObjectEntity type: INTEGER(4)
  integer i
  ! CHECK: k size=4 offset=4: ObjectEntity type: INTEGER(4) init:10_4
  integer :: k = 10
  ! CHECK: m size=4 offset=8: ObjectEntity type: INTEGER(4) init:12_4
  integer :: m = 12

  ! CHECK: OtherConstruct scope
  ! CHECK: i (OmpPrivate, OmpPreDetermined): HostAssoc
  ! CHECK: k (OmpReduction, OmpInclusiveScan, OmpInScanReduction): HostAssoc
  !$omp parallel do  reduction(inscan, +:k)
  do i=1,10
   !$omp scan inclusive(k)
  end do
  !$omp end parallel do
  ! CHECK: m (OmpReduction, OmpExclusiveScan, OmpInScanReduction): HostAssoc
  !$omp parallel do  reduction(inscan, +:m)
  do i=1,10
   !$omp scan exclusive(m)
  end do
  !$omp end parallel do
end program omp_reduction
