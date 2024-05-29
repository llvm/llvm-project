! RUN: %flang_fc1 -fopenmp -fdebug-dump-symbols -o - %s 2>&1 | FileCheck %s
! Check intrinsic reduction symbols (in this case "max" are marked as INTRINSIC

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
  ! CHECK: k (OmpReduction): HostAssoc
  ! CHECK: max, INTRINSIC: ProcEntity
  !$omp parallel do  reduction(max:k)
  do i=1,10
    k = i
  end do
  !$omp end parallel do
end program omp_reduction
