! RUN: %flang_fc1 -fopenmp -fdebug-dump-symbols %s | FileCheck %s

program main
  !CHECK: a (InCommonBlock) size=4 offset=0: ObjectEntity type: REAL(4)
  !CHECK: b (InCommonBlock) size=8 offset=4: ObjectEntity type: INTEGER(4) shape: 1_8:2_8
  !CHECK: c (InCommonBlock) size=4 offset=12: ObjectEntity type: REAL(4)
  !CHECK: blk size=16 offset=0: CommonBlockDetails alignment=4: a b c
  real :: a, c
  integer :: b(2)
  common /blk/ a, b, c
  !$omp parallel private(/blk/)
    !CHECK: OtherConstruct scope: size=0 alignment=1
    !CHECK:   a (OmpPrivate, OmpExplicit): HostAssoc
    !CHECK:   b (OmpPrivate, OmpExplicit): HostAssoc
    !CHECK:   c (OmpPrivate, OmpExplicit): HostAssoc
    call sub(a, b, c)
  !$omp end parallel
end program
