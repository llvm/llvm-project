!RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=60 %s | FileCheck %s

subroutine f
  integer :: i, j, k
  !$omp do collapse(4)
  !$omp tile sizes(2, 2)
  do i = 1, 10
    do j = 1, 10
      do k = 1, 10
      end do
    end do
  end do
end

! Check that the k-loop is not affected.

!CHECK: Subprogram scope: f size=12 alignment=4 sourceRange=139 bytes
!CHECK:   f (Subroutine): HostAssoc => f (Subroutine): Subprogram ()
!CHECK:   i size=4 offset=0: ObjectEntity type: INTEGER(4)
!CHECK:   j size=4 offset=4: ObjectEntity type: INTEGER(4)
!CHECK:   k size=4 offset=8: ObjectEntity type: INTEGER(4)
!CHECK:   OtherConstruct scope: size=0 alignment=1 sourceRange=98 bytes
!CHECK:     i (OmpPrivate, OmpPreDetermined): HostAssoc => i size=4 offset=0: ObjectEntity type: INTEGER(4)
!CHECK:     j (OmpPrivate, OmpPreDetermined): HostAssoc => j size=4 offset=4: ObjectEntity type: INTEGER(4)

!CHECK-NOT:    k {{.*}}OmpPreDetermined
