!RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp %s | FileCheck %s

subroutine sub
  common /block/ a
  equivalence (b,c), (d,e), (a,f)
!$omp parallel firstprivate(/block/)
!$omp end parallel
end subroutine

!CHECK: a (Implicit, InCommonBlock) size=4 offset=0: ObjectEntity type: REAL(4)
!CHECK: b (Implicit) size=4 offset=0: ObjectEntity type: REAL(4)
!CHECK: c (Implicit) size=4 offset=0: ObjectEntity type: REAL(4)
!CHECK: d (Implicit) size=4 offset=4: ObjectEntity type: REAL(4)
!CHECK: e (Implicit) size=4 offset=4: ObjectEntity type: REAL(4)
!CHECK: f (Implicit) size=4 offset=0: ObjectEntity type: REAL(4)
!CHECK: sub (Subroutine): HostAssoc => sub (Subroutine): Subprogram ()
!CHECK: Equivalence Sets: (b,c) (d,e) (a,f)
!CHECK: block size=4 offset=0: CommonBlockDetails alignment=4: a
!CHECK: OtherConstruct scope:
!CHECK:   a (OmpFirstPrivate, OmpExplicit): HostAssoc => a (Implicit, InCommonBlock) size=4 offset=0: ObjectEntity type: REAL(4)
