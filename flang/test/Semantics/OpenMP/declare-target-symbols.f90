!RUN: %flang_fc1 -fdebug-dump-symbols %openmp_flags %s | FileCheck %s

function baz(a)
  !$omp declare target to(baz)
  real, intent(in) :: a
  baz = a
end

subroutine foo
  !$omp declare target(baz)
  integer, save :: baz(10)
end

subroutine bar
  real :: a
  !$omp declare target(baz)
  a = 1.0
  !$omp target
    a = baz(a)
  !$omp end target
end

!CHECK: Subprogram scope: baz size=8 alignment=4
!CHECK:   a, INTENT(IN) size=4 offset=4: ObjectEntity dummy type: REAL(4)
!CHECK:   baz (Implicit, OmpDeclareTarget) size=4 offset=0: ObjectEntity funcResult type: REAL(4)
!CHECK:   OtherClause scope: size=0 alignment=1
!CHECK: Subprogram scope: foo size=40 alignment=4 sourceRange=69 bytes
!CHECK:   baz, SAVE (OmpDeclareTarget) size=40 offset=0: ObjectEntity type: INTEGER(4) shape: 1_8:10_8 OmpDeclareTargetFlags:(enter)
!CHECK:   foo (Subroutine): HostAssoc => foo (Subroutine): Subprogram ()
!CHECK: Subprogram scope: bar size=4 alignment=4
!CHECK:   a size=4 offset=0: ObjectEntity type: REAL(4)
!CHECK:   bar (Subroutine): HostAssoc => bar (Subroutine): Subprogram ()
!CHECK:   baz, EXTERNAL (Function, OmpDeclareTarget): HostAssoc => baz, EXTERNAL (Function, OmpDeclareTarget): Subprogram result:REAL(4) baz (REAL(4) a) OmpDeclareTargetFlags:(enter to)
!CHECK:   OtherConstruct scope: size=0 alignment=1
!CHECK:     baz, EXTERNAL (Function, OmpDeclareTarget): HostAssoc => baz, EXTERNAL (Function, OmpDeclareTarget): Subprogram result:REAL(4) baz (REAL(4) a) OmpDeclareTargetFlags:(enter to)
