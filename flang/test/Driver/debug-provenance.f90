! Ensure argument -fdebug-dump-provenance works as expected.

! RUN: %flang_fc1 -fdebug-dump-provenance %s  2>&1 | FileCheck %s

! CHECK: AllSources:
! CHECK-NEXT: AllSources range_ [{{[0-9]*}}..{{[0-9]*}}] ({{[0-9]*}} bytes)
! CHECK-NEXT:    [1..1] (1 bytes) -> compiler '?'(0x3f)
! CHECK-NEXT:    [2..2] (1 bytes) -> compiler ' '(0x20)
! CHECK-NEXT:    [3..3] (1 bytes) -> compiler '\'(0x5c)
! CHECK-NEXT:    [{{[0-9]*}}..{{[0-9]*}}] ({{[0-9]*}} bytes) -> file {{.*[/\\]}}debug-provenance.f90
! CHECK-NEXT:    [{{[0-9]*}}..{{[0-9]*}}] ({{[0-9]*}} bytes) -> compiler '(after end of source)'
! CHECK-NEXT: SourceFile '{{.*[/\\]}}debug-provenance.f90'
! CHECK-NEXT:   origin_[1] -> '{{.*[/\\]}}debug-provenance.f90' 1
! CHECK-NEXT: CookedSource::provenanceMap_:
! CHECK-NEXT: offsets [{{[0-9]*}}..{{[0-9]*}}] -> provenances [{{[0-9]*}}..{{[0-9]*}}] ({{[0-9]*}} bytes)
! CHECK-NEXT: CookedSource::invertedMap_:
! CHECK-NEXT: provenances [{{[0-9]*}}..{{[0-9]*}}] ({{[0-9]*}} bytes) -> offsets [{{[0-9]*}}..{{[0-9]*}}]
! CHECK-EMPTY:

program A
end
