!UNSUPPORTED: target={{.*}}
!RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=60 %s | FileCheck %s

module m
integer :: x
!$omp declare_target link(x) device_type(nohost)
real :: w(10), u(10)
common /named_block/ w, u
!$omp declare_target link(/named_block/)

interface
  real function g(v)
    real :: v(10)
    !$omp declare_target
  end
end interface

contains
subroutine f
  !$omp declare_target(f)
end
subroutine h
  integer, save :: a(10)
  !$omp declare_target enter(h, a)
  continue
end
end module

!CHECK: Module scope: m size=4 alignment=4 sourceRange=383 bytes
!CHECK:     f, PUBLIC (Subroutine): Subprogram () OmpDeclareTargetFlags:(enter)
!CHECK:     g, EXTERNAL, PUBLIC (Function, OmpDeclareTarget): Subprogram isInterface result:REAL(4) g (REAL(4) v) OmpDeclareTargetFlags:(enter)
!CHECK:     h, PUBLIC (Subroutine): Subprogram () OmpDeclareTargetFlags:(enter)
!CHECK:     u, PUBLIC (InCommonBlock, OmpDeclareTarget) size=40 offset=40: ObjectEntity type: REAL(4) shape: 1_8:10_8
!CHECK:     w, PUBLIC (InCommonBlock, OmpDeclareTarget) size=40 offset=0: ObjectEntity type: REAL(4) shape: 1_8:10_8
!CHECK:     x, PUBLIC (OmpDeclareTarget) size=4 offset=0: ObjectEntity type: INTEGER(4) OmpDeclareTargetFlags:(device_type(nohost) link)
!CHECK:     named_block size=80 offset=0: CommonBlockDetails alignment=4: w u OmpDeclareTargetFlags:(link)
!CHECK:     Subprogram scope: g size=44 alignment=4 sourceRange=57 bytes
!CHECK:       g size=4 offset=0: ObjectEntity funcResult type: REAL(4)
!CHECK:       v size=40 offset=4: ObjectEntity dummy type: REAL(4) shape: 1_8:10_8
!CHECK:     Subprogram scope: f size=0 alignment=1 sourceRange=40 bytes
!CHECK:       f (Subroutine, OmpDeclareTarget): HostAssoc => f, PUBLIC (Subroutine): Subprogram () OmpDeclareTargetFlags:(enter)
!CHECK:     Subprogram scope: h size=40 alignment=4 sourceRange=81 bytes
!CHECK:       a, SAVE (OmpDeclareTarget) size=40 offset=0: ObjectEntity type: INTEGER(4) shape: 1_8:10_8 OmpDeclareTargetFlags:(enter)
!CHECK:       h (Subroutine, OmpDeclareTarget): HostAssoc => h, PUBLIC (Subroutine): Subprogram () OmpDeclareTargetFlags:(enter)
