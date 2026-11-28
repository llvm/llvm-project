! RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s

! COMMON block members are placed sequentially with no padding between them,
! as ifx and XLF do by default.  The alignment of a COMMON block is still the
! largest alignment of its members.  Zero-size members have no size or offset
! in the dump, but must not cause the next member to be aligned.

! CHECK-LABEL: Subprogram scope: p05
! CHECK-DAG: blk05 size=9 offset=0: CommonBlockDetails alignment=8:
! CHECK-DAG: c1 (InCommonBlock) size=1 offset=0:
! CHECK-DAG: r1 (InCommonBlock) size=8 offset=1:
subroutine p05
  character(1) :: c1
  real(8) :: r0(0)
  real(8) :: r1
  common /blk05/ c1, r0, r1
end subroutine

! CHECK-LABEL: Subprogram scope: p20
! CHECK-DAG: blk20 size=2 offset=0: CommonBlockDetails alignment=1:
! CHECK-DAG: c1 (InCommonBlock) size=1 offset=0:
! CHECK-DAG: c2 (InCommonBlock) size=1 offset=1:
subroutine p20
  character(1) :: c1
  real(8) :: z0(0)
  character(1) :: c2
  common /blk20/ c1, z0, c2
end subroutine

! CHECK-LABEL: Subprogram scope: p17
! CHECK-DAG: blk17 size=9 offset=0: CommonBlockDetails alignment=8:
! CHECK-DAG: c1 (InCommonBlock) size=1 offset=0:
! CHECK-DAG: r1 (InCommonBlock) size=8 offset=1:
! CHECK-DAG: d8 size=8 offset=1:
subroutine p17
  character(1) :: c1
  real(8) :: r0(0)
  real(8) :: r1
  real(8) :: d8
  common /blk17/ c1, r0, r1
  equivalence (d8, r0)
end subroutine
