! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! Test that the members of a COMMON block are laid out sequentially with no
! padding between them, as ifx and XLF do by default. This includes COMMON
! blocks that contain zero-size members, which must not cause the following
! member to be aligned either.

! The size of each block is the sum of its member sizes. The alignment of the
! block is still the largest alignment of its members.
! CHECK-DAG: fir.global common @blk05_(dense<0> : vector<9xi8>) <{alignment = 8 : i64}> : !fir.array<9xi8>
! CHECK-DAG: fir.global common @blk17_(dense<0> : vector<9xi8>) <{alignment = 8 : i64}> : !fir.array<9xi8>
! CHECK-DAG: fir.global common @blk20_(dense<0> : vector<2xi8>) <{alignment = 1 : i64}> : !fir.array<2xi8>

! p05: zero-size array between a CHARACTER(1) and a REAL(8).
! Block /blk05/ layout:
!   offset 0 : c1 (1 byte)
!   offset 1 : r0 (0 bytes)
!   offset 1 : r1 (8 bytes, not aligned to 8)
subroutine p05
  character(1) :: c1
  real(8) :: r0(0)
  real(8) :: r1
  common /blk05/ c1, r0, r1
  call use_p05(c1, r0, r1)
end subroutine

! CHECK-LABEL: func.func @_QPp05
! CHECK: hlfir.declare {{.*}} storage({{.*}}[0]) uniq_name("_QFp05Ec1")
! CHECK: hlfir.declare {{.*}} storage({{.*}}[1]) uniq_name("_QFp05Er0")
! CHECK: hlfir.declare {{.*}} storage({{.*}}[1]) uniq_name("_QFp05Er1")

! p20: zero-size REAL(8) array between two CHARACTER(1).
! Block /blk20/ layout:
!   offset 0 : c1
!   offset 1 : z0 (0 bytes)
!   offset 1 : c2
subroutine p20
  character(1) :: c1
  real(8) :: z0(0)
  character(1) :: c2
  common /blk20/ c1, z0, c2
  call use_p20(c1, z0, c2)
end subroutine

! CHECK-LABEL: func.func @_QPp20
! CHECK: hlfir.declare {{.*}} typeparams {{.*}}storage({{.*}}[0]) uniq_name("_QFp20Ec1")
! CHECK: hlfir.declare {{.*}} typeparams {{.*}}storage({{.*}}[1]) uniq_name("_QFp20Ec2")
! CHECK: hlfir.declare {{.*}} storage({{.*}}[1]) uniq_name("_QFp20Ez0")

! p17: as p05, with d8 storage associated with the zero-size r0.
! Block /blk17/ layout:
!   offset 0 : c1
!   offset 1 : r0 (0 bytes), d8 (8 bytes, by EQUIVALENCE)
!   offset 1 : r1 (8 bytes)
subroutine p17
  character(1) :: c1
  real(8) :: r0(0)
  real(8) :: r1
  real(8) :: d8
  common /blk17/ c1, r0, r1
  equivalence (d8, r0)
  call use_p17(c1, r0, r1, d8)
end subroutine

! CHECK-LABEL: func.func @_QPp17
! CHECK: hlfir.declare {{.*}} typeparams {{.*}}storage({{.*}}[0]) uniq_name("_QFp17Ec1")
! CHECK: hlfir.declare {{.*}} storage({{.*}}[1]) uniq_name("_QFp17Ed8")
! CHECK: hlfir.declare {{.*}} storage({{.*}}[1]) uniq_name("_QFp17Er0")
! CHECK: hlfir.declare {{.*}} storage({{.*}}[1]) uniq_name("_QFp17Er1")
