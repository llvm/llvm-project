! RUN: %flang_fc1 -fopenacc -fdebug-dump-symbols %s 2>&1 | FileCheck %s --implicit-check-not=openACCRoutineInfos

! When the name on an !$acc routine directive in an interface block matches no
! interface body, flang resolves it the same lenient way it does in every other
! position: the name is bound to an (implicit) external procedure rather than
! diagnosed, the directive is accepted without a crash, and the interface body
! receives no ROUTINE information.  Both placements of the directive are covered:
!   - 'm_in_block': the directive sits directly in the interface block.
!   - 'm_in_body': the directive sits inside the subroutine interface body.

module m_in_block
  implicit none
  interface
  !$acc routine (no_block) seq
  subroutine sub_block() bind(c, name="sub_block")
  end subroutine sub_block
  end interface
end module m_in_block

module m_in_body
  implicit none
  interface
  subroutine sub_body() bind(c, name="sub_body")
  !$acc routine (no_body) seq
  end subroutine sub_body
  end interface
end module m_in_body

! Each mismatched name resolves to an implicit external procedure ...
! CHECK-DAG: no_block: ProcEntity
! CHECK-DAG: no_body: ProcEntity
! ... and neither interface body receives the ROUTINE information (asserted by
! --implicit-check-not=openACCRoutineInfos on the RUN line).
! CHECK-DAG: sub_block, BIND(C), EXTERNAL, PUBLIC (Subroutine): Subprogram isInterface
! CHECK-DAG: sub_body, BIND(C), EXTERNAL, PUBLIC (Subroutine): Subprogram isInterface
