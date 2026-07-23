! RUN: bbc -emit-hlfir -fopenmp --wrap-unstructured-constructs-in-execute-region %s -o - | FileCheck %s

! An unstructured SELECT CASE inside an OpenMP DO body.  With the wrap
! flag on, the enclosing DoConstruct is wrappable, so its isUnstructured
! no longer propagates to the OpenMPConstruct — the top-level
! createEmptyBlocks does not recurse into the DO body, and the CaseStmt /
! constructExit blocks the SelectCase lowering wants would be left
! unallocated.  OpenMP's own createEmptyRegionBlocks now creates the
! missing blocks in the loop-nest region so genFIR(SelectCaseStmt)
! finds targets for its fir.select_case.
!
! The outer IF is wrappable, so the whole subroutine body sits inside an
! scf.execute_region — this exercises the interaction between the wrap
! machinery (which does allocate its own nested blocks) and the OMP loop
! lowering (which now allocates missing body blocks itself).
subroutine s(a, k)
  integer a, k
  if (a > 0) then
!$omp do
    do k = 1, 2
      select case (k)
        case (1)
          a = 10
        case (2)
          a = 20
        case default
          a = 30
      end select
    end do
    stop
  end if
end subroutine

! CHECK-LABEL: func.func @_QPs
! CHECK:         scf.execute_region no_inline {
! CHECK:           cf.cond_br
! CHECK:           omp.wsloop
! CHECK:             omp.loop_nest
! CHECK:               fir.load
! CHECK:               fir.select_case %{{[0-9]+}} : i32 [#fir.point, %{{.*}}, ^[[C1:bb[0-9]+]], #fir.point, %{{.*}}, ^[[C2:bb[0-9]+]], unit, ^[[CDFLT:bb[0-9]+]]]
! CHECK:             ^[[C1]]:
! CHECK:               hlfir.assign
! CHECK:             ^[[C2]]:
! CHECK:               hlfir.assign
! CHECK:             ^[[CDFLT]]:
! CHECK:               hlfir.assign
! CHECK:           fir.call @_FortranAStopStatement
! CHECK:           scf.yield
! CHECK:         }
