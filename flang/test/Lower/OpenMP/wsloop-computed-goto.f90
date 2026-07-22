! RUN: bbc -emit-hlfir -fopenmp --wrap-unstructured-constructs-in-execute-region %s -o - | FileCheck %s

! An unstructured DO whose body has a computed GO TO targeting a label
! inside the loop.  With the wrap flag on, the inner DO is wrappable, so
! its isUnstructured no longer propagates to the enclosing
! OpenMPConstruct.  The top-level createEmptyBlocks therefore treats the
! OMP construct as structured and does not recurse into the DO body — the
! label 17 ContinueStmt block would be left unallocated.  OpenMP's own
! createEmptyRegionBlocks now creates the missing block in the loop-nest
! region so genMultiwayBranch's fir.select finds a target.
!
! The outer IF is wrappable, so the whole subroutine body sits inside an
! scf.execute_region — this exercises the interaction between the wrap
! machinery (which does allocate its own nested blocks) and the OMP loop
! lowering (which now allocates missing body blocks itself).
subroutine s(ii1, a)
  integer ii1, a
  if (a > 0) then
!$omp do
    do ii1 = 0, 1
      go to (17), ii1
17    continue
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
! CHECK:               fir.select %{{[0-9]+}} : i32 [1, ^[[TARGET:bb[0-9]+]], unit, ^[[TARGET]]]
! CHECK:             ^[[TARGET]]:
! CHECK:               omp.yield
! CHECK:           fir.call @_FortranAStopStatement
! CHECK:           scf.yield
! CHECK:         }
