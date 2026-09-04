! RUN: bbc -emit-hlfir --wrap-unstructured-constructs-in-execute-region %s -o - | FileCheck %s

! An IF/ELSE whose THEN branch nests another IF, and whose ELSE branch
! ends in STOP.  Used to trip the MLIR verifier ("operation with block
! successors must terminate its parent block") because the inner IF's
! own scf.execute_region wrap was created at the outer IF's terminated
! entry block (right after the outer's cf.cond_br).  Fixed by starting
! the entry block that the outer's createEmptyBlocks allocated for the
! inner (wrappable) IfConstruct before recursing.
subroutine wrapped_if_nested_with_stop(a, b)
  integer :: a, b
  if (a == 1) then
     if (b == 2) then
        stop 1
     end if
  else
     stop 1
  end if
end subroutine

! CHECK-LABEL: func.func @_QPwrapped_if_nested_with_stop
! CHECK:         scf.execute_region
! CHECK:           cf.cond_br
! CHECK:           scf.execute_region
! CHECK:             cf.cond_br
! CHECK:             fir.call @_FortranAStopStatement
! CHECK:             fir.unreachable
! CHECK:             scf.yield
! CHECK:           }
! CHECK:           fir.call @_FortranAStopStatement
! CHECK:           fir.unreachable
! CHECK:           scf.yield
! CHECK:         }

! Subroutine with an alternate ENTRY point followed by a structured outer
! IF that nests a wrappable inner IF.  The shared PFT is lowered once per
! entry, and the inner IfConstruct's firstStmt.block field still points
! at the previous entry's wrap-region block on the second pass.  Reading
! that stale pointer used to move the builder into the previous entry's
! function, so both wraps ended up in _QPfoo and _QPbar was left empty
! with a load referencing an SSA value from the other function.  Fixed by
! only starting firstStmt.block when it belongs to the current builder's
! region.
subroutine foo_with_entry(a)
  integer a
entry bar_with_entry(a)
  if (a .eq. 1) then
    if (a .ne. 3) stop
  end if
end subroutine

! CHECK-LABEL: func.func @_QPfoo_with_entry
! CHECK:         %[[FOO_A:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFfoo_with_entryEa"
! CHECK:         fir.if
! CHECK:           scf.execute_region
! CHECK:             fir.load %[[FOO_A]]#0
! CHECK:             cf.cond_br
! CHECK:             fir.call @_FortranAStopStatement
! CHECK:             scf.yield
! CHECK:           }

! CHECK-LABEL: func.func @_QPbar_with_entry
! CHECK:         %[[BAR_A:.*]]:2 = hlfir.declare {{.*}}uniq_name = "_QFfoo_with_entryEa"
! CHECK:         fir.if
! CHECK:           scf.execute_region
! CHECK:             fir.load %[[BAR_A]]#0
! CHECK:             cf.cond_br
! CHECK:             fir.call @_FortranAStopStatement
! CHECK:             scf.yield
! CHECK:           }
