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
