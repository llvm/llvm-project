! An implementation-defined extension sentinel (!$omx in fixed form, !$ompx in
! free form; OpenMP 5.2, section 3.1) that is followed by a directive which the
! implementation does not recognize is ignored with a warning rather than being
! reported as an error.  This keeps programs that use vendor extensions portable
! to implementations that do not support them.

! RUN: %flang_fc1 -fopenmp -fdebug-unparse %s 2>&1 | FileCheck %s

! Diagnostics are emitted before the unparsed program.  An unrecognized
! extension directive produces a warning, not an error.
! CHECK: warning: Unrecognized OpenMP extension directive was ignored
! CHECK-SAME: [-Wignored-directive]
! Regression: an unrecognized extension directive that appears before the first
! statement of a program unit must warn without crashing (the source location of
! the ignored directive lies outside the range of any statement scope).
! CHECK: warning: Unrecognized OpenMP extension directive was ignored

! The unrecognized extension directives are ignored but unparsed with the
! extension sentinel preserved.
! CHECK: SUBROUTINE ompx_unrecognized
! CHECK: !$OMPX SOME_VENDOR_DIRECTIVE
! CHECK: END SUBROUTINE
! CHECK: !$OMPX ANOTHER_VENDOR_DIRECTIVE
! CHECK: END PROGRAM

! No error diagnostics are produced for the unrecognized extension directives.
! CHECK-NOT: error:

subroutine ompx_unrecognized
  !$ompx some_vendor_directive
end subroutine

!$ompx another_vendor_directive
end
