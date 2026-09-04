!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

! A loop-sequence-associated METADIRECTIVE variant (FUSE) with no loop sequence
! to associate with is diagnosed like the loop-nest case, but names a sequence.
subroutine no_sequence_at_end()
  !ERROR: This construct should contain a DO-loop or a loop-sequence-generating construct
  !$omp metadirective when(implementation={vendor(llvm)}: fuse) otherwise(nothing)
end subroutine

! A variant that cannot be selected on this target needs no loop sequence.
subroutine no_sequence_dead_variant()
  !$omp metadirective when(device={kind(nohost)}: fuse) otherwise(nothing)
end subroutine
