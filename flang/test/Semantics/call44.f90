! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
subroutine assumedshape(normal, contig)
  real normal(:)
  real, contiguous :: contig(:)
  !WARNING: If the procedure's interface were explicit, this reference would be in error
  !BECAUSE: Element of assumed-shape array may not be associated with a dummy argument 'assumedsize=' array
  call seqAssociate(normal(1))
  !PORTABILITY: Element of contiguous assumed-shape array is accepted for storage sequence association
  call seqAssociate(contig(1))
end
subroutine seqAssociate(assumedSize)
  real assumedSize(*)
end
