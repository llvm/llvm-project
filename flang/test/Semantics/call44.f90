! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Wno-portability -Werror
subroutine assumedshape(normal, contig)
  real normal(:)
  real, contiguous :: contig(:)
  !WARNING: If the procedure's interface were explicit, this reference would be in error [-Wknown-bad-implicit-interface]
  !BECAUSE: Element of assumed-shape array may not be associated with a dummy argument 'assumedsize=' array
  call seqAssociate(normal(1))
  !PORTABILITY: Element of contiguous assumed-shape array is accepted for storage sequence association [-Wcontiguous-ok-for-seq-association]
  call seqAssociate(contig(1))
end
subroutine seqAssociate(assumedSize)
  real assumedSize(*)
end
