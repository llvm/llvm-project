! RUN: %python %S/test_errors.py %s %flang_fc1
! Interoperable objects that require descriptors cannot be CONTIGUOUS
subroutine interop(ptr,ashape,arank,eshape,asize) bind(c)
  !ERROR: An interoperable pointer must not be CONTIGUOUS
  real, pointer, contiguous :: ptr(:)
  real, contiguous :: ashape(:) ! ok
  real, contiguous :: arank(..) ! ok
  !ERROR: CONTIGUOUS entity 'eshape' must be an array pointer, assumed-shape, or assumed-rank
  real, contiguous :: eshape(10)
  !ERROR: CONTIGUOUS entity 'asize' must be an array pointer, assumed-shape, or assumed-rank
  real, contiguous :: asize(*)
end
