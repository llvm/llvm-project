! RUN: %flang -c %s -fsyntax-only
module contig
  implicit none
  integer, dimension(:), pointer, contiguous, save :: ptr
end module
