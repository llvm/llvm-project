! RUN: %python %S/test_errors.py %s %flang_fc1
! The RHS of a pointer assignment can be unlimited polymorphic
! if the LHS is a sequence type.
program main
  type nonSeqType
    integer j
  end type
  type seqType
    sequence
    integer j
  end type
  type(nonSeqType), target :: xNonSeq = nonSeqType(1)
  type(nonSeqType), pointer :: pNonSeq
  type(seqType), target :: xSeq = seqType(1), aSeq(1)
  type(seqType), pointer :: pSeq, paSeq(:)
  !ERROR: function result type 'CLASS(*)' is not compatible with pointer type 'nonseqtype'
  pNonSeq => polyPtr(xNonSeq)
  pSeq => polyPtr(xSeq) ! ok
  !ERROR: Pointer has rank 1 but target has rank 0
  paSeq => polyPtr(xSeq)
  !ERROR: Pointer has rank 0 but target has rank 1
  pSeq => polyPtrArr(aSeq)
 contains
  function polyPtr(target)
    class(*), intent(in), target :: target
    class(*), pointer :: polyPtr
    polyPtr => target
  end
  function polyPtrArr(target)
    class(*), intent(in), target :: target(:)
    class(*), pointer :: polyPtrArr(:)
    polyPtrArr => target
  end
  function err1(target)
    class(*), intent(in), target :: target(:)
    class(*), pointer :: err1
    !ERROR: Pointer has rank 0 but target has rank 1
    err1 => target
  end
  function err2(target)
    class(*), intent(in), target :: target
    class(*), pointer :: err2(:)
    !ERROR: Pointer has rank 1 but target has rank 0
    err2 => target
  end
end
