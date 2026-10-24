! RUN: %python %S/test_errors.py %s %flang_fc1

subroutine pdt
  type t(k)
    integer, kind :: k
    !ERROR: KIND parameter value (5) of intrinsic type LOGICAL did not resolve to a supported value
    !ERROR: unsupported LOGICAL(KIND=5)
    logical(k) :: x = .true._k
  end type
  type(t(4)) :: ok
  type(t(5)) :: v
end
