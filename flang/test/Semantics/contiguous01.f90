! RUN: %python %S/test_errors.py %s %flang_fc1
module m0
  real, pointer, contiguous :: p1(:) ! ok
  real, pointer :: p2(:)
end
module m
  use m0
  !ERROR: Cannot change CONTIGUOUS attribute on use-associated 'p1'
  contiguous p1
  !ERROR: Cannot change CONTIGUOUS attribute on use-associated 'p2'
  contiguous p2
  !ERROR: CONTIGUOUS entity 'x' must be an array pointer, assumed-shape, or assumed-rank
  real, contiguous :: x
  !ERROR: CONTIGUOUS entity 'scalar' must be an array pointer, assumed-shape, or assumed-rank
  real, contiguous, pointer :: scalar
  !ERROR: CONTIGUOUS entity 'allocatable' must be an array pointer, assumed-shape, or assumed-rank
  real, contiguous, allocatable :: allocatable
 contains
  !ERROR: CONTIGUOUS entity 'func' must be an array pointer, assumed-shape, or assumed-rank
  function func(ashape,arank) result(r)
    real, contiguous :: ashape(:) ! ok
    real, contiguous :: arank(..) ! ok
    !ERROR: CONTIGUOUS entity 'r' must be an array pointer, assumed-shape, or assumed-rank
    real :: r(10)
    !ERROR: CONTIGUOUS entity 'r2' must be an array pointer, assumed-shape, or assumed-rank
    real :: r2(10)
    contiguous func
    contiguous r
    contiguous e
    contiguous r2
    !ERROR: CONTIGUOUS entity 'e' must be an array pointer, assumed-shape, or assumed-rank
    entry e() result(r2)
  end
  function fp()
    real, pointer, contiguous :: fp(:) ! ok
  end
end
