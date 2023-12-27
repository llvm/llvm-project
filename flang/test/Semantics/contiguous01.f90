! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
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
  !PORTABILITY: CONTIGUOUS entity 'x' should be an array pointer, assumed-shape, or assumed-rank
  real, contiguous :: x
  !PORTABILITY: CONTIGUOUS entity 'scalar' should be an array pointer, assumed-shape, or assumed-rank
  real, contiguous, pointer :: scalar
  !PORTABILITY: CONTIGUOUS entity 'allocatable' should be an array pointer, assumed-shape, or assumed-rank
  real, contiguous, allocatable :: allocatable
 contains
  !PORTABILITY: CONTIGUOUS entity 'func' should be an array pointer, assumed-shape, or assumed-rank
  function func(ashape,arank) result(r)
    real, contiguous :: ashape(:) ! ok
    real, contiguous :: arank(..) ! ok
    !PORTABILITY: CONTIGUOUS entity 'r' should be an array pointer, assumed-shape, or assumed-rank
    real :: r(10)
    !PORTABILITY: CONTIGUOUS entity 'r2' should be an array pointer, assumed-shape, or assumed-rank
    real :: r2(10)
    contiguous func
    contiguous r
    contiguous e
    contiguous r2
    !PORTABILITY: CONTIGUOUS entity 'e' should be an array pointer, assumed-shape, or assumed-rank
    entry e() result(r2)
  end
  function fp()
    real, pointer, contiguous :: fp(:) ! ok
  end
end
