! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests actual/dummy pointer argument shape mismatches
module m
 contains
  subroutine s0(p)
    real, pointer, intent(in) :: p
  end
  subroutine s1(p)
    real, pointer, intent(in) :: p(:)
  end
  subroutine sa(p)
    real, pointer, intent(in) :: p(..)
  end
  subroutine sao(p)
    real, intent(in), optional, pointer :: p(..)
  end
  subroutine so(x)
    real, intent(in), optional :: x(..)
  end
  subroutine soa(a)
    real, intent(in), optional, allocatable :: a(..)
  end
  subroutine test
    real, pointer :: a0, a1(:)
    call s0(null(a0)) ! ok
    !ERROR: Rank of dummy argument is 0, but actual argument has rank 1
    !ERROR: Rank of pointer is 0, but function result has rank 1
    call s0(null(a1))
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 0
    !ERROR: Rank of pointer is 1, but function result has rank 0
    call s1(null(a0))
    call s1(null(a1)) ! ok
    call sa(null(a0)) ! ok
    call sa(null(a1)) ! ok
    !ERROR: NULL() without MOLD= must not be associated with an assumed-rank dummy argument that is ALLOCATABLE, POINTER, or non-OPTIONAL
    call sa(null())
    call sao ! ok
    !ERROR: NULL() without MOLD= must not be associated with an assumed-rank dummy argument that is ALLOCATABLE, POINTER, or non-OPTIONAL
    call sao(null())
    call so ! ok
    call so(null()) ! ok
    call soa ! ok
    !ERROR: NULL() without MOLD= must not be associated with an assumed-rank dummy argument that is ALLOCATABLE, POINTER, or non-OPTIONAL
    call soa(null())
  end
end
