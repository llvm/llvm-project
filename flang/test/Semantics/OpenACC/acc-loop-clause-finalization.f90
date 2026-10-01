! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Standalone LOOP clauses must be finalized after the complete header and
! before visiting the associated DO construct.
subroutine loop_clause_finalization(a, i, j)
  integer :: a(10), i, j, k, x
  !$acc parallel

  !WARNING: 'x' appears more than once in the same kind of data-sharing clause on an OpenACC directive; duplicate ignored [-Wopenacc-usage]
  !$acc loop private(x, x)
  do k = 1, 10
    x = k
  end do

  !ERROR: 'x' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc loop private(x) reduction(+:x)
  do k = 1, 10
    x = x + k
  end do

  !ERROR: 'x' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc loop reduction(+:x) private(x)
  do k = 1, 10
    x = x + k
  end do

  !WARNING: 'a(i)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
  !WARNING: 'a(j)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
  !$acc loop private(a(i), a(j)) private(a)
  do k = 1, 10
    a(k) = k
  end do

  !WARNING: 'a(i)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
  !WARNING: 'a(j)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
  !$acc loop private(a(i)) private(a) private(a(j))
  do k = 1, 10
    a(k) = k
  end do

  !WARNING: 'a(i)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
  !WARNING: 'a(j)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
  !$acc loop private(a) private(a(i), a(j))
  do k = 1, 10
    a(k) = k
  end do

  ! Explicit clauses are finalized before implicit loop-index privatization.
  !$acc loop reduction(+:k)
  !ERROR: 'k' appears in more than one data-sharing clause on the same OpenACC directive
  do k = 1, 10
    x = k
  end do

  !$acc end parallel
end subroutine
