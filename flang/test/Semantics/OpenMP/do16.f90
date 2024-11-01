! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Loop Construct

program omp
  integer i, j, k

  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do  collapse(3)
  do i = 0, 10
    select case (i)
    case(1)
      !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
      cycle
    end select
    do j = 0, 10
      do k  = 0, 10
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do

  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do  collapse(3)
  do i = 0, 10
    do j = 0, 10
      select case (i)
      case(1)
        !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
        cycle
      end select
      do k  = 0, 10
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do

  !$omp do  collapse(2)
  foo: do i = 0, 10
    foo1: do j = 0, 10
      select case (i)
      case(1)
        !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
        cycle foo
      case(5)
        cycle foo1
      end select
      do k  = 0, 10
        print *, i, j, k
      end do
    end do foo1
  end do foo
  !$omp end do

  !$omp do  ordered(3)
  foo: do i = 0, 10
    foo1: do j = 0, 10
      foo2: do k  = 0, 10
        select case (i)
        case(1)
          !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
          cycle foo
        case(5)
          !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
          cycle foo1
        case(7)
          cycle foo2
        end select
        print *, i, j, k
      end do foo2
    end do foo1
  end do foo
  !$omp end do

end program omp
