! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Loop Construct

program omp
  integer i, j, k

  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do  collapse(3)
  do i = 0, 10
    if (i .lt. 1) then
      !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
      cycle
    end if
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
      if (i .lt. 1) then
        !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
        cycle
      end if
      do k  = 0, 10
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do

  !!ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do  collapse(2)
  foo: do i = 0, 10
    foo1: do j = 0, 10
      if (i .lt. 1) then
        !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
        cycle foo
      else if (i .gt. 3) then
        cycle foo1
      end if
      do k  = 0, 10
        print *, i, j, k
      end do
    end do foo1
  end do foo
  !$omp end do


  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do  collapse(3)
  foo: do i = 0, 10
    foo1: do j = 0, 10
      if (i .lt. 1) then
        !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
        cycle foo
      else if (i .gt. 3) then
        !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
        cycle foo1
      end if
         foo2:  do k  = 0, 10
             print *, i, j, k
           end do foo2
         end do foo1
  end do foo
  !$omp end do

  !$omp do  ordered(3)
  foo: do i = 0, 10
    foo1: do j = 0, 10
         foo2:  do k  = 0, 10
           if (i .lt. 1) then
             !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
             cycle foo
           else if (i .gt. 3) then
             !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
             cycle foo1
          else
             !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
             cycle foo
          end if
             print *, i, j, k
           end do foo2
         end do foo1
  end do foo
  !$omp end do

end program omp
