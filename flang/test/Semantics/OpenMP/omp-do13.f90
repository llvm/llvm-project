! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Loop Construct

program omp
  integer i, j, k

  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do  collapse(3)
  do i = 0, 10
    !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
    cycle
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
      !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
      cycle
      do k  = 0, 10
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do

  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do  collapse(2)
  do i = 0, 10
    !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
    cycle
    do j = 0, 10
      do k  = 0, 10
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do


  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do  collapse(2)
  foo: do i = 0, 10
    !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
    cycle foo
    do j = 0, 10
      do k  = 0, 10
        print *, i, j, k
      end do
    end do
  end do foo
  !$omp end do


  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp do collapse(3)
  do 60 i=1,10
    do j=1,10
      !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
      cycle
      do k=1,10
        print *,i
      end do
    end do
  60 continue
  !$omp end do

  !$omp do  collapse(3)
  foo: do i = 0, 10
    foo1: do j = 0, 10
         foo2:  do k  = 0, 10
             !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
             cycle foo
             print *, i, j, k
           end do foo2
         end do foo1
  end do foo
  !$omp end do

  !$omp do  collapse(3)
  foo: do i = 0, 10
    foo1: do j = 0, 10
         foo2:  do k  = 0, 10
             !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
              cycle foo1
             print *, i, j, k
           end do foo2
         end do foo1
  end do foo
  !$omp end do


  !$omp do  collapse(2)
  foo: do i = 0, 10
    foo1: do j = 0, 10
         foo2:  do k  = 0, 10
             !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
             cycle foo
             print *, i, j, k
           end do foo2
         end do foo1
  end do foo
  !$omp end do


  !$omp do  collapse(2)
  foo: do i = 0, 10
    foo1: do j = 0, 10
             !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
             cycle foo
         foo2:  do k  = 0, 10
             print *, i, j, k
           end do foo2
         end do foo1
  end do foo
  !$omp end do

  !$omp do  ordered(2)
  foo: do i = 0, 10
    foo1: do j = 0, 10
      !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
      cycle foo
      !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
      !$omp do  collapse(1)
      foo2:  do k  = 0, 10
        print *, i, j, k
        end do foo2
     !$omp end do
     end do foo1
  end do foo
  !$omp end do

  !$omp parallel
  !$omp do collapse(2)
  foo: do i = 0, 10
    foo1: do j = 0, 10
      !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
      cycle foo
      !$omp parallel
      !$omp do collapse(2)
      foo2:  do k  = 0, 10
        foo3:  do l  = 0, 10
          print *, i, j, k, l
          !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
          cycle foo2
        end do foo3
      end do foo2
      !$omp end do
      !$omp end parallel
    end do foo1
  end do foo
  !$omp end do
  !$omp end parallel

  !$omp parallel
  !ERROR: The value of the parameter in the COLLAPSE or ORDERED clause must not be larger than the number of nested loops following the construct.
  !$omp parallel do ordered(3) collapse(2)
  foo: do i = 0, 10
    foo1: do j = 0, 10
      !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
      cycle foo
      !$omp parallel
      !$omp parallel do collapse(2)
      foo2:  do k  = 0, 10
        foo3:  do l  = 0, 10
          print *, i, j, k, l
          !ERROR: CYCLE statement to non-innermost associated loop of an OpenMP DO construct
          cycle foo2
        end do foo3
      end do foo2
      !$omp end parallel do
      !$omp end parallel
    end do foo1
  end do foo
!$omp end parallel do
!$omp end parallel

end program omp
