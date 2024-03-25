! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.15.3.4 firstprivate Clause
! Variables that appear in a firstprivate clause on a distribute or
! worksharing constructs must not appear in the private or
! reduction clause in a teams or parallel constructs in the outer context
!
! A list item may appear in a firstprivate or lastprivate clause but not both on
! a distribute directive

program omp_firstprivate
  integer :: i, a(10), b(10), c(10)

  a = 10
  b = 20

  !ERROR: TARGET construct with nested TEAMS region contains statements or directives outside of the TEAMS construct
  !$omp target
  !$omp teams private(a, b)
  !ERROR: FIRSTPRIVATE variable 'a' is PRIVATE in outer context
  !$omp distribute firstprivate(a)
  do i = 1, 10
    a(i) = a(i) + b(i) - i
  end do
  !$omp end distribute
  !$omp end teams
  !$omp teams reduction(+:a)
  !ERROR: FIRSTPRIVATE variable 'a' is PRIVATE in outer context
  !$omp distribute firstprivate(a)
  do i = 1, 10
   b(i) = b(i) + a(i) + i
  end do
  !$omp end distribute
  !$omp end teams

  !$omp teams distribute firstprivate(a) lastprivate(b)
  do i = 1, 10
    a(i) = a(i) + b(i) - i
  end do
  !$omp end teams distribute
  !ERROR: Variable 'b' may not appear on both FIRSTPRIVATE and LASTPRIVATE clauses on a TEAMS DISTRIBUTE construct
  !$omp teams distribute firstprivate(a,b) lastprivate(b)
  do i = 1, 10
    a(i) = a(i) + b(i) - i
  end do
  !$omp end teams distribute
  !ERROR: Variable 'a' may not appear on both FIRSTPRIVATE and LASTPRIVATE clauses on a TEAMS DISTRIBUTE construct
  !ERROR: Variable 'b' may not appear on both FIRSTPRIVATE and LASTPRIVATE clauses on a TEAMS DISTRIBUTE construct
  !$omp teams distribute firstprivate(a,b) lastprivate(a,b)
  do i = 1, 10
    a(i) = a(i) + b(i) - i
  end do
  !$omp end teams distribute
  !ERROR: Variable 'b' may not appear on both FIRSTPRIVATE and LASTPRIVATE clauses on a TEAMS DISTRIBUTE construct
  !$omp teams distribute lastprivate(a,b) firstprivate(b)
  do i = 1, 10
    a(i) = a(i) + b(i) - i
  end do
  !$omp end teams distribute
  !ERROR: Variable 'b' may not appear on both FIRSTPRIVATE and LASTPRIVATE clauses on a TEAMS DISTRIBUTE construct
  !ERROR: Variable 'a' may not appear on both FIRSTPRIVATE and LASTPRIVATE clauses on a TEAMS DISTRIBUTE construct
  !$omp teams distribute lastprivate(a,b) firstprivate(b,a)
  do i = 1, 10
    a(i) = a(i) + b(i) - i
  end do
  !$omp end teams distribute
  !$omp end target

  print *, a, b

  !$omp parallel private(a,b)
  !ERROR: FIRSTPRIVATE variable 'b' is PRIVATE in outer context
  !$omp do firstprivate(b)
  do i = 1, 10
    c(i) = a(i) + b(i) + i
  end do
  !$omp end do
  !$omp end parallel

  !$omp parallel reduction(*:a)
  !ERROR: FIRSTPRIVATE variable 'a' is PRIVATE in outer context
  !$omp do firstprivate(a,b)
  do i = 1, 10
    c(i) =  c(i) * a(i) * b(i) * i
  end do
  !$omp end do
  !$omp end parallel

  !$omp parallel reduction(+:a)
  !ERROR: FIRSTPRIVATE variable 'a' is PRIVATE in outer context
  !$omp sections firstprivate(a, b)
  !$omp section
  c = c * a + b
  !$omp end sections
  !$omp end parallel

  !$omp parallel reduction(*:a)
  !ERROR: FIRSTPRIVATE variable 'a' is PRIVATE in outer context
  !$omp task firstprivate(a,b)
  c =  c * a * b
  !$omp end task
  !$omp end parallel

  !$omp parallel reduction(+:b)
  !ERROR: FIRSTPRIVATE variable 'b' is PRIVATE in outer context
  !$omp taskloop firstprivate(b)
  do i = 1, 10
    c(i) = a(i) + b(i) + i
    a = a+i
    b = b-i
  end do
  !$omp end taskloop
  !$omp end parallel

  !$omp parallel firstprivate(a)
  !ERROR: FIRSTPRIVATE variable 'a' is PRIVATE in outer context
  !$omp single firstprivate(a)
  print *, a
  !$omp end single
  !$omp end parallel

  print *, c

end program omp_firstprivate
