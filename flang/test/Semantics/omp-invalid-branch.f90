! RUN: not %flang -fsyntax-only -fopenmp %s 2>&1 | FileCheck %s
! OpenMP Version 4.5
! Check invalid branches into or out of OpenMP structured blocks.

subroutine omp_err_end_eor(a, b, x)
  integer x

  !$omp parallel
  !CHECK: invalid branch into an OpenMP structured block
  !CHECK: In the enclosing PARALLEL directive branched into

  !CHECK: invalid branch leaving an OpenMP structured block
  !CHECK: Outside the enclosing PARALLEL directive
  open (10, file="myfile.dat", err=100)
  !CHECK: invalid branch leaving an OpenMP structured block
  !CHECK: Outside the enclosing PARALLEL directive

  !CHECK: invalid branch into an OpenMP structured block
  !CHECK: In the enclosing PARALLEL directive branched into

  !CHECK: invalid branch leaving an OpenMP structured block
  !CHECK: Outside the enclosing PARALLEL directive
  read (10, 20, end=200, size=x, advance='no', eor=300) a
  !$omp end parallel

  goto 99
  99 close (10)
  goto 40
  !$omp parallel
  100 print *, "error opening"
  !$omp end parallel
  101 return
  200 print *, "end of file"
  202 return

  !$omp parallel
  300 print *, "end of record"
  !$omp end parallel

  303 return
  20 format (1x,F5.1)
  30 format (2x,F6.2)
  !CHECK: invalid branch into an OpenMP structured block
  !CHECK: In the enclosing PARALLEL directive branched into
  40 open (11, file="myfile2.dat", err=100)
  goto 50
  !CHECK: invalid branch into an OpenMP structured block
  !CHECK: In the enclosing PARALLEL directive branched into
  50 write (11, 30, err=100) b
  close (11)
end subroutine

subroutine omp_alt_return_spec(n, *, *)
  if (n .eq. 0) return
  if (n .eq. 1) return 1
  return 2
end subroutine

program omp_invalid_branch
  integer :: n = 0, a = 3, b

  !CHECK: invalid branch into an OpenMP structured block
  !CHECK: In the enclosing PARALLEL directive branched into

  !CHECK: invalid branch into an OpenMP structured block
  !CHECK: In the enclosing PARALLEL directive branched into
  goto (1, 2, 3) a

  assign 2 to b
  !CHECK: invalid branch into an OpenMP structured block
  !CHECK: In the enclosing PARALLEL directive branched into
  goto b (1, 2)

  !$omp parallel
  !CHECK: invalid branch into an OpenMP structured block
  !CHECK: In the enclosing SINGLE directive branched into

  !CHECK: invalid branch leaving an OpenMP structured block
  !CHECK: Outside the enclosing PARALLEL directive
  3 if(n) 4, 5, 6

  6 print *, 6
  2 print *, 2

  !$omp single
  4 print *, 4
  !$omp end single
  !$omp end parallel

  1 print *, 1
  5 print *, 5

  !$omp parallel
  !CHECK: invalid branch into an OpenMP structured block
  !CHECK: In the enclosing SINGLE directive branched into

  !CHECK: invalid branch leaving an OpenMP structured block
  !CHECK: Outside the enclosing PARALLEL directive
  call omp_alt_return_spec(n, *8, *9)
  print *, "Normal Return"
  !$omp single
  8 print *, "1st alternate return"
  !$omp end single
  !$omp end parallel
  9 print *, "2nd alternate return"

end program
