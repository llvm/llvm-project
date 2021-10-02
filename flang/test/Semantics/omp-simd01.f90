! RUN: %python %S/test_errors.py %s %flang -fopenmp
! OpenMP Version 5.0
! 2.9.3.1 simd Construct
!   - A program that branches into or out of a simd region is non-conforming.
!   - The associated loops must be structured blocks

program omp_simd
   integer i, j

   !$omp simd
   do i = 1, 10
      do j = 1, 10
         print *, "omp simd"
         !ERROR: invalid branch leaving an OpenMP structured block
         goto 10
      end do
      if (i .EQ. 5) THEN
         call function1()
      else if (i .EQ. 7) THEN
         open (10, file="random-file-name.txt", err=20)
20       print *, "Error message doesn't branch out of the loop's structured block"
      else
         !ERROR: invalid branch leaving an OpenMP structured block
         open (10, file="random-file-name.txt", err=10)
      end if
   end do
   !$omp end simd
10 stop

end program omp_simd

subroutine function1()
   integer i, option
   option = 1
   !$omp simd
   do i = 1, 10
      print *, "CORRECT SIMD LOOP"
   end do
   !$omp end simd
end subroutine function1
