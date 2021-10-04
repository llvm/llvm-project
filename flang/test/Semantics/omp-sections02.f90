! RUN: %python %S/test_errors.py %s %flang -fopenmp
! OpenMP version 5.0.0
! 2.8.1 sections construct
! The code enclosed in a sections construct must be a structured block.
program OmpConstructSections01
   use omp_lib
   integer :: section_count = 0
   integer, parameter :: NT = 4
   print *, 'section_count', section_count
!ERROR: invalid branch into an OpenMP structured block
!ERROR: invalid branch into an OpenMP structured block
!ERROR: invalid branch into an OpenMP structured block
   if (NT) 20, 30, 40
!ERROR: invalid branch into an OpenMP structured block
   goto 20
!$omp sections
   !$omp section
   print *, "This is a single statement structured block"
   !$omp section
   open (10, file="random-file-name.txt", err=30)
   !ERROR: invalid branch into an OpenMP structured block
   !ERROR: invalid branch leaving an OpenMP structured block
   open (10, file="random-file-name.txt", err=40)
   !$omp section
   section_count = section_count + 1
20 print *, 'Entering into section'
   call calledFromWithinSection()
   print *, 'section_count', section_count
   !$omp section
   section_count = section_count + 1
   print *, 'section_count', section_count
   !ERROR: invalid branch leaving an OpenMP structured block
   goto 10
   !$omp section
30 print *, "Error in opening file"
!$omp end sections
10 print *, 'Jump from section'

!$omp sections
   !$omp section
40 print *, 'Error in opening file'
!$omp end sections
end program OmpConstructSections01

function returnFromSections()
   !$omp sections
   !$omp section
   !ERROR: RETURN statement is not allowed in a SECTIONS construct
   RETURN
   !$omp end sections
end function

subroutine calledFromWithinSection()
   print *, "I am called from within a 'section' structured block"
   return
end subroutine calledFromWithinSection

subroutine continueWithinSections()
   integer i
   do i = 1, 10
      print *, "Statement within loop but outside section construct"
      !$omp sections
      !$omp section
      IF (i .EQ. 5) THEN
         !ERROR: CYCLE to construct outside of SECTIONS construct is not allowed
         CYCLE
      END IF
      !$omp end sections
      print *, "Statement within loop but outside section contruct"
   end do

   !$omp sections
   !$omp section
   do i = 1, 10
      CYCLE
   end do
   !$omp end sections

   !$omp sections
   !$omp section
   loop_1: do i = 1, 10
      IF (i .EQ. 5) THEN
         CYCLE loop_1
      END IF
   end do loop_1
   !$omp end sections

   loop_2: do i = 1, 10
      !$omp sections
      !$omp section
      IF (i .EQ. 5) THEN
         !ERROR: CYCLE to construct 'loop_2' outside of SECTIONS construct is not allowed
         CYCLE loop_2
      END IF
      !$omp end sections
   end do loop_2
end subroutine continueWithinSections

subroutine breakWithinSections()
   loop_3: do i = 1, 10
      !$omp sections
      !$omp section
      IF (i .EQ. 5) THEN
         !ERROR: EXIT to construct 'loop_3' outside of SECTIONS construct is not allowed
         EXIT loop_3
      END IF
      !$omp end sections
   end do loop_3

   loop_4: do i = 1, 10
      !$omp sections
      !$omp section
      IF (i .EQ. 5) THEN
         !ERROR: EXIT to construct outside of SECTIONS construct is not allowed
         EXIT
      END IF
      !$omp end sections
   end do loop_4

   !$omp sections
   !$omp section
   do i = 1, 10
      IF (i .EQ. 5) THEN
         EXIT
      END IF
   end do
   !$omp end sections

   !$omp sections
   !$omp section
   loop_5: do i = 1, 10
      IF (i .EQ. 5) THEN
         EXIT loop_5
      END IF
   end do loop_5
   !$omp end sections
end subroutine breakWithinSections
