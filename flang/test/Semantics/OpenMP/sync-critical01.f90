! RUN: %python %S/../test_errors.py %s %flang -fopenmp

! OpenMP Version 5.0
! 2.17.1 critical construct
! CRITICAL start and end CRITICAL directive names mismatch
integer function timer_tick_sec()
   implicit none
   integer t

   !$OMP CRITICAL
   t = t + 1
   !$OMP END CRITICAL

   !$OMP CRITICAL (foo)
   t = t + 1
   !$OMP END CRITICAL (foo)

   !$OMP CRITICAL (foo)
   t = t + 1
   !ERROR: The names on CRITICAL and END CRITICAL must match
   !$OMP END CRITICAL (bar)

   !$OMP CRITICAL (bar)
   t = t + 1
   !ERROR: The names on CRITICAL and END CRITICAL must match
   !$OMP END CRITICAL (foo)

   !ERROR: Either both CRITICAL and END CRITICAL should have an argument, or none of them should
   !$OMP CRITICAL (bar)
   t = t + 1
   !$OMP END CRITICAL

   !$OMP CRITICAL
   t = t + 1
   !ERROR: Either both CRITICAL and END CRITICAL should have an argument, or none of them should
   !$OMP END CRITICAL (foo)

   timer_tick_sec = t
   return

end function timer_tick_sec
