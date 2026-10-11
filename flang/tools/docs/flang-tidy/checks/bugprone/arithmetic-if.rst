.. title:: flang-tidy - bugprone-arithmetic-if

bugprone-arithmetic-if
======================

Detects the use of arithmetic IF statements, which are obsolescent in Fortran and should be replaced with more structured IF constructs for better readability and maintainability.

.. code-block:: fortran

    program example
      real :: x = -1.0
      if (x) 10, 20, 30  ! This will trigger a warning
    10 print *, "X is negative"
      stop
    20 print *, "X is zero"
      stop
    30 print *, "X is positive"
      stop
    end program
