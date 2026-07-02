.. title:: flang-tidy - modernize-avoid-assign-stmt

modernize-avoid-assign-stmt
===========================

Warns about the use of obsolete ASSIGN and assigned GOTO statements, which were deprecated in Fortran 95 and removed from the Fortran 2018 standard. Modern Fortran code should use more structured alternatives.

.. code-block:: fortran

    program example
      implicit none
      integer :: i

      assign 100 to i  ! This will trigger a warning
      goto i, (100, 200, 300)  ! This will trigger a warning

    100 print *, "Label 100"
      stop
    200 print *, "Label 200"
      stop
    300 print *, "Label 300"
      stop
    end program
