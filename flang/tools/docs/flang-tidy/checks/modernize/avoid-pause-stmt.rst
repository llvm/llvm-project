.. title:: flang-tidy - modernize-avoid-pause-stmt

modernize-avoid-pause-stmt
==========================

Identifies usage of the PAUSE statement, which was deprecated in Fortran 95 and removed from the Fortran 2018 standard. Modern code should use READ or other interactive techniques instead.

.. code-block:: fortran

    program example
      implicit none

      print *, "Processing data..."
      pause  ! This will trigger a warning
      print *, "Continuing execution..."
    end program
