.. title:: flang-tidy - bugprone-precision-loss

bugprone-precision-loss
=======================

Detects potential loss of precision in assignments between variables of different types or kind parameters. This helps prevent subtle numerical errors in computation.

.. code-block:: fortran

    program example
      implicit none
      real(kind=8) :: x = 1.0d0
      real(kind=4) :: y
      y = x  ! This will trigger a warning - precision loss
      print *, y
    end program
