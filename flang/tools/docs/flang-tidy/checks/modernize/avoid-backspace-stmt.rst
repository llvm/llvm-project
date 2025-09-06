.. title:: flang-tidy - modernize-avoid-backspace-stmt

modernize-avoid-backspace-stmt
==============================

Identifies usage of the BACKSPACE statement, which can lead to inefficient I/O operations. Modern Fortran code should use more reliable file positioning methods.

.. code-block:: fortran

    program example
      implicit none
      integer :: unit = 10, x

      open(unit, file="data.txt", action="read")
      read(unit, *) x
      backspace(unit)  ! This will trigger a warning
      read(unit, *) x
      close(unit)
    end program
