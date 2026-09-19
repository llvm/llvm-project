.. title:: flang-tidy - bugprone-missing-action

bugprone-missing-action
=======================

Verifies that all OPEN statements include an ACTION specifier, and that file unit numbers are not constant literals. This helps ensure proper file handling behavior and avoids hard-coded unit numbers.

.. code-block:: fortran

    program example
      implicit none
      integer :: unit = 10
      open(10, file="data.txt")  ! This will trigger two warnings:
                                 ! Missing ACTION and constant unit number
      ! Better: open(unit, file="data.txt", action="read")
      write(10,*) "Hello"
      close(10)
    end program
