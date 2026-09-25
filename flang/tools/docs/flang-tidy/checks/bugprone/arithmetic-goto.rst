.. title:: flang-tidy - bugprone-arithmetic-goto

bugprone-arithmetic-goto
========================

Warns about the use of computed GOTO statements, which are considered problematic for code maintainability and clarity. The arithmetic (computed) GOTO is an obsolescent feature in Fortran that allows for complex and hard-to-maintain transfer of control logic. Modern code should use more structured control statements instead.

.. code-block:: fortran

    program example
      integer :: i = 2
      go to (10, 20, 30), i  ! This will trigger a warning
    10 print *, "Label 10"
      stop
    20 print *, "Label 20"
      stop
    30 print *, "Label 30"
      stop
    end program
