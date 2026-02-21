.. title:: flang-tidy - bugprone-implied-save

bugprone-implied-save
=====================

Identifies variables that are implicitly saved between procedure invocations. Variables with implied SAVE status can cause unexpected behavior and side effects, particularly in parallel code.

.. code-block:: fortran

    subroutine example
      integer :: counter = 0  ! This will trigger a warning - implied SAVE
      ! Should be: integer, save :: counter = 0
      counter = counter + 1
      print *, "Called", counter, "times"
    end subroutine
