.. title:: flang-tidy - bugprone-mismatched-intent

bugprone-mismatched-intent
==========================


Warns when a variable is passed multiple times to a procedure with conflicting INTENT attributes or when a component of a variable is passed with less restrictive intent than its parent object.

.. code-block:: fortran

    program example
      implicit none
      real :: x = 1.0
      call update(x, x)  ! This will trigger a warning
    contains
      subroutine update(a, b)
        real, intent(in) :: a
        real, intent(out) :: b
        b = a * 2.0
      end subroutine
    end program
