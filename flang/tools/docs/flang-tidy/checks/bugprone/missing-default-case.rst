.. title:: flang-tidy - bugprone-missing-default-case

bugprone-missing-default-case
=============================

Ensures that all SELECT CASE constructs include a DEFAULT case to handle unexpected values. This makes code more robust by providing a catch-all for unforeseen values.

.. code-block:: fortran

    program example
      implicit none
      integer :: i = 3
      select case (i)  ! This will trigger a warning
        case (1)
          print *, "One"
        case (2)
          print *, "Two"
        ! Missing: case default
      end select
    end program
