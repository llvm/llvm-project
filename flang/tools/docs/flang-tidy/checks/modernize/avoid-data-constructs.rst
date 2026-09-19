.. title:: flang-tidy - modernize-avoid-data-constructs

modernize-avoid-data-constructs
===============================

Warns about the use of DATA statements, which provide a less flexible and less clear way to initialize variables compared to modern initialization approaches.

.. code-block:: fortran

    program example
      implicit none
      integer :: array(3)

      data array /1, 2, 3/  ! This will trigger a warning
      ! Better: array = [1, 2, 3]

      print *, array
    end program
