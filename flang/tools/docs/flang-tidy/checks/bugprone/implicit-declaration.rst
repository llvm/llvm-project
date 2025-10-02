.. title:: flang-tidy - bugprone-implicit-declaration

bugprone-implicit-declaration
=============================

Warns about implicitly declared variables, which can lead to subtle bugs. All variables should be explicitly declared to catch typos and ensure proper typing.

.. code-block:: fortran

    program example
      ! Missing: implicit none
      integer :: i
      i = 10
      j = 20  ! This will trigger a warning - j is implicitly declared
      print *, i + j
    end program
