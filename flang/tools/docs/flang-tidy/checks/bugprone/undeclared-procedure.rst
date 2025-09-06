.. title:: flang-tidy - bugprone-undeclared-procedure

bugprone-undeclared-procedure
=============================

Warns about procedures that are implicitly declared or lack an explicit interface. Using procedures without explicit interfaces can lead to subtle bugs and prevents the compiler from performing argument checking.

.. code-block:: fortran

    program example
      implicit none
      integer :: result
      external :: no_interface_procedure
      call no_interface_procedure(5, result)  ! This will trigger a warning (no interface)
      call implicit_procedure(5, result)  ! This will also trigger a warning (implicit declaration)
      print *, result
    end program
