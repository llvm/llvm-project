.. title:: flang-tidy - modernize-avoid-common-blocks

modernize-avoid-common-blocks
=============================

Detects the use of COMMON blocks, which were replaced by modules in Fortran 90 for data sharing. Using modules provides better encapsulation, type safety, and explicit interfaces.

.. code-block:: fortran

    program example
      implicit none
      integer :: a, b
      common /data/ a, b  ! This will trigger a warning

      a = 1
      b = 2
      call process()
    end program

    subroutine process()
      implicit none
      integer :: a, b
      common /data/ a, b

      print *, a, b
    end subroutine
