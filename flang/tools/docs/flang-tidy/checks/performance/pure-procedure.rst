.. title:: flang-tidy - performance-pure-procedure

performance-pure-procedure
==========================

Identifies procedures that could be declared as PURE but are not. PURE procedures have no side effects and can enable compiler optimizations, particularly for array operations and parallel processing. This check analyzes procedures and suggests adding the PURE attribute when possible.

.. code-block:: fortran

    ! This will trigger a warning - could be PURE
    subroutine add(a, b, result)
      implicit none
      integer, intent(in) :: a, b
      integer, intent(out) :: result
      result = a + b
    end subroutine

    ! Fixed version
    pure subroutine add_pure(a, b, result)
      implicit none
      integer, intent(in) :: a, b
      integer, intent(out) :: result
      result = a + b
    end subroutine
