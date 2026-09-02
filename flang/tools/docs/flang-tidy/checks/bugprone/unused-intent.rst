.. title:: flang-tidy - bugprone-unused-intent

bugprone-unused-intent
======================

Identifies dummy arguments with INTENT(INOUT) that are never modified, suggesting they should be INTENT(IN) instead. It also warns about dummy arguments without explicit intent declarations, which could lead to confusion about how they're used.

.. code-block:: fortran

    subroutine process(a, b, c)
      implicit none
      real, intent(in) :: a
      real, intent(inout) :: b  ! This will trigger a warning if b is never modified
      real :: c               ! This will trigger a warning for missing intent

      print *, a, b, c
    end subroutine
