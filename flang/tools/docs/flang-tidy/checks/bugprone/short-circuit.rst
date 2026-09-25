.. title:: flang-tidy - bugprone-short-circuit

bugprone-short-circuit
======================

Detects optional arguments used in logical expressions alongside ``present()`` calls. This causes undefined behavior since Fortran logical operators don't guarantee short-circuit evaluation.

.. code-block:: fortran

    subroutine example(a)
      integer, optional, intent(in) :: a

      ! Warning - undefined behavior
      if (present(a) .and. a > 0) then
        ! ...
      end if

      ! Correct approach
      if (present(a)) then
        if (a > 0) then
          ! ...
        end if
      end if
    end subroutine
