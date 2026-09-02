.. title:: flang-tidy - bugprone-subprogram-trampoline

bugprone-subprogram-trampoline
==============================

Warns when a contained subprogram is passed as an actual argument to another procedure. This pattern can lead to unexpected behavior due to the way procedure pointers and contained subprograms interact.

.. code-block:: fortran

    program example
      implicit none
      call process(inner)  ! This will trigger a warning
    contains
      subroutine inner()
        print *, "Inside inner"
      end subroutine

      subroutine process(proc)
        interface
          subroutine proc()
          end subroutine
        end interface
        call proc()
      end subroutine
    end program
