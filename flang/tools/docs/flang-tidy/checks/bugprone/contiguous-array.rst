.. title:: flang-tidy - bugprone-contiguous-array

bugprone-contiguous-array
=========================

Verifies that assumed-shape arrays passed to interfaces are declared with the CONTIGUOUS attribute. Using contiguous arrays can lead to better performance by allowing the compiler to optimize memory access patterns.

.. code-block:: fortran

    module example
      interface
        subroutine contig(A)
          real, contiguous, intent(in) :: A(:)
        end subroutine contig

        subroutine possibly_noncontig(A) ! This will trigger a warning
          real, intent(in) :: A(:)
        end subroutine possibly_noncontig
      end interface
    end module example
