! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Tests for ignore_tkr(ac) (allocatable/all + contiguous) with pointers
! Should suppress warnings about applying to pointer/descriptor
! and suppress errors about rank/type mismatch.

module m_ptr_tkr
  interface
    subroutine s1(p)
      real, pointer :: p(:)
      !dir$ ignore_tkr(ac) p
    end subroutine

    subroutine s2(p)
      real, allocatable :: p(:)
      !dir$ ignore_tkr(ac) p
    end subroutine
  end interface

contains
  subroutine test_ptr_tkr()
    real(8), pointer :: p3(:,:,:)
    real(8), allocatable :: a3(:,:,:)
    
    ! Rank mismatch (1 vs 3), Type mismatch (real(4) vs real(8))
    ! Should be ignored due to ignore_tkr(ac) which implies TKR + C.
    call s1(p3) 
    call s2(a3)
  end subroutine
end module
