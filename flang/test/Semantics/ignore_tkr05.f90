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

    subroutine s_ct(p)
      real, pointer :: p(:)
      !dir$ ignore_tkr(ct) p
    end subroutine

    subroutine s_c(p)
      integer, pointer :: p(:)
      !dir$ ignore_tkr(c) p
    end subroutine

    subroutine s_ck(p)
      real, pointer :: p(:)
      !dir$ ignore_tkr(ck) p
    end subroutine

    subroutine s_cr(p)
      real, pointer :: p(:)
      !dir$ ignore_tkr(cr) p
    end subroutine

    subroutine s_ckr(p)
      real, pointer :: p(:)
      !dir$ ignore_tkr(ckr) p
    end subroutine

    subroutine s_ctr(p)
      real, pointer :: p(:)
      !dir$ ignore_tkr(ctr) p
    end subroutine

    subroutine s_ctk(p)
      real, pointer :: p(:)
      !dir$ ignore_tkr(ctk) p
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

    ! ignore_tkr(c): still have type/kind/rank differences
    !ERROR: Actual argument type 'REAL(8)' is not compatible with dummy argument type 'INTEGER(4)'
    !ERROR: Pointer has rank 1 but target has rank 3
    call s_c(p3)

    ! ignore_tkr(ct): ignore type differences, still have kind/rank differences
    !ERROR: Actual argument type 'REAL(8)' is not compatible with dummy argument type 'REAL(4)'
    !ERROR: Pointer has rank 1 but target has rank 3
    call s_ct(p3)

    ! ignore_tkr(ck): ignore kind differences, still have type/rank differences
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 3
    !ERROR: Pointer has rank 1 but target has rank 3
    call s_ck(p3)

    ! ignore_tkr(cr): ignore rank differences, still have type/kind differences
    !ERROR: Actual argument type 'REAL(8)' is not compatible with dummy argument type 'REAL(4)'
    !ERROR: Target type REAL(8) is not compatible with pointer type REAL(4)
    call s_cr(p3)

    ! ignore_tkr(ckr): ignore kind/rank differences, still have type differences
    !ERROR: Target type REAL(8) is not compatible with pointer type REAL(4)
    call s_ckr(p3)

    ! ignore_tkr(ctr): ignore type/rank differences, still have kind differences
    !ERROR: Actual argument type 'REAL(8)' is not compatible with dummy argument type 'REAL(4)'
    call s_ctr(p3)

    ! ignore_tkr(ctk): ignore type/kind differences, still have rank differences
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 3
    !ERROR: Pointer has rank 1 but target has rank 3
    call s_ctk(p3)
  end subroutine
end module
