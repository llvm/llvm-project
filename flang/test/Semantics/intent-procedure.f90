! RUN: %python %S/test_errors.py %s %flang_fc1
! F2018 C843: Interface-body dummy procedures require POINTER to have INTENT.
subroutine test(in_proc, out_proc, inout_proc, ptr)
  interface
    !ERROR: A dummy procedure without the POINTER attribute may not have an INTENT attribute
    subroutine in_proc()
    end subroutine
    !ERROR: A dummy procedure without the POINTER attribute may not have an INTENT attribute
    integer function out_proc()
    end function
    !ERROR: A dummy procedure without the POINTER attribute may not have an INTENT attribute
    subroutine inout_proc()
    end subroutine
    subroutine ptr()
    end subroutine
  end interface
  pointer :: ptr
  intent(in) :: in_proc, ptr
  intent(out) :: out_proc
  intent(inout) :: inout_proc
end subroutine
