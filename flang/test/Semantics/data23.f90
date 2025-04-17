! RUN: %python %S/test_errors.py %s %flang_fc1
program p
  interface
    subroutine s
    end subroutine
  end interface
  !ERROR: DATA statement initializations affect 'p' more than once
  procedure(s), pointer :: p
  type t
    procedure(s), pointer, nopass :: p
  end type
  !ERROR: DATA statement initializations affect 'x%p' more than once
  type(t) x
  data p /s/
  data p /s/
  data x%p /s/
  data x%p /s/
end
