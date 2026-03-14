! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
program p
  interface
    subroutine s1
    end subroutine
    subroutine s2
    end subroutine
  end interface
  !ERROR: DATA statement initializations affect 'p1' more than once, distinctly
  procedure(s1), pointer :: p1
  !PORTABILITY: DATA statement initializations affect 'p2' more than once, identically [-Wmultiple-identical-data]
  procedure(s2), pointer :: p2
  type t
    procedure(s1), pointer, nopass :: p
  end type
  !ERROR: DATA statement initializations affect 'x1%p' more than once, distinctly
  !PORTABILITY: DATA statement initializations affect 'x2%p' more than once, identically [-Wmultiple-identical-data]
  type(t) x1, x2
  !PORTABILITY: Procedure pointer 'p1' in a DATA statement is not standard [-Wdata-stmt-extensions]
  data p1 /s1/
  !PORTABILITY: Procedure pointer 'p1' in a DATA statement is not standard [-Wdata-stmt-extensions]
  data p1 /s2/
  !PORTABILITY: Procedure pointer 'p2' in a DATA statement is not standard [-Wdata-stmt-extensions]
  data p2 /s1/
  !PORTABILITY: Procedure pointer 'p2' in a DATA statement is not standard [-Wdata-stmt-extensions]
  data p2 /s1/
  !PORTABILITY: Procedure pointer 'p' in a DATA statement is not standard [-Wdata-stmt-extensions]
  data x1%p /s1/
  !PORTABILITY: Procedure pointer 'p' in a DATA statement is not standard [-Wdata-stmt-extensions]
  data x1%p /s2/
  !PORTABILITY: Procedure pointer 'p' in a DATA statement is not standard [-Wdata-stmt-extensions]
  data x2%p /s1/
  !PORTABILITY: Procedure pointer 'p' in a DATA statement is not standard [-Wdata-stmt-extensions]
  data x2%p /s1/
end
