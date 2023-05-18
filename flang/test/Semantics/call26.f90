! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  contains
    subroutine simple_arg(x)
      integer, intent(in) :: x
    end subroutine simple_arg
    subroutine procedure_arg(x)
      procedure(simple_arg) :: x
    end subroutine procedure_arg
    subroutine s
      !ERROR: Alternate return label '42' cannot be associated with dummy argument 'x='
      call simple_arg(*42)
      !ERROR: Alternate return label '42' cannot be associated with dummy argument 'x='
      call procedure_arg(*42)
      42 stop
    end subroutine s
end module m
