! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror -pedantic
! This test is responsible for checking the fix for passing non-variables as
! actual arguments to subroutines/functions whose corresponding dummy argument
! expects a VOLATILE variable
! c.f. llvm-project GitHub issue #58973

module m
  contains
  subroutine vol_dum_int(my_int)
    integer, volatile :: my_int
  end subroutine vol_dum_int

  subroutine vol_dum_real(my_real)
    real, volatile :: my_real
  end subroutine vol_dum_real

  subroutine vol_dum_complex(my_complex)
    complex, volatile :: my_complex
  end subroutine vol_dum_complex

  subroutine vol_dum_int_arr(my_int_arr)
    integer, dimension(2,2), volatile :: my_int_arr
  end subroutine vol_dum_int_arr

  subroutine test_all_subprograms()
    !WARNING: actual argument associated with VOLATILE dummy argument 'my_int=' is not a variable
    call vol_dum_int(6)
    !WARNING: actual argument associated with VOLATILE dummy argument 'my_int=' is not a variable
    call vol_dum_int(6+12)
    !WARNING: actual argument associated with VOLATILE dummy argument 'my_int=' is not a variable
    call vol_dum_int(6*12)
    !WARNING: actual argument associated with VOLATILE dummy argument 'my_int=' is not a variable
    call vol_dum_int(-6/2)
    !WARNING: actual argument associated with VOLATILE dummy argument 'my_real=' is not a variable
    call vol_dum_real(3.141592653)
    !WARNING: actual argument associated with VOLATILE dummy argument 'my_real=' is not a variable
    call vol_dum_real(3.141592653 + (-10.6e-11))
    !WARNING: actual argument associated with VOLATILE dummy argument 'my_real=' is not a variable
    call vol_dum_real(3.141592653 * 10.6e-11)
    !WARNING: actual argument associated with VOLATILE dummy argument 'my_real=' is not a variable
    call vol_dum_real(3.141592653 / (-10.6e-11))
    !WARNING: actual argument associated with VOLATILE dummy argument 'my_complex=' is not a variable
    call vol_dum_complex((1., 3.2))
    !WARNING: actual argument associated with VOLATILE dummy argument 'my_complex=' is not a variable
    call vol_dum_complex((1., 3.2) + (-2., 3.14))
    !WARNING: actual argument associated with VOLATILE dummy argument 'my_complex=' is not a variable
    call vol_dum_complex((1., 3.2) * (-2., 3.14))
    !WARNING: actual argument associated with VOLATILE dummy argument 'my_complex=' is not a variable
    call vol_dum_complex((1., 3.2) / (-2., 3.14))
    !WARNING: actual argument associated with VOLATILE dummy argument 'my_int_arr=' is not a variable
    call vol_dum_int_arr((/ 1, 2, 3, 4 /))
    !WARNING: actual argument associated with VOLATILE dummy argument 'my_int_arr=' is not a variable
    call vol_dum_int_arr(reshape((/ 1, 2, 3, 4 /), (/ 2, 2/)))
    !WARNING: actual argument associated with VOLATILE dummy argument 'my_int_arr=' is not a variable
    call vol_dum_int_arr((/ 1, 2, 3, 4 /))
  end subroutine test_all_subprograms
end module m
