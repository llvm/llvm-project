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
    !WARNING: Actual argument associated with VOLATILE dummy argument 'my_int=' is not definable
    !BECAUSE: '6_4' is not a variable or pointer
    call vol_dum_int(6)
    !WARNING: Actual argument associated with VOLATILE dummy argument 'my_int=' is not definable
    !BECAUSE: '18_4' is not a variable or pointer
    call vol_dum_int(6+12)
    !WARNING: Actual argument associated with VOLATILE dummy argument 'my_int=' is not definable
    !BECAUSE: '72_4' is not a variable or pointer
    call vol_dum_int(6*12)
    !WARNING: Actual argument associated with VOLATILE dummy argument 'my_int=' is not definable
    !BECAUSE: '-3_4' is not a variable or pointer
    call vol_dum_int(-6/2)
    !WARNING: Actual argument associated with VOLATILE dummy argument 'my_real=' is not definable
    !BECAUSE: '3.1415927410125732421875_4' is not a variable or pointer
    call vol_dum_real(3.141592653)
    !WARNING: Actual argument associated with VOLATILE dummy argument 'my_real=' is not definable
    !BECAUSE: '3.1415927410125732421875_4' is not a variable or pointer
    call vol_dum_real(3.141592653 + (-10.6e-11))
    !WARNING: Actual argument associated with VOLATILE dummy argument 'my_real=' is not definable
    !BECAUSE: '3.3300884272335906644002534449100494384765625e-10_4' is not a variable or pointer
    call vol_dum_real(3.141592653 * 10.6e-11)
    !WARNING: Actual argument associated with VOLATILE dummy argument 'my_real=' is not definable
    !BECAUSE: '-2.9637666816e10_4' is not a variable or pointer
    call vol_dum_real(3.141592653 / (-10.6e-11))
    !WARNING: Actual argument associated with VOLATILE dummy argument 'my_complex=' is not definable
    !BECAUSE: '(1._4,3.2000000476837158203125_4)' is not a variable or pointer
    call vol_dum_complex((1., 3.2))
    !WARNING: Actual argument associated with VOLATILE dummy argument 'my_complex=' is not definable
    !BECAUSE: '(-1._4,6.340000152587890625_4)' is not a variable or pointer
    call vol_dum_complex((1., 3.2) + (-2., 3.14))
    !WARNING: Actual argument associated with VOLATILE dummy argument 'my_complex=' is not definable
    !BECAUSE: '(-1.2048000335693359375e1_4,-3.2599999904632568359375_4)' is not a variable or pointer
    call vol_dum_complex((1., 3.2) * (-2., 3.14))
    !WARNING: Actual argument associated with VOLATILE dummy argument 'my_complex=' is not definable
    !BECAUSE: '(5.80680549144744873046875e-1_4,-6.8833148479461669921875e-1_4)' is not a variable or pointer
    call vol_dum_complex((1., 3.2) / (-2., 3.14))
    !WARNING: Actual argument associated with VOLATILE dummy argument 'my_int_arr=' is not definable
    !BECAUSE: '[INTEGER(4)::1_4,2_4,3_4,4_4]' is not a variable or pointer
    call vol_dum_int_arr((/ 1, 2, 3, 4 /))
    !WARNING: Actual argument associated with VOLATILE dummy argument 'my_int_arr=' is not definable
    !BECAUSE: 'reshape([INTEGER(4)::1_4,2_4,3_4,4_4],shape=[2,2])' is not a variable or pointer
    call vol_dum_int_arr(reshape((/ 1, 2, 3, 4 /), (/ 2, 2/)))
    !WARNING: Actual argument associated with VOLATILE dummy argument 'my_int_arr=' is not definable
    !BECAUSE: '[INTEGER(4)::1_4,2_4,3_4,4_4]' is not a variable or pointer
    call vol_dum_int_arr((/ 1, 2, 3, 4 /))
  end subroutine test_all_subprograms
end module m
