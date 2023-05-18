! RUN: %python %S/test_folding.py %s %flang_fc1
! Test operation folding edge case (both expected value and messages)
! These tests make assumptions regarding real(4) and integer(4) extrema.

#define TEST_ISNAN(v) logical, parameter :: test_##v =.NOT.(v.EQ.v)

module integer_tests
  integer(4), parameter :: i4_pmax = 2147483647_4
  ! Fortran grammar rule R605 prevents from writing -2147483648_4 in an
  ! expression because literal-constant are not signed so this would parse
  ! to -(2147483648_4) and 2147483648_4 is not accepted as a literal-constant.
  ! However, one can reach this value with operations.
  integer(4), parameter :: i4_nmax = -2147483647_4 - 1_4

  ! Integer division by zero are not tested here because they are handled as fatal
  ! errors in constants.

  !WARN: warning: INTEGER(4) negation overflowed
  logical, parameter :: test_overflow_unary_minus1 = (-i4_nmax).EQ.i4_nmax
  logical, parameter :: test_no_overflow_unary_minus1 = (-i4_pmax).EQ.(i4_nmax+1_4)
  logical, parameter :: test_no_overflow_unary_plus1 = (+i4_pmax).EQ.i4_pmax
  logical, parameter :: test_no_overflow_unary_plus2 = (+i4_nmax).EQ.i4_nmax

  !WARN: warning: INTEGER(4) addition overflowed
  logical, parameter :: test_overflow_add1 = (i4_pmax+1_4).EQ.i4_nmax
  !WARN: warning: INTEGER(4) addition overflowed
  logical, parameter :: test_overflow_add2 = (i4_nmax + (-1_4)).EQ.i4_pmax
  !WARN: warning: INTEGER(4) addition overflowed
  logical, parameter :: test_overflow_add3 = (i4_pmax + i4_pmax).EQ.(-2_4)
  !WARN: warning: INTEGER(4) addition overflowed
  logical, parameter :: test_overflow_add4 = (i4_nmax + i4_nmax).EQ.(0_4)
  logical, parameter :: test_no_overflow_add1 = (i4_pmax + 0_4).EQ.i4_pmax
  logical, parameter :: test_no_overflow_add2 = (i4_nmax + (-0_4)).EQ.i4_nmax
  logical, parameter :: test_no_overflow_add3 = (i4_pmax + i4_nmax).EQ.(-1_4)
  logical, parameter :: test_no_overflow_add4 = (i4_nmax + i4_pmax).EQ.(-1_4)

  !WARN: warning: INTEGER(4) subtraction overflowed
  logical, parameter :: test_overflow_sub1 = (i4_nmax - 1_4).EQ.i4_pmax
  !WARN: warning: INTEGER(4) subtraction overflowed
  logical, parameter :: test_overflow_sub2 = (i4_pmax - (-1_4)).EQ.i4_nmax
  !WARN: warning: INTEGER(4) subtraction overflowed
  logical, parameter :: test_overflow_sub3 = (i4_nmax - i4_pmax).EQ.(1_4)
  !WARN: warning: INTEGER(4) subtraction overflowed
  logical, parameter :: test_overflow_sub4 = (i4_pmax - i4_nmax).EQ.(-1_4)
  logical, parameter :: test_no_overflow_sub1 = (i4_nmax - 0_4).EQ.i4_nmax
  logical, parameter :: test_no_overflow_sub2 = (i4_pmax - (-0_4)).EQ.i4_pmax
  logical, parameter :: test_no_overflow_sub3 = (i4_nmax - i4_nmax).EQ.0_4
  logical, parameter :: test_no_overflow_sub4 = (i4_pmax - i4_pmax).EQ.0_4


  !WARN: warning: INTEGER(4) multiplication overflowed
  logical, parameter :: test_overflow_mult1 = (i4_pmax*2_4).EQ.(-2_4)
  !WARN: warning: INTEGER(4) multiplication overflowed
  logical, parameter :: test_overflow_mult2 = (i4_nmax*2_4).EQ.(0_4)
  !WARN: warning: INTEGER(4) multiplication overflowed
  logical, parameter :: test_overflow_mult3 = (i4_nmax*i4_nmax).EQ.(0_4)
  !WARN: warning: INTEGER(4) multiplication overflowed
  logical, parameter :: test_overflow_mult4 = (i4_pmax*i4_pmax).EQ.(1_4)

  !WARN: warning: INTEGER(4) division overflowed
  logical, parameter :: test_overflow_div1 = (i4_nmax/(-1_4)).EQ.(i4_nmax)
  logical, parameter :: test_no_overflow_div1 = (i4_nmax/(-2_4)).EQ.(1_4 + i4_pmax/2_4)
  logical, parameter :: test_no_overflow_div2 = (i4_nmax/i4_nmax).EQ.(1_4)

  !WARN: warning: INTEGER(4) power overflowed
  logical, parameter :: test_overflow_pow1 = (i4_pmax**2_4).EQ.(1_4)
  !WARN: warning: INTEGER(4) power overflowed
  logical, parameter :: test_overflow_pow3 = (i4_nmax**2_4).EQ.(0_4)
  logical, parameter :: test_no_overflow_pow1 = ((-1_4)**i4_nmax).EQ.(1_4)
  logical, parameter :: test_no_overflow_pow2 = ((-1_4)**i4_pmax).EQ.(-1_4)

end module

module real_tests
  ! Test real(4) operation folding on edge cases (inf and NaN)

  real(4), parameter :: r4_pmax = 3.4028235E38
  real(4), parameter :: r4_nmax = -3.4028235E38
  !WARN: warning: invalid argument on division
  real(4), parameter :: r4_nan = 0._4/0._4
  TEST_ISNAN(r4_nan)
  !WARN: warning: division by zero
  real(4), parameter :: r4_pinf = 1._4/0._4
  !WARN: warning: division by zero
  real(4), parameter :: r4_ninf = -1._4/0._4

  logical, parameter :: test_r4_nan_parentheses1 = .NOT.(((r4_nan)).EQ.r4_nan)
  logical, parameter :: test_r4_nan_parentheses2 = .NOT.(((r4_nan)).LT.r4_nan)
  logical, parameter :: test_r4_nan_parentheses3 = .NOT.(((r4_nan)).GT.r4_nan)
  logical, parameter :: test_r4_nan_parentheses4 = ((r4_nan)).NE.r4_nan
  logical, parameter :: test_r4_pinf_parentheses = ((r4_pinf)).EQ.r4_pinf
  logical, parameter :: test_r4_ninf_parentheses = ((r4_ninf)).EQ.r4_ninf

  ! No warnings expected
  logical, parameter :: test_r4_negation1 = (-r4_pmax).EQ.r4_nmax
  logical, parameter :: test_r4_negation2 = (-r4_nmax).EQ.r4_pmax
  logical, parameter :: test_r4_negation3 = (-r4_pinf).EQ.r4_ninf
  logical, parameter :: test_r4_negation4 = (-r4_ninf).EQ.r4_pinf
  logical, parameter :: test_r4_plus1 = (+r4_pmax).EQ.r4_pmax
  logical, parameter :: test_r4_plus2 = (+r4_nmax).EQ.r4_nmax
  logical, parameter :: test_r4_plus3 = (+r4_pinf).EQ.r4_pinf
  logical, parameter :: test_r4_plus4 = (+r4_ninf).EQ.r4_ninf
  ! NaN propagation , no warnings expected (quiet)
  real(4), parameter :: r4_nan_minus = (-r4_nan)
  TEST_ISNAN(r4_nan_minus)
  real(4), parameter :: r4_nan_plus = (+r4_nan)
  TEST_ISNAN(r4_nan_plus)

  !WARN: warning: overflow on addition
  logical, parameter :: test_inf_r4_add9 = (r4_pmax + r4_pmax).eq.(r4_pinf)
  !WARN: warning: overflow on addition
  logical, parameter :: test_inf_r4_add10 = (r4_nmax + r4_nmax).eq.(r4_ninf)
  !WARN: warning: overflow on subtraction
  logical, parameter :: test_inf_r4_sub9 = (r4_pmax - r4_nmax).eq.(r4_pinf)
  !WARN: warning: overflow on subtraction
  logical, parameter :: test_inf_r4_sub10 = (r4_nmax - r4_pmax).eq.(r4_ninf)

  ! No warnings expected below (inf propagation).
  logical, parameter :: test_inf_r4_add1 = (r4_pinf + r4_pinf).EQ.(r4_pinf)
  logical, parameter :: test_inf_r4_add2 = (r4_ninf + r4_ninf).EQ.(r4_ninf)
  logical, parameter :: test_inf_r4_add3 = (r4_pinf + r4_nmax).EQ.(r4_pinf)
  logical, parameter :: test_inf_r4_add4 = (r4_pinf + r4_pmax).EQ.(r4_pinf)
  logical, parameter :: test_inf_r4_add5 = (r4_ninf + r4_pmax).EQ.(r4_ninf)
  logical, parameter :: test_inf_r4_add6 = (r4_ninf + r4_nmax).EQ.(r4_ninf)
  logical, parameter :: test_inf_r4_add7 = (r4_ninf + 0._4).EQ.(r4_ninf)
  logical, parameter :: test_inf_r4_add8 = (r4_pinf + 0._4).EQ.(r4_pinf)

  !WARN: warning: invalid argument on subtraction
  real(4), parameter :: r4_nan_sub1 = r4_pinf - r4_pinf
  TEST_ISNAN(r4_nan_sub1)
  !WARN: warning: invalid argument on subtraction
  real(4), parameter :: r4_nan_sub2 = r4_ninf - r4_ninf
  TEST_ISNAN(r4_nan_sub2)
  !WARN: warning: invalid argument on addition
  real(4), parameter :: r4_nan_add1 = r4_ninf + r4_pinf
  TEST_ISNAN(r4_nan_add1)
  !WARN: warning: invalid argument on addition
  real(4), parameter :: r4_nan_add2 = r4_pinf + r4_ninf
  TEST_ISNAN(r4_nan_add2)

  ! No warnings expected here (quiet NaN propagation)
  real(4), parameter :: r4_nan_sub3 = 0._4 - r4_nan
  TEST_ISNAN(r4_nan_sub3)
  real(4), parameter :: r4_nan_sub4 = r4_nan - r4_pmax
  TEST_ISNAN(r4_nan_sub4)
  real(4), parameter :: r4_nan_sub5 = r4_nan - r4_nmax
  TEST_ISNAN(r4_nan_sub5)
  real(4), parameter :: r4_nan_sub6 = r4_nan - r4_nan
  TEST_ISNAN(r4_nan_sub6)
  real(4), parameter :: r4_nan_add3 = 0._4 + r4_nan
  TEST_ISNAN(r4_nan_add3)
  real(4), parameter :: r4_nan_add4 = r4_nan + r4_pmax
  TEST_ISNAN(r4_nan_add4)
  real(4), parameter :: r4_nan_add5 = r4_nmax + r4_nan
  TEST_ISNAN(r4_nan_add5)
  real(4), parameter :: r4_nan_add6 = r4_nan + r4_nan
  TEST_ISNAN(r4_nan_add6)

  !WARN: warning: overflow on multiplication
  logical, parameter :: test_inf_r4_mult1 = (1.5_4*r4_pmax).eq.(r4_pinf)
  !WARN: warning: overflow on multiplication
  logical, parameter :: test_inf_r4_mult2 = (1.5_4*r4_nmax).eq.(r4_ninf)
  !WARN: warning: overflow on division
  logical, parameter :: test_inf_r4_div1 = (r4_nmax/(-0.5_4)).eq.(r4_pinf)
  !WARN: warning: overflow on division
  logical, parameter :: test_inf_r4_div2 = (r4_pmax/(-0.5_4)).eq.(r4_ninf)

  ! No warnings expected below (inf propagation).
  logical, parameter :: test_inf_r4_mult3 = (r4_pinf*r4_pinf).EQ.(r4_pinf)
  logical, parameter :: test_inf_r4_mult4 = (r4_ninf*r4_ninf).EQ.(r4_pinf)
  logical, parameter :: test_inf_r4_mult5 = (r4_pinf*0.1_4).EQ.(r4_pinf)
  logical, parameter :: test_inf_r4_mult6 = (r4_ninf*r4_nmax).EQ.(r4_pinf)
  logical, parameter :: test_inf_r4_div3 = (r4_pinf/0.).EQ.(r4_pinf)
  logical, parameter :: test_inf_r4_div4 = (r4_ninf/0.).EQ.(r4_ninf)
  logical, parameter :: test_inf_r4_div5 = (0./r4_pinf).EQ.(0.)
  logical, parameter :: test_inf_r4_div6 = (0./r4_ninf).EQ.(0.)
  logical, parameter :: test_inf_r4_div7 = (r4_pinf/r4_pmax).EQ.(r4_pinf)
  logical, parameter :: test_inf_r4_div8 = (r4_pinf/r4_nmax).EQ.(r4_ninf)
  logical, parameter :: test_inf_r4_div9 = (r4_nmax/r4_pinf).EQ.(0.)
  logical, parameter :: test_inf_r4_div10 = (r4_nmax/r4_ninf).EQ.(0.)

  !WARN: warning: invalid argument on division
  real(4), parameter :: r4_nan_div1 = 0._4/0._4
  TEST_ISNAN(r4_nan_div1)
  !WARN: warning: invalid argument on division
  real(4), parameter :: r4_nan_div2 = r4_ninf/r4_ninf
  TEST_ISNAN(r4_nan_div2)
  !WARN: warning: invalid argument on division
  real(4), parameter :: r4_nan_div3 = r4_ninf/r4_pinf
  TEST_ISNAN(r4_nan_div3)
  !WARN: warning: invalid argument on division
  real(4), parameter :: r4_nan_div4 = r4_pinf/r4_ninf
  TEST_ISNAN(r4_nan_div4)
  !WARN: warning: invalid argument on division
  real(4), parameter :: r4_nan_div5 = r4_pinf/r4_pinf
  TEST_ISNAN(r4_nan_div5)
  !WARN: warning: invalid argument on multiplication
  real(4), parameter :: r4_nan_mult1 = r4_pinf*0._4
  TEST_ISNAN(r4_nan_mult1)
  !WARN: warning: invalid argument on multiplication
  real(4), parameter :: r4_nan_mult2 = 0._4*r4_ninf
  TEST_ISNAN(r4_nan_mult2)

  ! No warnings expected here (quiet NaN propagation)
  real(4), parameter :: r4_nan_div6 = 0._4/r4_nan
  TEST_ISNAN(r4_nan_div6)
  real(4), parameter :: r4_nan_div7 = r4_nan/r4_nan
  TEST_ISNAN(r4_nan_div7)
  real(4), parameter :: r4_nan_div8 = r4_nan/0._4
  TEST_ISNAN(r4_nan_div8)
  real(4), parameter :: r4_nan_div9 = r4_nan/1._4
  TEST_ISNAN(r4_nan_div9)
  real(4), parameter :: r4_nan_mult3 = r4_nan*1._4
  TEST_ISNAN(r4_nan_mult3)
  real(4), parameter :: r4_nan_mult4 = r4_nan*r4_nan
  TEST_ISNAN(r4_nan_mult4)
  real(4), parameter :: r4_nan_mult5 = 0._4*r4_nan
  TEST_ISNAN(r4_nan_mult5)

  ! TODO: ** operator folding
  !  logical, parameter :: test_inf_r4_exp1 = (r4_pmax**2._4).EQ.(r4_pinf)

  ! Relational operator edge cases (No warnings expected?)
  logical, parameter :: test_inf_r4_eq1 = r4_pinf.EQ.r4_pinf
  logical, parameter :: test_inf_r4_eq2 = r4_ninf.EQ.r4_ninf
  logical, parameter :: test_inf_r4_eq3 = .NOT.(r4_pinf.EQ.r4_ninf)
  logical, parameter :: test_inf_r4_eq4 = .NOT.(r4_pinf.EQ.r4_pmax)

  logical, parameter :: test_inf_r4_ne1 = .NOT.(r4_pinf.NE.r4_pinf)
  logical, parameter :: test_inf_r4_ne2 = .NOT.(r4_ninf.NE.r4_ninf)
  logical, parameter :: test_inf_r4_ne3 = r4_pinf.NE.r4_ninf
  logical, parameter :: test_inf_r4_ne4 = r4_pinf.NE.r4_pmax

  logical, parameter :: test_inf_r4_gt1 = .NOT.(r4_pinf.GT.r4_pinf)
  logical, parameter :: test_inf_r4_gt2 = .NOT.(r4_ninf.GT.r4_ninf)
  logical, parameter :: test_inf_r4_gt3 = r4_pinf.GT.r4_ninf
  logical, parameter :: test_inf_r4_gt4 = r4_pinf.GT.r4_pmax

  logical, parameter :: test_inf_r4_lt1 = .NOT.(r4_pinf.LT.r4_pinf)
  logical, parameter :: test_inf_r4_lt2 = .NOT.(r4_ninf.LT.r4_ninf)
  logical, parameter :: test_inf_r4_lt3 = r4_ninf.LT.r4_pinf
  logical, parameter :: test_inf_r4_lt4 = r4_pmax.LT.r4_pinf

  logical, parameter :: test_inf_r4_ge1 = r4_pinf.GE.r4_pinf
  logical, parameter :: test_inf_r4_ge2 = r4_ninf.GE.r4_ninf
  logical, parameter :: test_inf_r4_ge3 = .NOT.(r4_ninf.GE.r4_pinf)
  logical, parameter :: test_inf_r4_ge4 = .NOT.(r4_pmax.GE.r4_pinf)

  logical, parameter :: test_inf_r4_le1 = r4_pinf.LE.r4_pinf
  logical, parameter :: test_inf_r4_le2 = r4_ninf.LE.r4_ninf
  logical, parameter :: test_inf_r4_le3 = .NOT.(r4_pinf.LE.r4_ninf)
  logical, parameter :: test_inf_r4_le4 = .NOT.(r4_pinf.LE.r4_pmax)

  ! Invalid relational argument
  logical, parameter :: test_nan_r4_eq1 = .NOT.(r4_nan.EQ.r4_nan)
  logical, parameter :: test_nan_r4_lt1 = .NOT.(r4_nan.LE.r4_nan)
  logical, parameter :: test_nan_r4_gt1 = .NOT.(r4_nan.GT.r4_nan)
  logical, parameter :: test_nan_r4_ne1 = r4_nan.NE.r4_nan

end module

! TODO: edge case conversions
! TODO: complex tests (or is real tests enough?)

! Logical operation (with logical arguments) cannot overflow or be invalid.
! CHARACTER folding operations may cause host memory exhaustion if the
! string are very large. This will cause a fatal error for the program
! doing folding (e.g. f18), so there is nothing very interesting to test here.
