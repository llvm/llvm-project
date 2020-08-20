! RUN: %S/test_folding.sh %s %t %f18

! Test intrinsic operation folding

module m
! Check logical intrinsic operation folding
  logical, parameter :: test_not1 = .NOT..false.
  logical, parameter :: test_not2 = .NOT..NOT..true.

  logical, parameter :: test_parentheses1 = .NOT.(.false.)
  logical, parameter :: test_parentheses2 = .NOT..NOT.(.true.)

  logical, parameter :: test_and1 = .true..AND..true.
  logical, parameter :: test_and2 = .NOT.(.false..AND..true.)
  logical, parameter :: test_and3 = .NOT.(.false..AND..false.)
  logical, parameter :: test_and4 = .NOT.(.true..AND..false.)

  logical, parameter :: test_or1 = .true..OR..true.
  logical, parameter :: test_or2 = .false..OR..true.
  logical, parameter :: test_or3 = .NOT.(.false..OR..false.)
  logical, parameter :: test_or4 = .true..OR..false.

  logical, parameter :: test_eqv1 = .false..EQV..false.
  logical, parameter :: test_eqv2 = .true..EQV..true.
  logical, parameter :: test_eqv3 = .NOT.(.false..EQV..true.)
  logical, parameter :: test_eqv4 = .NOT.(.true..EQV..false.)

  logical, parameter :: test_neqv1 = .true..NEQV..false.
  logical, parameter :: test_neqv2 = .false..NEQV..true.
  logical, parameter :: test_neqv3 = .NOT.(.false..NEQV..false.)
  logical, parameter :: test_neqv4 = .NOT.(.true..NEQV..true.)

! Check integer intrinsic operator folding

! Check integer relational intrinsic operation folding
  logical, parameter :: test_le_i1 = 1.LE.2
  logical, parameter :: test_le_i2 = .NOT.(2.LE.1)
  logical, parameter :: test_le_i3 = 2.LE.2
  logical, parameter :: test_le_i4 = -1.LE.2
  logical, parameter :: test_le_i5 = .NOT.(-2.LE.-3)

  logical, parameter :: test_lt_i1 = 1.LT.2
  logical, parameter :: test_lt_i2 = .NOT.(2.LT.1)
  logical, parameter :: test_lt_i3 = .NOT.(2.LT.2)
  logical, parameter :: test_lt_i4 = -1.LT.2
  logical, parameter :: test_lt_i5 = .NOT.(-2.LT.-3)

  logical, parameter :: test_ge_i1 = .NOT.(1.GE.2)
  logical, parameter :: test_ge_i2 =  2.GE.1
  logical, parameter :: test_ge_i3 = 2.GE.2
  logical, parameter :: test_ge_i4 = .NOT.(-1.GE.2)
  logical, parameter :: test_ge_i5 = -2.GE.-3

  logical, parameter :: test_gt_i1 = .NOT.(1.GT.2)
  logical, parameter :: test_gt_i2 =  2.GT.1
  logical, parameter :: test_gt_i3 = .NOT.(2.GT.2)
  logical, parameter :: test_gt_i4 = .NOT.(-1.GT.2)
  logical, parameter :: test_gt_i5 = -2.GT.-3

  logical, parameter :: test_eq_i1 = 2.EQ.2
  logical, parameter :: test_eq_i2 = .NOT.(-2.EQ.2)

  logical, parameter :: test_ne_i1 =.NOT.(2.NE.2)
  logical, parameter :: test_ne_i2 = -2.NE.2

! Check conversions
  logical, parameter :: test_cmplx1 = cmplx((1._4, -1._4)).EQ.((1._4, -1._4))
  logical, parameter :: test_cmplx2 = cmplx((1._4, -1._4), 8).EQ.((1._8, -1._8))
  logical, parameter :: test_cmplx3 = cmplx(1._4, -1._4).EQ.((1._4, -1._4))
  logical, parameter :: test_cmplx4 = cmplx(1._4, -1._4, 8).EQ.((1._8, -1._8))
  logical, parameter :: test_cmplx5 = cmplx(1._4).EQ.((1._4, 0._4))
  logical, parameter :: test_cmplx6 = cmplx(1._4, kind=8).EQ.((1._8, 0._8))

! Check integer intrinsic operation folding
  logical, parameter :: test_unaryminus_i = (-(-1)).EQ.1
  logical, parameter :: test_unaryplus_i = (+1).EQ.1

  logical, parameter :: test_plus_i1 = (1+1).EQ.2
  logical, parameter :: test_plus_i2 = ((-3)+1).EQ.-2

  logical, parameter :: test_minus_i1 = (1-1).EQ.0
  logical, parameter :: test_minus_i2 = (1-(-1)).EQ.2

  logical, parameter :: test_multiply_i1 = (2*2).EQ.4
  logical, parameter :: test_multiply_i2 = (0*1).EQ.0
  logical, parameter :: test_multiply_i3= ((-3)*2).EQ.(-6)

  logical, parameter :: test_divide_i1 = (5/3).EQ.(1)
  logical, parameter :: test_divide_i2 = (6/3).EQ.(2)
  logical, parameter :: test_divide_i3 = ((-7)/2).EQ.(-3)
  logical, parameter :: test_divide_i4 = (0/127).EQ.(0)

  logical, parameter :: test_pow1 = (2**0).EQ.(1)
  logical, parameter :: test_pow2 = (1**100).EQ.(1)
  logical, parameter :: test_pow3 = (2**4).EQ.(16)
  logical, parameter :: test_pow4 = (7**5).EQ.(16807)
  logical, parameter :: test_pow5 = kind(real(1., kind=8)**cmplx(1., kind=4)).EQ.(8)
  logical, parameter :: test_pow6 = kind(cmplx(1., kind=4)**real(1., kind=8)).EQ.(8)

  ! test MIN and MAX
  real, parameter :: x1 = -35., x2= -35.05, x3=0., x4=35.05, x5=35.
  real, parameter :: res_max_r = max(x1, x2, x3, x4, x5)
  real, parameter :: res_min_r = min(x1, x2, x3, x4, x5)
  logical, parameter :: test_max_r = res_max_r.EQ.x4
  logical, parameter :: test_min_r = res_min_r.EQ.x2

  logical, parameter :: test_min_i = min(-3, 3).EQ.-3
  logical, parameter :: test_max_i = max(-3, 3).EQ.3
  integer, parameter :: i1 = 35, i2= 36, i3=0, i4=-35, i5=-36
  integer, parameter :: res_max_i = max(i1, i2, i3, i4, i5)
  integer, parameter :: res_min_i = min(i1, i2, i3, i4, i5)
  logical, parameter :: test_max_i2 = res_max_i.EQ.i2
  logical, parameter :: test_min_i2 = res_min_i.EQ.i5

  character(*), parameter :: c1 = "elephant", c2="elevator"
  character(*), parameter :: c3 = "excalibur", c4="z", c5="epsilon"
  character(*), parameter :: res_max_c = max(c1, c2, c3, c4, c5)
  character(*), parameter :: res_min_c = min(c1, c2, c3, c4, c5)
  ! length of result is length of longest arguments!
  character(len(c3)), parameter :: exp_min = c1
  character(len(c3)), parameter :: exp_max = c4
  logical, parameter :: test_max_c_1 = res_max_c.EQ.exp_max
  logical, parameter :: test_max_c_2 = res_max_c.NE.c4
  logical, parameter :: test_max_c_3 = len(res_max_c).EQ.len(c3)
  logical, parameter :: test_min_c_1 = res_min_c.NE.c1
  logical, parameter :: test_min_c_2 = res_min_c.EQ.exp_min
  logical, parameter :: test_min_c_3 = len(res_min_c).EQ.len(c3)

  integer, parameter :: x1a(*) = [1, 12, 3, 14]
  integer, parameter :: x2a(*) = [11, 2, 13, 4]
  logical, parameter :: test_max_a1 = all(max(x1a, x2a).EQ.[11, 12, 13, 14])
  logical, parameter :: test_min_a1 = all(min(x1a, x2a).EQ.[1, 2, 3, 4])

end module
