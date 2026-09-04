! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of TRANSFER(...) when the physical representation of the
! result is longer than that of SOURCE.  F2023 16.9.212 p.5 requires the
! leading part of the result's physical representation to be that of
! SOURCE, and requires TRANSFER(TRANSFER(E, D), E) to have the value of E
! for scalar D and E (and likewise TRANSFER(TRANSFER(E, D), E, SIZE(E))
! when D is an array and E has rank one); the Examples paragraph's
! Case (ii) shows a trailing array element only partially covered by
! SOURCE.  The remainder of the result beyond SOURCE's representation is
! processor dependent; flang zero-fills it (as already pinned for
! CHARACTER by fold-transfer.f90's test_i2c_s).
! Same-size and mold-shorter values are covered by fold-transfer.f90;
! this file pins the mold-longer cases.  All checks are byte-order
! independent: the round trips prove the leading-part byte placement,
! and the two-endian .or. checks (idiom precedent: fold-transfer.f90's
! test_c2i_s) are portable value/zero-fill pins, not placement proofs.

module m
  ! Scalar MOLD longer than SOURCE: round trips (16.9.212 p.5), ...
  logical, parameter :: test_rt_scalar = transfer(transfer(1_4, 0_8), 0_4) == 1_4
  logical, parameter :: test_rt_neg = transfer(transfer(-1_4, 0_8), 0_4) == -1_4
  logical, parameter :: test_rt_real = transfer(transfer(1.5, 0._8), 0.0) == 1.5
  ! ... and a portable leading-part + zero-fill value pin (either
  ! byte order's correct value; placement is proven by the round trips)
  integer(8), parameter :: w1 = transfer(1_4, 0_8)
  logical, parameter :: test_lead_zfill = w1 == 1_8 .or. w1 == 4294967296_8

  ! Rank-one results whose trailing element is only partially covered
  ! by SOURCE, with and without SIZE=
  integer(8), parameter :: via8(2) = transfer([1_4, 2_4, 3_4], 0_8, 2)
  logical, parameter :: test_rt_array = all(transfer(via8, 0_4, 3) == [1_4, 2_4, 3_4])
  logical, parameter :: test_elem2_zfill = via8(2) == 3_8 .or. via8(2) == 12884901888_8
  integer(8), parameter :: via8b(*) = transfer([1_4, 2_4, 3_4], [0_8])
  logical, parameter :: test_rt_array2 = all(transfer(via8b, 0_4, 3) == [1_4, 2_4, 3_4])
  real(8), parameter :: rvia8(2) = transfer([1.5, 2.5, 3.5], 0._8, 2)
  logical, parameter :: test_rt_real_arr = all(transfer(rvia8, 0.0, 3) == [1.5, 2.5, 3.5])

  ! Rank-2 SOURCE: flattened, then the same trailing partial coverage
  ! and zero fill as the rank-one cases
  integer(8), parameter :: via8c(*) = transfer(reshape([1_4, 2_4, 3_4], [3, 1]), [0_8])
  logical, parameter :: test_rt_rank2 = all(transfer(via8c, 0_4, 3) == [1_4, 2_4, 3_4])
  logical, parameter :: test_rank2_zfill = via8c(2) == 3_8 .or. via8c(2) == 12884901888_8
  ! ... and with ORDER= so that array element order (1,4,2,5,3,6 here)
  ! differs from the constructor's sequence, proving the flattening
  ! order, plus a trailing element wholly beyond SOURCE
  integer(4), parameter :: r2(2, 3) = reshape([1_4, 2_4, 3_4, 4_4, 5_4, 6_4], [2, 3], order=[2, 1])
  integer(8), parameter :: via8d(4) = transfer(r2, 0_8, 4)
  logical, parameter :: test_rt_rank2b = all(transfer(via8d, 0_4, 8) == [1_4, 4_4, 2_4, 5_4, 3_4, 6_4, 0_4, 0_4])

  ! The standard's own Case (ii) example (16.9.212 p.6): the second
  ! element's real part has the value 3.3; its imaginary part is
  ! processor dependent
  complex, parameter :: cx(2) = transfer([1.1, 2.2, 3.3], [(0.0, 0.0)])
  logical, parameter :: test_case_ii = cx(1) == (1.1, 2.2) .and. real(cx(2)) == 3.3

  ! Derived-type MOLD longer than SOURCE: the leading part is preserved
  ! (observed portably via round trips); components at or beyond the end
  ! of SOURCE's representation are zero-filled
  type t1
    integer(8) :: a, b
  end type
  type(t1), parameter :: x1 = transfer([1_4, 2_4, 3_4], t1(0, 0)) ! b partially covered
  logical, parameter :: test_derived_rt = all(transfer(x1, 0_4, 3) == [1_4, 2_4, 3_4])
  type(t1), parameter :: x2 = transfer(7_4, t1(-1, -1)) ! a partial, b wholly beyond
  logical, parameter :: test_derived_lead = transfer(x2, 0_4) == 7_4
  logical, parameter :: test_derived_zero = x2%b == 0_8
  type(t1), parameter :: x4 = transfer(1_8, t1(-1, -1)) ! b exactly at the end
  logical, parameter :: test_at_end = x4%a == 1_8 .and. x4%b == 0_8
  type t2
    integer(4) :: x
    integer(4) :: y ! keeps c beyond a 4-byte SOURCE even where integer(8) has 4-byte alignment
    integer(8) :: c(4) ! wholly beyond SOURCE's representation
  end type
  type(t2), parameter :: x3 = transfer(9_4, t2(0, 0, [0_8, 0_8, 0_8, 0_8]))
  logical, parameter :: test_beyond = x3%x == 9_4 .and. x3%y == 0_4 .and. all(x3%c == 0_8)

  ! CHARACTER MOLD with elements beyond SOURCE: NUL fill
  character(1), parameter :: ch(50) = transfer(1_8, 'x', 50)
  logical, parameter :: test_char_rt = transfer(ch(1:8), 0_8) == 1_8
  logical, parameter :: test_char_zero = ichar(ch(9)) == 0 .and. ichar(ch(50)) == 0
  character(8), parameter :: c8 = transfer('AB', 'xxxxxxxx')
  logical, parameter :: test_char_scalar = c8(1:2) == 'AB' .and. ichar(c8(3:3)) == 0 .and. ichar(c8(8:8)) == 0
end module
