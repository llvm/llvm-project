! RUN: %python %S/test_folding.py %s %flang_fc1
! Regression tests: empty typed array constructors and zero-size
! transformational results must preserve the kind (and character length)
! of their type. An array constructor with a type-spec has the type and
! type parameters of the type-spec (F2023 7.8), even when it has no
! element values to take them from.
module m
  ! Empty typed array constructors, each intrinsic category.
  logical, parameter :: test_ec4 = kind([character(kind=4, len=0) ::]) == 4
  logical, parameter :: test_ec2 = kind([character(kind=2, len=0) ::]) == 2
  logical, parameter :: test_ec4len = len([character(kind=4, len=5) ::]) == 5
  logical, parameter :: test_ec4size = size([character(kind=4, len=0) ::]) == 0
  logical, parameter :: test_ei2 = kind([integer(2) ::]) == 2
  logical, parameter :: test_er2 = kind([real(2) ::]) == 2
  logical, parameter :: test_el8 = kind([logical(8) ::]) == 8
  ! Empty typed implied-do array constructor.
  logical, parameter :: test_eido = kind([integer(1) :: (int(i, 1), i = 1, 0)]) == 1
  logical, parameter :: test_eidosize = size([integer(1) :: (int(i, 1), i = 1, 0)]) == 0
  ! Zero-size transformational results keep the source's kind.
  logical, parameter :: test_pack2 = kind(pack([1_2, 2_2], [.false., .false.])) == 2
  logical, parameter :: test_pack2size = size(pack([1_2, 2_2], [.false., .false.])) == 0
  logical, parameter :: test_pack4c = kind(pack([4_"ab"], [.false.])) == 4
  logical, parameter :: test_pack4clen = len(pack([4_"ab"], [.false.])) == 2
  logical, parameter :: test_spread2 = kind(spread(1_2, 1, 0)) == 2
  logical, parameter :: test_reshape8 = kind(reshape([integer(8) ::], [0])) == 8
  logical, parameter :: test_reshape2r = kind(reshape([real(2) ::], [0])) == 2
  logical, parameter :: test_unpack2 = kind(unpack([integer(2) ::], [logical ::], 0_2)) == 2
end module
