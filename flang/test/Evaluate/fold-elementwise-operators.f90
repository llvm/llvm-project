! RUN: %python %S/test_folding.py %s %flang_fc1
! Exercise the operator state retained by elementwise folding callbacks,
! including scalar expansion on either side of a binary operation.
module m
  logical, parameter :: a(*) = [.true., .false., .true., .false.]
  logical, parameter :: b(*) = [.true., .true., .false., .false.]
  logical, parameter :: test_and = &
    all((a .and. b) .eqv. [.true., .false., .false., .false.])
  logical, parameter :: test_or = &
    all((a .or. b) .eqv. [.true., .true., .true., .false.])
  logical, parameter :: test_eqv = &
    all((a .eqv. b) .eqv. [.true., .false., .false., .true.])
  logical, parameter :: test_neqv = &
    all((a .neqv. b) .eqv. [.false., .true., .true., .false.])
  logical, parameter :: test_logical_right_scalar = &
    all((a .and. .true.) .eqv. a)
  logical, parameter :: test_logical_left_scalar = &
    all((.false. .or. a) .eqv. a)

  integer, parameter :: x(*) = [1, 2, 3], y(*) = [2, 2, 2]
  logical, parameter :: test_eq = all((x == y) .eqv. [.false., .true., .false.])
  logical, parameter :: test_ne = all((x /= y) .eqv. [.true., .false., .true.])
  logical, parameter :: test_lt = all((x < y) .eqv. [.true., .false., .false.])
  logical, parameter :: test_le = all((x <= y) .eqv. [.true., .true., .false.])
  logical, parameter :: test_gt = all((x > y) .eqv. [.false., .false., .true.])
  logical, parameter :: test_ge = all((x >= y) .eqv. [.false., .true., .true.])
  logical, parameter :: test_relation_right_scalar = &
    all((x < 2) .eqv. [.true., .false., .false.])
  logical, parameter :: test_relation_left_scalar = &
    all((2 < x) .eqv. [.false., .false., .true.])
  logical, parameter :: test_real_relation = &
    all(([1., 2., 3.] >= 2.) .eqv. [.false., .true., .true.])
  logical, parameter :: test_character_relation = &
    all((['ant', 'bee', 'cat'] < 'bee') .eqv. [.true., .false., .false.])
  logical, parameter :: test_complex_relation = &
    all(([(1., 2.), (2., 1.)] == (1., 2.)) .eqv. [.true., .false.])

  logical, parameter :: test_min = all(min(x, y) == [1, 2, 2])
  logical, parameter :: test_max = all(max(x, y) == [2, 2, 3])
  logical, parameter :: test_min_right_scalar = all(min(x, 2) == [1, 2, 2])
  logical, parameter :: test_max_left_scalar = all(max(2, x) == [2, 2, 3])
  logical, parameter :: test_real_min = &
    all(min([1., 2., 3.], 2.) == [1., 2., 2.])
  logical, parameter :: test_real_max = &
    all(max(2., [1., 2., 3.]) == [2., 2., 3.])
  logical, parameter :: test_character_min = &
    all(min(['ant', 'cat'], 'bee') == ['ant', 'bee'])
  logical, parameter :: test_character_max = &
    all(max('bee', ['ant', 'cat']) == ['bee', 'cat'])

  complex, parameter :: z(*) = [(1., 2.), (3., 4.)]
  logical, parameter :: test_re = all(z%re == [1., 3.])
  logical, parameter :: test_im = all(z%im == [2., 4.])
  logical, parameter :: test_real = all(real(z) == [1., 3.])
  logical, parameter :: test_aimag = all(aimag(z) == [2., 4.])
end module
