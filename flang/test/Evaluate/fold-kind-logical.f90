! RUN: %python %S/test_folding.py %s %flang_fc1
! Regression tests: constant folding must preserve non-default LOGICAL kinds.
module m
  ! Unary and binary logical operations have the kind of their operands
  ! (F2023 10.1.5.4.2); use same-kind operands only, since the kind of a
  ! mixed-kind logical operation is processor dependent.
  logical, parameter :: test_not1 = kind(.not. .true._1) == 1
  logical, parameter :: test_not2 = kind(.not. .true._2) == 2
  logical, parameter :: test_not8 = kind(.not. .true._8) == 8
  logical, parameter :: test_and8 = kind(.true._8 .and. .false._8) == 8
  logical, parameter :: test_or2 = kind(.true._2 .or. .false._2) == 2
  logical, parameter :: test_eqv8 = kind(.true._8 .eqv. .true._8) == 8
  logical, parameter :: test_neqv1 = kind(.true._1 .neqv. .false._1) == 1
  ! STORAGE_SIZE of a folded non-default-kind logical operation.
  logical, parameter :: test_ss8 = storage_size(.not. .true._8) == 64
  logical, parameter :: test_ss2 = storage_size(.true._2 .and. .true._2) == 16
  ! ALL/ANY/PARITY results have the same type and kind type parameters as
  ! MASK (F2023 16.9.14, 16.9.16, 16.9.148), including for empty masks,
  ! where there is no element value to take a kind from.
  logical, parameter :: test_any8 = kind(any([.true._8])) == 8
  logical, parameter :: test_all2 = kind(all([.false._2])) == 2
  logical, parameter :: test_parity8 = kind(parity([.true._8, .false._8])) == 8
  logical, parameter :: test_parity1 = kind(parity([.true._1])) == 1
  logical, parameter :: test_anyempty8 = kind(any([logical(8) ::])) == 8
  logical, parameter :: test_allempty2 = kind(all([logical(2) ::])) == 2
  ! DOT_PRODUCT of logical arrays is ANY(X.AND.Y) (F2023 16.9.79), so its
  ! kind is that of the operands.
  logical, parameter :: test_dot8 = kind(dot_product([.true._8], [.true._8])) == 8
  logical, parameter :: test_dot2 = kind(dot_product([.true._2], [.false._2])) == 2
  ! MERGE and TRANSFER results keep the kind of TSOURCE/MOLD.
  logical, parameter :: test_merge8 = kind(merge(.true._8, .false._8, .true.)) == 8
  logical, parameter :: test_transfer8 = kind(transfer(.true._8, .false._8)) == 8
  ! A relational operation yields default logical regardless of the kinds
  ! of its operands (F2023 10.1.5.5.1); 4 is the default logical kind under
  ! default compilation options.
  logical, parameter :: test_rel2 = kind(1_2 == 2_2) == 4
  ! The default BOUNDARY fill element of a folded EOSHIFT must have the
  ! array's kind. Probe the extracted fill element: a wrong-kind element
  ! can hide inside a constant whose overall type is correct.
  logical(8), parameter :: eo8(2) = eoshift([.true._8, .true._8], 1)
  logical, parameter :: test_eo8kind = storage_size(eo8(2)) == 64
  logical, parameter :: test_eo8value = eo8(1) .and. .not. eo8(2)
end module
