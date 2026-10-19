! RUN: %flang_fc1 -fdebug-dump-symbols -fenumeration-type %s 2>&1 | FileCheck %s
! Regression test: the enumeration-type intrinsics INT/NEXT/PREVIOUS are
! elemental, so a constant array argument must fold elementwise in a constant
! context.  Previously the enum fold path extracted only a scalar ordinal, so an
! array-valued call was left unfolded and a named constant reported "cannot be
! computed as a constant value".  Ordinals are 1-based, so
! [red, green, blue] -> [1, 2, 3].

module enum_array_fold_mod
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type

  ! Default-kind INT() over an array constructor folds to INTEGER(4) [1,2,3].
  integer, parameter :: a(3) = int([red, green, blue])
  !CHECK: a, PARAMETER, PUBLIC size={{[0-9]+}} offset={{[0-9]+}}: ObjectEntity type: INTEGER(4) shape: 1_8:3_8 init:[INTEGER(4)::1_4,2_4,3_4]

  ! KIND= is honored: folds to INTEGER(8) with the reversed order [3,2,1].
  integer(8), parameter :: b(3) = int([blue, green, red], kind=8)
  !CHECK: b, PARAMETER, PUBLIC size={{[0-9]+}} offset={{[0-9]+}}: ObjectEntity type: INTEGER(8) shape: 1_8:3_8 init:[INTEGER(8)::3_8,2_8,1_8]

  ! NEXT() over a (non-boundary) constant array folds to [green, blue].
  type(color), parameter :: cn(2) = next([red, green])
  !CHECK: cn, PARAMETER, PUBLIC size={{[0-9]+}} offset={{[0-9]+}}: ObjectEntity type: TYPE(color) shape: 1_8:2_8 init:[color::color(2_4),color(3_4)]

  ! PREVIOUS() over a (non-boundary) constant array folds to [red, green].
  type(color), parameter :: cp(2) = previous([green, blue])
  !CHECK: cp, PARAMETER, PUBLIC size={{[0-9]+}} offset={{[0-9]+}}: ObjectEntity type: TYPE(color) shape: 1_8:2_8 init:[color::color(1_4),color(2_4)]
end module
