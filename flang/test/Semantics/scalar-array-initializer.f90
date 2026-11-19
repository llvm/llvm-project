! RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s
! A scalar initializer of an array variable of intrinsic numeric or logical
! type is kept as a scalar rather than expanded to one value per element.
! Named constants, character, derived types and array-valued initializers
! are still expanded.
module m
  real :: a(3) = 1.5
  logical :: l(2) = .true.
  integer, parameter :: p(2) = 4
  integer :: q(2) = p
  character(2) :: cs(2) = 'ab'
  type t
    real :: comp(2) = 6.0
  end type
end module

!CHECK: a, PUBLIC size=12 offset=0: ObjectEntity type: REAL(4) shape: 1_8:3_8 init:1.5_4
!CHECK: cs, PUBLIC size=4 offset={{[0-9]+}}: ObjectEntity type: CHARACTER(2_4,1) shape: 1_8:2_8 init:[CHARACTER(KIND=1,LEN=2)::"ab","ab"]
!CHECK: l, PUBLIC size=8 offset={{[0-9]+}}: ObjectEntity type: LOGICAL(4) shape: 1_8:2_8 init:.true._4
!CHECK: p, PARAMETER, PUBLIC size=8 offset={{[0-9]+}}: ObjectEntity type: INTEGER(4) shape: 1_8:2_8 init:[INTEGER(4)::4_4,4_4]
!CHECK: q, PUBLIC size=8 offset={{[0-9]+}}: ObjectEntity type: INTEGER(4) shape: 1_8:2_8 init:[INTEGER(4)::4_4,4_4]
!CHECK: comp size=8 offset=0: ObjectEntity type: REAL(4) shape: 1_8:2_8 init:[REAL(4)::6._4,6._4]
