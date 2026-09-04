! RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s
! Ensure that integer conversions inserted by the frontend use __builtin_int.

subroutine sub0 (n, int )
  integer(kind=4) :: n
  !CHECK: Subprogram scope: sub0
  !CHECK:   int {{.*}}: ObjectEntity dummy type: INTEGER(4) shape: 1_8:__builtin_int(n,kind=8)
  integer(kind=4) :: int(n)
end subroutine
