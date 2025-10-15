! RUN: not %flang_fc1 -fopenmp -fsyntax-only %s 2>&1 | FileCheck %s
type t
end type
type(t), pointer :: a1, a2
!$omp atomic write
a1 = a2
! CHECK: error: ATOMIC operation requires an intrinsic scalar variable; 'a1' has the POINTER attribute and derived type 't'
end
