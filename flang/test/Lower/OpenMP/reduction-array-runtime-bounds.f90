! RUN: split-file %s %t
! RUN: bbc -emit-hlfir -fopenmp -o - %t/whole.f90 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %t/whole.f90 | FileCheck %s
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %t/partial.f90 2>&1 | FileCheck %s --check-prefix=TODO
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %t/partial.f90 2>&1 | FileCheck %s --check-prefix=TODO
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %t/other-array.f90 2>&1 | FileCheck %s --check-prefix=TODO
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %t/other-array.f90 2>&1 | FileCheck %s --check-prefix=TODO
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %t/other-dimension.f90 2>&1 | FileCheck %s --check-prefix=TODO
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %t/other-dimension.f90 2>&1 | FileCheck %s --check-prefix=TODO
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %t/mutable.f90 2>&1 | FileCheck %s --check-prefix=TODO
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %t/mutable.f90 2>&1 | FileCheck %s --check-prefix=TODO
! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %t/stride.f90 2>&1 | FileCheck %s --check-prefix=TODO
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %t/stride.f90 2>&1 | FileCheck %s --check-prefix=TODO

! TODO: not yet implemented: REDUCTION of a partial array section using a user-defined reduction

!--- whole.f90
! The caller must provide nonempty reduction sections. No compile-time proof
! of a nonzero extent is needed to recognize their bounds.

! CHECK-LABEL: func.func @_QPexplicit_bounds
! CHECK: omp.parallel reduction(byref @_QQFexplicit_boundsmyred_byref_box_Uxi32 {{.*}} -> %[[EXPLICIT:arg[0-9]+]]
! CHECK: %[[EXPLICIT_DECL:.*]]:2 = hlfir.declare %[[EXPLICIT]]
! CHECK: %[[EXPLICIT_BOX:.*]] = fir.load %[[EXPLICIT_DECL]]#0
! CHECK: hlfir.designate %[[EXPLICIT_BOX]]
subroutine explicit_bounds(a, n)
  integer, intent(in) :: n
  integer :: a(-2:n)
  !$omp declare reduction(myred: integer: omp_out=omp_out+omp_in) initializer(omp_priv=0)
  !$omp parallel reduction(myred: a(-2:n))
  a(-2:n) = a(-2:n) + 1
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPassumed_bounds
! CHECK: omp.parallel reduction(byref @_QQFassumed_boundsmyred_byref_box_Uxi32 {{.*}} -> %[[ASSUMED:arg[0-9]+]]
! CHECK: %[[ASSUMED_DECL:.*]]:2 = hlfir.declare %[[ASSUMED]]
! CHECK: %[[ASSUMED_BOX:.*]] = fir.load %[[ASSUMED_DECL]]#0
! CHECK: hlfir.designate %[[ASSUMED_BOX]]
subroutine assumed_bounds(a)
  integer :: a(-2:)
  !$omp declare reduction(myred: integer: omp_out=omp_out+omp_in) initializer(omp_priv=0)
  !$omp parallel reduction(myred: a(lbound(a,1):ubound(a,1)))
  a(:) = a(:) + 1
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPsize_bound
! CHECK: omp.parallel reduction(byref @_QQFsize_boundmyred_byref_box_Uxi32 {{.*}} -> %[[SIZE:arg[0-9]+]]
! CHECK: %[[SIZE_DECL:.*]]:2 = hlfir.declare %[[SIZE]]
! CHECK: %[[SIZE_BOX:.*]] = fir.load %[[SIZE_DECL]]#0
! CHECK: hlfir.designate %[[SIZE_BOX]]
subroutine size_bound(a)
  integer :: a(:)
  !$omp declare reduction(myred: integer: omp_out=omp_out+omp_in) initializer(omp_priv=0)
  !$omp parallel reduction(myred: a(1:size(a)))
  a(:) = a(:) + 1
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPrank_two_bounds
! CHECK: omp.parallel reduction(byref @_QQFrank_two_boundsmyred_byref_box_UxUxi32 {{.*}} -> %[[RANK_TWO:arg[0-9]+]]
! CHECK: %[[RANK_TWO_DECL:.*]]:2 = hlfir.declare %[[RANK_TWO]]
! CHECK: %[[RANK_TWO_BOX:.*]] = fir.load %[[RANK_TWO_DECL]]#0
! CHECK: hlfir.designate %[[RANK_TWO_BOX]]
subroutine rank_two_bounds(a)
  integer :: a(-2:,-3:)
  !$omp declare reduction(myred: integer: omp_out=omp_out+omp_in) initializer(omp_priv=0)
  !$omp parallel reduction(myred: a(-2:ubound(a,1),lbound(a,2,kind=8):))
  a(:,:) = a(:,:) + 1
  !$omp end parallel
end subroutine

!--- partial.f90
subroutine partial(a, n)
  integer, intent(in) :: n
  integer :: a(-2:n)
  !$omp declare reduction(myred: integer: omp_out=omp_out+omp_in) initializer(omp_priv=0)
  !$omp parallel reduction(myred: a(-1:n))
  a(-1:n) = a(-1:n) + 1
  !$omp end parallel
end subroutine

!--- other-array.f90
subroutine other_array(a, b)
  integer :: a(-2:), b(-2:)
  !$omp declare reduction(myred: integer: omp_out=omp_out+omp_in) initializer(omp_priv=0)
  !$omp parallel reduction(myred: a(lbound(a,1):ubound(b,1)))
  a(lbound(a,1):ubound(b,1)) = a(lbound(a,1):ubound(b,1)) + 1
  !$omp end parallel
end subroutine

!--- other-dimension.f90
subroutine other_dimension(a)
  integer :: a(-2:,-3:)
  !$omp declare reduction(myred: integer: omp_out=omp_out+omp_in) initializer(omp_priv=0)
  !$omp parallel reduction(myred: a(-2:ubound(a,2),:))
  a(-2:ubound(a,2),:) = a(-2:ubound(a,2),:) + 1
  !$omp end parallel
end subroutine

!--- mutable.f90
subroutine mutable(a, n)
  integer :: n
  integer :: a(-2:n)
  !$omp declare reduction(myred: integer: omp_out=omp_out+omp_in) initializer(omp_priv=0)
  n = n - 1
  !$omp parallel reduction(myred: a(-2:n))
  a(-2:n) = a(-2:n) + 1
  !$omp end parallel
end subroutine

!--- stride.f90
subroutine stride(a, n)
  integer, intent(in) :: n
  integer :: a(-2:n)
  !$omp declare reduction(myred: integer: omp_out=omp_out+omp_in) initializer(omp_priv=0)
  !$omp parallel reduction(myred: a(-2:n:2))
  a(-2:n:2) = a(-2:n:2) + 1
  !$omp end parallel
end subroutine
