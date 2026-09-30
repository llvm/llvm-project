! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s \
! RUN:   | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s \
! RUN:   | FileCheck %s

! Range selection follows resolved source occurrences, even when 0*i folds.
! The enclosing i is distinct from the iterator i in the first clause.

subroutine depend_folded(a, m, d, i)
  integer :: a(2), m, d, i
  !$omp task depend(iterator(i = 1:m, j = 1:2), &
  !$omp& in: a(j/d + 0*i), a(1 + 0*i), a(j), a(1))
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPdepend_folded(
! CHECK: omp.iterator(%{{.*}}: index, %{{.*}}: index) =
! CHECK: arith.divsi
! CHECK: omp.yield
! CHECK: omp.iterator(%{{[^,:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.iterator(%{{[^,:]+}}: index) =
! CHECK: omp.yield
! CHECK-NOT: omp.iterator
! CHECK: omp.task
! CHECK: return

! The host i in the second clause must not be confused with iterator j.
subroutine depend_shadowed(a, b, m, i)
  integer :: a(2), b(2), m, i
  !$omp task depend(iterator(i = 1:m), in: a(1 + 0*i)) &
  !$omp& depend(iterator(j = 1:m), in: b(1 + 0*i))
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPdepend_shadowed(
! CHECK: omp.iterator(%{{[^,:]+}}: index) =
! CHECK: omp.yield
! CHECK-NOT: omp.iterator
! CHECK: omp.task
! CHECK: return

subroutine affinity_folded(a, m, d, i)
  integer :: a(2), m, d, i
  !$omp task affinity(iterator(i = 1:m, j = 1:2): &
  !$omp& a(j/d + 0*i), a(1 + 0*i), a(j), a(1))
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPaffinity_folded(
! CHECK: omp.iterator(%{{.*}}: index, %{{.*}}: index) =
! CHECK: arith.divsi
! CHECK: omp.yield
! CHECK: omp.iterator(%{{[^,:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.iterator(%{{[^,:]+}}: index) =
! CHECK: omp.yield
! CHECK-NOT: omp.iterator
! CHECK: omp.task
! CHECK: return

! The host i in the second task must not be confused with iterator j.
! OpenMP 5.2 permits only one affinity clause per task directive.
subroutine affinity_shadowed(a, b, m, i)
  integer :: a(2), b(2), m, i
  !$omp task affinity(iterator(i = 1:m): a(1 + 0*i))
  !$omp end task
  !$omp task affinity(iterator(j = 1:m): b(1 + 0*i))
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPaffinity_shadowed(
! CHECK: omp.iterator(%{{[^,:]+}}: index) =
! CHECK: omp.yield
! CHECK-NOT: omp.iterator
! CHECK: omp.task
! CHECK-NOT: omp.iterator
! CHECK: omp.task
! CHECK-NOT: omp.iterator
! CHECK: return

! These locators have equal folded designators but distinct source references.
! The last one references no iterator; the middle one references both even
! though all of its iterator references disappear during folding.
subroutine depend_equal_designators(a, m)
  integer :: a(2), m
  !$omp task depend(iterator(i=1:m, j=1:2), in: a(1+0*j), a(1+0*j+0*i), a(1))
  !$omp end task
end
! CHECK-LABEL: func.func @_QPdepend_equal_designators(
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.iterator(%{{[^:]+}}: index, %{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK-NOT: omp.iterator
! CHECK: omp.task depend(

subroutine affinity_equal_designators(a, m)
  integer :: a(2), m
  !$omp task affinity(iterator(i=1:m, j=1:2): a(1+0*j), a(1+0*j+0*i), a(1))
  !$omp end task
end
! CHECK-LABEL: func.func @_QPaffinity_equal_designators(
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.iterator(%{{[^:]+}}: index, %{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK-NOT: omp.iterator
! CHECK: omp.task affinity(

! Identically spelled iterator names in different clauses have distinct scopes.
subroutine depend_scoped_references(a, m)
  integer :: a(2), m
  !$omp task depend(iterator(i=1:m), in: a(1+0*i)) &
  !$omp& depend(iterator(i=1:2), out: a(i))
  !$omp end task
end
! CHECK-LABEL: func.func @_QPdepend_scoped_references(
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.task depend(

! Per-locator source metadata must also survive target clause decomposition.
subroutine target_folded_reference(a, m)
  integer :: a(2), m
  !$omp target parallel do map(tofrom:a) depend(iterator(i=1:m), in: a(1+0*i))
  do k=1,2
    a(k)=k
  end do
  !$omp end target parallel do
  !$omp target enter data map(to:a) depend(iterator(i=1:m), in: a(1+0*i))
  !$omp target update to(a) depend(iterator(i=1:m), in: a(1+0*i))
  !$omp target exit data map(from:a) depend(iterator(i=1:m), in: a(1+0*i))
end
! CHECK-LABEL: func.func @_QPtarget_folded_reference(
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.target
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.target_enter_data
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.target_update
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.yield
! CHECK: omp.target_exit_data
