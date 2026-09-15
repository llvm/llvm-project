! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o %t
! RUN: FileCheck %s < %t
! RUN: fir-opt --canonicalize --verify-each %t | FileCheck %s --check-prefix=CAN

! Fortran endpoints are inclusive, including singleton and descending ranges.
subroutine singleton(a)
  integer :: a(4)
  !$omp target update to(iterator(i=1:1): a(i))
end
! CHECK-LABEL: func.func @_QPsingleton
! CHECK: omp.iterator
! CHECK: } inclusive -> !omp.iterated<!llvm.ptr>
! CAN-LABEL: func.func @_QPsingleton
! CAN: %[[ONE:.*]] = arith.constant 1 : index
! CAN: omp.iterator(%{{.*}}: index) =
! CAN-SAME: (%[[ONE]] to %[[ONE]] step %[[ONE]])
! CAN: } inclusive -> !omp.iterated<!llvm.ptr>

subroutine ascending(a)
  integer :: a(4)
  !$omp target enter data map(iterator(i=1:4), to: a(i))
end
! CHECK-LABEL: func.func @_QPascending
! CHECK: omp.iterator
! CHECK: } inclusive -> !omp.iterated<!llvm.ptr>
! CAN-LABEL: func.func @_QPascending
! CAN-DAG: %[[ONE:.*]] = arith.constant 1 : index
! CAN-DAG: %[[FOUR:.*]] = arith.constant 4 : index
! CAN: omp.iterator(%{{.*}}: index) =
! CAN-SAME: (%[[ONE]] to %[[FOUR]] step %[[ONE]])
! CAN: } inclusive -> !omp.iterated<!llvm.ptr>

subroutine descending(a)
  integer :: a(4)
  !$omp target exit data map(iterator(i=4:1:-1), from: a(i))
end
! CHECK-LABEL: func.func @_QPdescending
! CHECK: omp.iterator
! CHECK: } inclusive -> !omp.iterated<!llvm.ptr>
! CAN-LABEL: func.func @_QPdescending
! CAN-DAG: %[[ONE:.*]] = arith.constant 1 : index
! CAN-DAG: %[[FOUR:.*]] = arith.constant 4 : index
! CAN-DAG: %[[NEG:.*]] = arith.constant -1 : index
! CAN: omp.iterator(%{{.*}}: index) =
! CAN-SAME: (%[[FOUR]] to %[[ONE]] step %[[NEG]])
! CAN: } inclusive -> !omp.iterated<!llvm.ptr>

! Empty ranges must survive canonicalization and verification. The containing
! directive and its unrelated map entry must remain present.
subroutine empty_positive(a, b)
  integer :: a(4), b
  !$omp target update to(iterator(i=2:1): a(i), b) nowait
end
! CHECK-LABEL: func.func @_QPempty_positive
! CHECK: omp.iterator
! CAN-LABEL: func.func @_QPempty_positive
! CAN-DAG: %[[ONE:.*]] = arith.constant 1 : index
! CAN-DAG: %[[TWO:.*]] = arith.constant 2 : index
! CAN: omp.iterator(%{{.*}}: index) =
! CAN-SAME: (%[[TWO]] to %[[ONE]] step %[[ONE]])
! CAN: } inclusive -> !omp.iterated<!llvm.ptr>
! CAN: omp.target_update
! CAN-SAME: map_entries(
! CAN-SAME: map_iterated(
! CAN-SAME: nowait

subroutine empty_negative(a)
  integer :: a(4)
  !$omp target update from(iterator(i=1:2:-1): a(i))
end
! CHECK-LABEL: func.func @_QPempty_negative
! CHECK: omp.iterator
! CAN-LABEL: func.func @_QPempty_negative
! CAN-DAG: %[[ONE:.*]] = arith.constant 1 : index
! CAN-DAG: %[[TWO:.*]] = arith.constant 2 : index
! CAN-DAG: %[[NEG:.*]] = arith.constant -1 : index
! CAN: omp.iterator(%{{.*}}: index) =
! CAN-SAME: (%[[ONE]] to %[[TWO]] step %[[NEG]])
! CAN: } inclusive -> !omp.iterated<!llvm.ptr>

subroutine dynamic_range(a, lo, hi, step)
  integer :: a(4), lo, hi, step
  !$omp target data map(iterator(i=lo:hi:step), to: a(i))
  !$omp end target data
end
! CHECK-LABEL: func.func @_QPdynamic_range
! CHECK: omp.iterator
! CAN-LABEL: func.func @_QPdynamic_range
! CAN: omp.iterator
! CAN: } inclusive -> !omp.iterated<!llvm.ptr>
! CAN: omp.target_data
! CAN-SAME: map_iterated(

! Each locator uses only its own dimensions. An unused empty range cannot
! suppress a(i), and unrelated range expressions are still evaluated.
subroutine separate_ranges(a, b, c, n)
  integer :: a(4), b(4), c(4, 4), n
  integer, external :: extent
  !$omp target update to(iterator(i=1:4, j=1:n, k=1:extent()): &
  !$omp& a(i), b(j), c(i,j))
end
! CHECK-LABEL: func.func @_QPseparate_ranges
! CHECK: fir.call @_QPextent()
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CHECK: omp.iterator(%{{[^:]+}}: index, %{{[^:]+}}: index) =
! CHECK: omp.target_update
! CAN-LABEL: func.func @_QPseparate_ranges
! CAN: fir.call @_QPextent()
! CAN: omp.iterator(%{{[^:]+}}: index) =
! CAN: omp.iterator(%{{[^:]+}}: index) =
! CAN: omp.iterator(%{{[^:]+}}: index, %{{[^:]+}}: index) =
! CAN: omp.target_update

subroutine unused_empty_range(a)
  integer :: a(4)
  !$omp target enter data map(iterator(i=1:4, j=1:0), to: a(i))
end
! CHECK-LABEL: func.func @_QPunused_empty_range
! CHECK: omp.iterator(%{{[^:]+}}: index) =
! CAN-LABEL: func.func @_QPunused_empty_range
! CAN-DAG: %[[ONE:.*]] = arith.constant 1 : index
! CAN-DAG: %[[FOUR:.*]] = arith.constant 4 : index
! CAN: omp.iterator(%{{[^:]+}}: index) =
! CAN-SAME: (%[[ONE]] to %[[FOUR]] step %[[ONE]])

! Values beyond the index width must reach the iterator and its body intact.
subroutine wide_range(a, hi, step)
  integer :: a(2)
  integer(16) :: hi, step
  !$omp target update to(iterator(integer(16) :: i=0_16:hi:step): &
  !$omp& a(int(i/step)+1))
end
! CHECK-LABEL: func.func @_QPwide_range
! CHECK: %[[HI:.*]] = fir.load {{.*}} : !fir.ref<i128>
! CHECK: %[[STEP:.*]] = fir.load {{.*}} : !fir.ref<i128>
! CHECK-NOT: fir.convert
! CHECK: omp.iterator(%[[IV:.*]]: i128) =
! CHECK-SAME: ({{.*}} to %[[HI]] step %[[STEP]])
! CHECK: fir.store %[[IV]] to {{.*}} : !fir.ref<i128>
! CHECK: arith.divsi {{.*}} : i128
! CHECK: } inclusive -> !omp.iterated<!llvm.ptr>
! CAN-LABEL: func.func @_QPwide_range
! CAN: omp.iterator(%{{.*}}: i128) =
! CAN: arith.divsi {{.*}} : i128
! CAN: } inclusive -> !omp.iterated<!llvm.ptr>

subroutine wide_default_step(a, hi)
  integer :: a(2)
  integer(16) :: hi
  !$omp target enter data map(iterator(integer(16) :: i=hi:hi), &
  !$omp& to: a(int(i-hi)+1))
end
! CHECK-LABEL: func.func @_QPwide_default_step
! CHECK: %[[ONE:.*]] = arith.constant 1 : i128
! CHECK: omp.iterator(%{{.*}}: i128) =
! CHECK-SAME: ({{.*}} to {{.*}} step %[[ONE]])
! CAN-LABEL: func.func @_QPwide_default_step
! CAN: omp.iterator(%{{.*}}: i128) =
! CAN: } inclusive -> !omp.iterated<!llvm.ptr>
