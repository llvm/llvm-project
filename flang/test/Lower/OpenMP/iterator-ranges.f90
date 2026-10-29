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
! CAN: %[[ONE:.*]] = arith.constant 1 : i32
! CAN: omp.iterator(%{{.*}}: i32) =
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
! CAN-DAG: %[[ONE:.*]] = arith.constant 1 : i32
! CAN-DAG: %[[FOUR:.*]] = arith.constant 4 : i32
! CAN: omp.iterator(%{{.*}}: i32) =
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
! CAN-DAG: %[[ONE:.*]] = arith.constant 1 : i32
! CAN-DAG: %[[FOUR:.*]] = arith.constant 4 : i32
! CAN-DAG: %[[NEG:.*]] = arith.constant -1 : i32
! CAN: omp.iterator(%{{.*}}: i32) =
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
! CAN-DAG: %[[ONE:.*]] = arith.constant 1 : i32
! CAN-DAG: %[[TWO:.*]] = arith.constant 2 : i32
! CAN: omp.iterator(%{{.*}}: i32) =
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
! CAN-DAG: %[[ONE:.*]] = arith.constant 1 : i32
! CAN-DAG: %[[TWO:.*]] = arith.constant 2 : i32
! CAN-DAG: %[[NEG:.*]] = arith.constant -1 : i32
! CAN: omp.iterator(%{{.*}}: i32) =
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
! CHECK-DAG: %[[FOUR:.*]] = arith.constant 4 : i32
! CHECK-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}}uniq_name("_QFseparate_rangesEa")
! CHECK-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}}uniq_name("_QFseparate_rangesEb")
! CHECK-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}}uniq_name("_QFseparate_rangesEc")
! CHECK-DAG: %[[N:.*]]:2 = hlfir.declare {{.*}}uniq_name("_QFseparate_rangesEn")
! CHECK: %[[N_VAL:.*]] = fir.load %[[N]]#0 : !fir.ref<i32>
! CHECK: fir.call @_QPextent()
! CHECK: omp.iterator(%{{[^:]+}}: i32) = (%{{[^ ]+}} to %[[FOUR]] step %{{[^)]+}}) {
! CHECK:   omp.map.info var_ptr(%[[A]]#0
! CHECK: omp.iterator(%{{[^:]+}}: i32) = (%{{[^ ]+}} to %[[N_VAL]] step %{{[^)]+}}) {
! CHECK:   omp.map.info var_ptr(%[[B]]#0
! CHECK: omp.iterator(%{{[^:]+}}: i32, %{{[^:]+}}: i32) =
! CHECK-SAME: (%{{[^ ]+}} to %[[FOUR]] step %{{[^,]+}}, %{{[^ ]+}} to %[[N_VAL]] step %{{[^)]+}}) {
! CHECK:   omp.map.info var_ptr(%[[C]]#0
! CHECK: omp.target_update
! CAN-LABEL: func.func @_QPseparate_ranges
! CAN-DAG: %[[FOUR:.*]] = arith.constant 4 : i32
! CAN-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}}uniq_name("_QFseparate_rangesEa")
! CAN-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}}uniq_name("_QFseparate_rangesEb")
! CAN-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}}uniq_name("_QFseparate_rangesEc")
! CAN-DAG: %[[N:.*]]:2 = hlfir.declare {{.*}}uniq_name("_QFseparate_rangesEn")
! CAN: %[[N_VAL:.*]] = fir.load %[[N]]#0 : !fir.ref<i32>
! CAN: fir.call @_QPextent()
! CAN: omp.iterator(%{{[^:]+}}: i32) = (%{{[^ ]+}} to %[[FOUR]] step %{{[^)]+}}) {
! CAN:   omp.map.info var_ptr(%[[A]]#0
! CAN: omp.iterator(%{{[^:]+}}: i32) = (%{{[^ ]+}} to %[[N_VAL]] step %{{[^)]+}}) {
! CAN:   omp.map.info var_ptr(%[[B]]#0
! CAN: omp.iterator(%{{[^:]+}}: i32, %{{[^:]+}}: i32) =
! CAN-SAME: (%{{[^ ]+}} to %[[FOUR]] step %{{[^,]+}}, %{{[^ ]+}} to %[[N_VAL]] step %{{[^)]+}}) {
! CAN:   omp.map.info var_ptr(%[[C]]#0
! CAN: omp.target_update

subroutine unused_empty_range(a)
  integer :: a(4)
  !$omp target enter data map(iterator(i=1:4, j=1:0), to: a(i))
end
! CHECK-LABEL: func.func @_QPunused_empty_range
! CHECK: omp.iterator(%{{[^:]+}}: i32) =
! CAN-LABEL: func.func @_QPunused_empty_range
! CAN-DAG: %[[ONE:.*]] = arith.constant 1 : i32
! CAN-DAG: %[[FOUR:.*]] = arith.constant 4 : i32
! CAN: omp.iterator(%{{[^:]+}}: i32) =
! CAN-SAME: (%[[ONE]] to %[[FOUR]] step %[[ONE]])

! Items of a clause without an iterator modifier still use ordinary maps.
subroutine other_clause(a, b)
  integer :: a(4), b(4)
  !$omp target update to(iterator(i=1:2): a(i)) from(b(1:2))
end
! CHECK-LABEL: func.func @_QPother_clause
! CHECK: %[[IT:.*]] = omp.iterator
! CHECK: %[[B:.*]] = omp.map.info {{.*}}map_clauses(from) {{.*}}name("b(1:2)")
! CHECK: omp.target_update map_entries(%[[B]] : {{.*}}) map_iterated(%[[IT]] :

! A reference removed by folding still selects its range, so an empty range
! removes the locator instead of mapping it unconditionally.
subroutine folded_reference(a, n)
  integer, parameter :: stride = 0
  integer :: a(10), n
  !$omp target update from(iterator(i=1:n): a(stride*i + 1))
  !$omp target update to(iterator(i=1:n): a(stride*i + 1))
  !$omp target enter data map(iterator(i=1:0), to: a(1 + 0*i))
end
! CHECK-LABEL: func.func @_QPfolded_reference
! CHECK: omp.iterator(%{{[^:]+}}: i32) =
! CHECK: omp.target_update map_iterated(
! CHECK: omp.iterator(%{{[^:]+}}: i32) =
! CHECK: omp.target_update map_iterated(
! CHECK: omp.iterator(%{{[^:]+}}: i32) =
! CHECK: omp.target_enter_data map_iterated(
! CAN-LABEL: func.func @_QPfolded_reference
! CAN: omp.target_update map_iterated(
! CAN: omp.target_update map_iterated(
! CAN: omp.target_enter_data map_iterated(

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
