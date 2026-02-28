! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s

subroutine omp_task_affinity_elem()
  implicit none
  integer, parameter :: n = 100
  integer :: a(n)

  !$omp parallel
  !$omp task affinity(a(1))
    a(1) = 1
  !$omp end task
  !$omp end parallel
end subroutine omp_task_affinity_elem

! CHECK-LABEL: func.func @_QPomp_task_affinity_elem()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFomp_task_affinity_elemEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK: omp.parallel {
! CHECK:   %[[C1:.*]] = arith.constant 1 : index
! CHECK:   %[[ELEM:.*]] = hlfir.designate %[[A]]#0 (%[[C1]]) : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:   %[[C0:.*]] = arith.constant 0 : index
! CHECK:   %[[ONE:.*]] = arith.constant 1 : index
! CHECK:   %[[SUB:.*]] = arith.subi %[[C0]], %[[C0]] : index
! CHECK:   %[[MUL:.*]] = arith.muli %[[SUB]], %[[ONE]] : index
! CHECK:   %[[ADD:.*]] = arith.addi %[[ONE]], %[[MUL]] : index
! CHECK:   %[[CAST:.*]] = fir.convert %[[ADD]] : (index) -> i64
! CHECK:   %[[C4:.*]] = arith.constant 4 : i64
! CHECK:   %[[LEN:.*]] = arith.muli %[[CAST]], %[[C4]] : i64
! CHECK:   %[[ADDRI8:.*]] = fir.convert %[[ELEM]] : (!fir.ref<i32>) -> !fir.ref<i8>
! CHECK:   %[[ENTRY:.*]] = omp.affinity_entry %[[ADDRI8]], %[[LEN]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:   omp.task affinity(%[[ENTRY]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>) {

subroutine omp_task_affinity_array_section()
  implicit none
  integer, parameter :: n = 100
  integer :: a(n)
  integer :: i

  !$omp parallel
  !$omp task affinity(a(2:50)) private(i)
    do i = 2, 50
      a(i) = i
    end do
  !$omp end task
  !$omp end parallel
end subroutine omp_task_affinity_array_section

! CHECK-LABEL: func.func @_QPomp_task_affinity_array_section()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFomp_task_affinity_array_sectionEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK: %[[I:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFomp_task_affinity_array_sectionEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: omp.parallel {
! CHECK:   %[[C2:.*]] = arith.constant 2 : index
! CHECK:   %[[C50:.*]] = arith.constant 50 : index
! CHECK:   %[[C1:.*]] = arith.constant 1 : index
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[SLICE:.*]] = hlfir.designate %[[A]]#0 (%[[C2]]:%[[C50]]:%[[C1]])  shape %[[SHAPE]] : (!fir.ref<!fir.array<100xi32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<49xi32>>
! CHECK:   %[[SPAN_I64:.*]] = fir.convert {{.*}} : (index) -> i64
! CHECK:   %[[C4:.*]] = arith.constant 4 : i64
! CHECK:   %[[LEN:.*]] = arith.muli %[[SPAN_I64]], %[[C4]] : i64
! CHECK:   %[[ADDRI8:.*]] = fir.convert %[[SLICE]] : (!fir.ref<!fir.array<49xi32>>) -> !fir.ref<i8>
! CHECK:   %[[ENTRY:.*]] = omp.affinity_entry %[[ADDRI8]], %[[LEN]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:   omp.task affinity(%[[ENTRY]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>) private(@_QFomp_task_affinity_array_sectionEi_private_i32 %[[I]]#0 -> %{{.*}} : !fir.ref<i32>) {

subroutine omp_task_affinity_scalar()
  implicit none
  integer :: s
  s = 7

  !$omp parallel
  !$omp task affinity(s)
    s = s + 1
  !$omp end task
  !$omp end parallel
end subroutine omp_task_affinity_scalar

! CHECK-LABEL: func.func @_QPomp_task_affinity_scalar()
! CHECK: %[[S:.*]] = fir.alloca i32 {bindc_name = "s", uniq_name = "_QFomp_task_affinity_scalarEs"}
! CHECK: %[[SDECL:.*]]:2 = hlfir.declare %[[S]] {uniq_name = "_QFomp_task_affinity_scalarEs"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: hlfir.assign %{{.*}} to %[[SDECL]]#0 : i32, !fir.ref<i32>
! CHECK: omp.parallel {
! CHECK:     %[[LEN:.*]] = arith.constant 4 : i64
! CHECK:     %[[ADDRI8:.*]] = fir.convert %[[SDECL]]#0 : (!fir.ref<i32>) -> !fir.ref<i8>
! CHECK:     %[[ENTRY:.*]] = omp.affinity_entry %[[ADDRI8]], %[[LEN]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:     omp.task affinity(%[[ENTRY]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>) {

subroutine omp_task_affinity_multi()
  implicit none
  integer, parameter :: n = 100
  integer :: a(n), b(n)

  !$omp parallel
  !$omp task affinity(a(1), b(1))
    a(2) = 2
    b(2) = 2
  !$omp end task
  !$omp end parallel
end subroutine omp_task_affinity_multi

! CHECK-LABEL: func.func @_QPomp_task_affinity_multi()
! CHECK: omp.parallel {
! CHECK:     %[[AADDR:.*]] = fir.convert %{{.*}} : (!fir.ref<i32>) -> !fir.ref<i8>
! CHECK:     %[[AENT:.*]] = omp.affinity_entry %[[AADDR]], %{{.*}} : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:     %[[BADDR:.*]] = fir.convert %{{.*}} : (!fir.ref<i32>) -> !fir.ref<i8>
! CHECK:     %[[BENT:.*]] = omp.affinity_entry %[[BADDR]], %{{.*}} : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:     omp.task affinity(%[[AENT]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>, %[[BENT]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>) {

subroutine task_affinity_iterator_simple()
  integer, parameter :: n = 16
  integer :: a(n)
  integer :: i

  !$omp parallel
  !$omp single
  !$omp task affinity(iterator(i = 1:n) : a(i))
    a(i) = i
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtask_affinity_iterator_simple()
! CHECK: %[[ITERATED:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_I32:.*]] = fir.convert %[[IV]] : (index) -> i32
! CHECK:   %[[IV_I64:.*]] = fir.convert %[[IV_I32]] : (i32) -> i64
! CHECK:   %[[IV_IDX:.*]] = fir.convert %[[IV_I64]] : (i64) -> index
! CHECK:   %[[SHAPE:.*]] = fir.shape %c16 : (index) -> !fir.shape<1>
! CHECK:   %[[COOR:.*]] = fir.array_coor {{.*}}(%[[SHAPE]]) %[[IV_IDX]] : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   %[[C4:.*]] = arith.constant 4 : i64
! CHECK:   %[[ADDRI8:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !fir.ref<i8>
! CHECK:   %[[ENTRY:.*]] = omp.affinity_entry %[[ADDRI8]], %[[C4]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:   omp.yield(%[[ENTRY]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>)
! CHECK: } -> !omp.iterated<!omp.affinity_entry_ty<!fir.ref<i8>, i64>>
! CHECK: omp.task affinity(%{{.*}} : !omp.iterated<!omp.affinity_entry_ty<!fir.ref<i8>, i64>>) {

subroutine task_affinity_iterator_multi_dimension()
  integer, parameter :: n = 4, m = 6
  integer :: a(n, m)
  integer :: i, j

  !$omp parallel
  !$omp single
  !$omp task affinity(iterator(i = 3:n, j = 1:m) : a(i, j)) shared(a)
    do i = 1, n
      do j = 1, m
        a(i, j) = 100*i + j
      end do
    end do
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtask_affinity_iterator_multi_dimension()
! CHECK: %[[ITER:.*]] = omp.iterator(%[[IV0:.*]]: index, %[[IV1:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}, {{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV0_I32:.*]] = fir.convert %[[IV0]] : (index) -> i32
! CHECK:   %[[IV1_I32:.*]] = fir.convert %[[IV1]] : (index) -> i32
! CHECK:   %[[IV0_I64:.*]] = fir.convert %[[IV0_I32]] : (i32) -> i64
! CHECK:   %[[IV0_IDX:.*]] = fir.convert %[[IV0_I64]] : (i64) -> index
! CHECK:   %[[IV1_I64:.*]] = fir.convert %[[IV1_I32]] : (i32) -> i64
! CHECK:   %[[IV1_IDX:.*]] = fir.convert %[[IV1_I64]] : (i64) -> index
! CHECK:   %[[SHAPE:.*]] = fir.shape %c4, %c6 : (index, index) -> !fir.shape<2>
! CHECK:   %[[COOR:.*]] = fir.array_coor {{.*}}(%[[SHAPE]]) %[[IV0_IDX]], %[[IV1_IDX]] : (!fir.ref<!fir.array<4x6xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
! CHECK:   %[[C4:.*]] = arith.constant 4 : i64
! CHECK:   %[[ADDRI8:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !fir.ref<i8>
! CHECK:   %[[ENTRY:.*]] = omp.affinity_entry %[[ADDRI8]], %[[C4]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:   omp.yield(%[[ENTRY]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>)
! CHECK: } -> !omp.iterated<!omp.affinity_entry_ty<!fir.ref<i8>, i64>>
! CHECK: omp.task affinity(%[[ITER]] : !omp.iterated<!omp.affinity_entry_ty<!fir.ref<i8>, i64>>)

subroutine task_affinity_slice_2d
  integer, parameter :: n = 5
  integer, parameter :: m = 7
  integer :: a(n, m)
  integer :: i, j

  !$omp parallel
  !$omp single
  !$omp task affinity(a(2:4, 3:5))
    do i = 1, n
      do j = 1, m
        a(i, j) = i + j
      end do
    end do
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtask_affinity_slice_2d()
! CHECK: omp.parallel {
! CHECK:   omp.single {
! CHECK:     %[[BOX:.*]] = hlfir.designate {{.*}} : (!fir.ref<!fir.array<5x7xi32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<3x3xi32>>
! CHECK:     %[[BASE:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.array<3x3xi32>>) -> !fir.ref<!fir.array<3x3xi32>>
! CHECK:     %[[SPANI64:.*]] = fir.convert {{.*}} : (index) -> i64
! CHECK:     %[[C4:.*]] = arith.constant 4 : i64
! CHECK:     %[[LEN:.*]] = arith.muli %[[SPANI64]], %[[C4]] : i64
! CHECK:     %[[ADDRI8:.*]] = fir.convert %[[BASE]] : (!fir.ref<!fir.array<3x3xi32>>) -> !fir.ref<i8>
! CHECK:     %[[ENTRY:.*]] = omp.affinity_entry %[[ADDRI8]], %[[LEN]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:     omp.task affinity(%[[ENTRY]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>){{.*}} {

subroutine task_affinity_iterator_reordered()
  integer, parameter :: n = 4, m = 6
  integer :: a(n, m)
  integer :: i, j

  !$omp parallel
  !$omp single
  !$omp task affinity(iterator(i = 1:n, j = 1:m) : a(j, i)) shared(a)
    a(1, 1) = 1
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtask_affinity_iterator_reordered()
! CHECK: %[[ITER:.*]] = omp.iterator(%[[IV0:.*]]: index, %[[IV1:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}, {{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[RO_IV0_I32:.*]] = fir.convert %[[IV0]] : (index) -> i32
! CHECK:   %[[RO_IV1_I32:.*]] = fir.convert %[[IV1]] : (index) -> i32
! CHECK:   %[[RO_IV1_I64:.*]] = fir.convert %[[RO_IV1_I32]] : (i32) -> i64
! CHECK:   %[[RO_IV1_IDX:.*]] = fir.convert %[[RO_IV1_I64]] : (i64) -> index
! CHECK:   %[[RO_IV0_I64:.*]] = fir.convert %[[RO_IV0_I32]] : (i32) -> i64
! CHECK:   %[[RO_IV0_IDX:.*]] = fir.convert %[[RO_IV0_I64]] : (i64) -> index
! CHECK:   %[[SHAPE:.*]] = fir.shape %c4, %c6 : (index, index) -> !fir.shape<2>
! CHECK:   %[[COOR:.*]] = fir.array_coor {{.*}}(%[[SHAPE]]) %[[RO_IV1_IDX]], %[[RO_IV0_IDX]] : (!fir.ref<!fir.array<4x6xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>

subroutine task_affinity_iterator_expr_subscript()
  integer, parameter :: n = 5, m = 6
  integer :: a(n, m)
  integer :: i, j

  !$omp parallel
  !$omp single
  !$omp task affinity(iterator(i = 1:n-1, j = 1:m) : a(i+1, j)) shared(a)
    a(1, 1) = 1
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtask_affinity_iterator_expr_subscript()
! CHECK: %[[ITER2:.*]] = omp.iterator(%[[IVA:.*]]: index, %[[IVB:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}, {{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IVA_I32:.*]] = fir.convert %[[IVA]] : (index) -> i32
! CHECK:   %[[IVB_I32:.*]] = fir.convert %[[IVB]] : (index) -> i32
! CHECK:   %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:   %[[IP1_I32:.*]] = arith.addi %[[IVA_I32]], %[[C1_I32]] : i32
! CHECK:   %[[IP1_I64:.*]] = fir.convert %[[IP1_I32]] : (i32) -> i64
! CHECK:   %[[IP1_IDX:.*]] = fir.convert %[[IP1_I64]] : (i64) -> index
! CHECK:   %[[IVB_I64:.*]] = fir.convert %[[IVB_I32]] : (i32) -> i64
! CHECK:   %[[IVB_IDX:.*]] = fir.convert %[[IVB_I64]] : (i64) -> index
! CHECK:   %[[SHAPE2:.*]] = fir.shape %c5, %c6 : (index, index) -> !fir.shape<2>
! CHECK:   %[[COOR2:.*]] = fir.array_coor {{.*}}(%[[SHAPE2]]) %[[IP1_IDX]], %[[IVB_IDX]] : (!fir.ref<!fir.array<5x6xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>

subroutine task_affinity_iterator_section_subscript()
  integer, parameter :: n = 5, m = 6
  integer :: a(n, m)
  integer :: i, j

  !$omp parallel
  !$omp single
  !$omp task affinity(iterator(i = 1:n, j = 1:m) : a(i:i+1, j+2)) shared(a)
    a(1, 1) = 1
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtask_affinity_iterator_section_subscript()
! CHECK: %[[ITER3:.*]] = omp.iterator(%[[IVS0:.*]]: index, %[[IVS1:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}, {{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IVS0_I32:.*]] = fir.convert %[[IVS0]] : (index) -> i32
! CHECK:   %[[IVS1_I32:.*]] = fir.convert %[[IVS1]] : (index) -> i32
! CHECK:   %[[IVS0_I64:.*]] = fir.convert %[[IVS0_I32]] : (i32) -> i64
! CHECK:   %[[IVS0_IDX:.*]] = fir.convert %[[IVS0_I64]] : (i64) -> index
! CHECK:   %[[C2_I32:.*]] = arith.constant 2 : i32
! CHECK:   %[[JP2_I32:.*]] = arith.addi %[[IVS1_I32]], %[[C2_I32]] : i32
! CHECK:   %[[JP2_I64:.*]] = fir.convert %[[JP2_I32]] : (i32) -> i64
! CHECK:   %[[JP2_IDX:.*]] = fir.convert %[[JP2_I64]] : (i64) -> index
! CHECK:   %[[SHAPE3:.*]] = fir.shape %c5, %c6 : (index, index) -> !fir.shape<2>
! CHECK:   %[[COOR3:.*]] = fir.array_coor {{.*}}(%[[SHAPE3]]) %[[IVS0_IDX]], %[[JP2_IDX]] : (!fir.ref<!fir.array<5x6xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>

subroutine task_affinity_iterator_section_implicit_lower()
  integer, parameter :: n = 5, m = 6
  integer :: a(n, m)
  integer :: i, j

  !$omp parallel
  !$omp single
  !$omp task affinity(iterator(i = 1:n, j = 1:m) : a(:i+1, j+2)) shared(a)
    a(1, 1) = 1
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtask_affinity_iterator_section_implicit_lower()
! CHECK: %[[ITER4:.*]] = omp.iterator(%[[IVT0:.*]]: index, %[[IVT1:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}, {{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IVT1_I32:.*]] = fir.convert %[[IVT1]] : (index) -> i32
! CHECK:   %[[C1_IDX:.*]] = arith.constant 1 : index
! CHECK:   %[[C2_I32_2:.*]] = arith.constant 2 : i32
! CHECK:   %[[JP2_I32_2:.*]] = arith.addi %[[IVT1_I32]], %[[C2_I32_2]] : i32
! CHECK:   %[[JP2_I64_2:.*]] = fir.convert %[[JP2_I32_2]] : (i32) -> i64
! CHECK:   %[[JP2_IDX_2:.*]] = fir.convert %[[JP2_I64_2]] : (i64) -> index
! CHECK:   %[[SHAPE4:.*]] = fir.shape %c5, %c6 : (index, index) -> !fir.shape<2>
! CHECK:   %[[COOR4:.*]] = fir.array_coor {{.*}}(%[[SHAPE4]]) %[[C1_IDX]], %[[JP2_IDX_2]] : (!fir.ref<!fir.array<5x6xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>

subroutine task_affinity_iterator_char_simple()
  integer, parameter :: n = 8
  character(len=7) :: a(n)
  integer :: i

  !$omp parallel
  !$omp single
  !$omp task affinity(iterator(i = 1:n) : a(i))
    a(1) = "abcdefg"
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtask_affinity_iterator_char_simple()
! CHECK: %[[ITER5:.*]] = omp.iterator(%[[IVC:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IVC_I32:.*]] = fir.convert %[[IVC]] : (index) -> i32
! CHECK:   %[[IVC_I64:.*]] = fir.convert %[[IVC_I32]] : (i32) -> i64
! CHECK:   %[[IVC_IDX:.*]] = fir.convert %[[IVC_I64]] : (i64) -> index
! CHECK:   %[[SHAPE5:.*]] = fir.shape {{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[COOR5:.*]] = fir.array_coor {{.*}}(%[[SHAPE5]]) %[[IVC_IDX]] : ({{.*}}, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,7>>
! CHECK:   %[[C7_I64:.*]] = arith.constant 7 : i64
! CHECK:   %[[ADDR5:.*]] = fir.convert %[[COOR5]] : (!fir.ref<!fir.char<1,7>>) -> !fir.ref<i8>
! CHECK:   %[[ENTRY5:.*]] = omp.affinity_entry %[[ADDR5]], %[[C7_I64]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>

subroutine task_affinity_iterator_char_expr_subscript()
  integer, parameter :: n = 8
  character(len=7) :: a(n)
  integer :: i

  !$omp parallel
  !$omp single
  !$omp task affinity(iterator(i = 1:n-1) : a(i+1))
    a(1) = "abcdefg"
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtask_affinity_iterator_char_expr_subscript()
! CHECK: %[[ITER6:.*]] = omp.iterator(%[[IVC2:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IVC2_I32:.*]] = fir.convert %[[IVC2]] : (index) -> i32
! CHECK:   %[[C1_I32_6:.*]] = arith.constant 1 : i32
! CHECK:   %[[IP1C_I32:.*]] = arith.addi %[[IVC2_I32]], %[[C1_I32_6]] : i32
! CHECK:   %[[IP1C_I64:.*]] = fir.convert %[[IP1C_I32]] : (i32) -> i64
! CHECK:   %[[IP1C_IDX:.*]] = fir.convert %[[IP1C_I64]] : (i64) -> index
! CHECK:   %[[SHAPE6:.*]] = fir.shape {{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[COOR6:.*]] = fir.array_coor {{.*}}(%[[SHAPE6]]) %[[IP1C_IDX]] : ({{.*}}, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,7>>
! CHECK:   %[[C7_I64_2:.*]] = arith.constant 7 : i64
! CHECK:   %[[ADDR6:.*]] = fir.convert %[[COOR6]] : (!fir.ref<!fir.char<1,7>>) -> !fir.ref<i8>
! CHECK:   %[[ENTRY6:.*]] = omp.affinity_entry %[[ADDR6]], %[[C7_I64_2]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
