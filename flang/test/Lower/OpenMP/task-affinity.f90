! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s

! Non-iterator tests

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
! CHECK:   %[[C4:.*]] = arith.constant 4 : i64
! CHECK:   %[[ONE:.*]] = arith.constant 1 : index
! CHECK:   %[[SUB:.*]] = arith.subi %[[C0]], %[[C0]] : index
! CHECK:   %[[MUL:.*]] = arith.muli %[[SUB]], %[[ONE]] : index
! CHECK:   %[[ADD:.*]] = arith.addi %[[ONE]], %[[MUL]] : index
! CHECK:   %[[CAST:.*]] = fir.convert %[[ADD]] : (index) -> i64
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
! CHECK:   %[[C4:.*]] = arith.constant 4 : i64
! CHECK:   %[[SPAN_I64:.*]] = fir.convert {{.*}} : (index) -> i64
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

subroutine whole_array_affinity()
  implicit none
  integer :: a(10)

  !$omp task affinity(a)
    a(1) = 1
  !$omp end task
end subroutine whole_array_affinity

! CHECK-LABEL: func.func @_QPwhole_array_affinity()
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFwhole_array_affinityEa"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK: %[[C4:.*]] = arith.constant 4 : i64
! CHECK: %[[LEN:.*]] = arith.muli %{{.*}}, %[[C4]] : i64
! CHECK: %[[ADDRI8:.*]] = fir.convert %[[A]]#0 : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<i8>
! CHECK: %[[ENTRY:.*]] = omp.affinity_entry %[[ADDRI8]], %[[LEN]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>

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
! CHECK:     %[[C4:.*]] = arith.constant 4 : i64
! CHECK:     %[[SPANI64:.*]] = fir.convert {{.*}} : (index) -> i64
! CHECK:     %[[LEN:.*]] = arith.muli %[[SPANI64]], %[[C4]] : i64
! CHECK:     %[[ADDRI8:.*]] = fir.convert %[[BASE]] : (!fir.ref<!fir.array<3x3xi32>>) -> !fir.ref<i8>
! CHECK:     %[[ENTRY:.*]] = omp.affinity_entry %[[ADDRI8]], %[[LEN]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:     omp.task affinity(%[[ENTRY]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>){{.*}} {

subroutine assumed_shape_affinity(a)
  integer, intent(inout) :: a(:)

  !$omp task affinity(a)
    a(1) = 1
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPassumed_shape_affinity(
! CHECK: %[[A:.*]]:2 = hlfir.declare %arg0 dummy_scope %{{.*}} arg 1 {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFassumed_shape_affinityEa"}
! CHECK: %[[ELEM:.*]] = fir.box_elesize %[[A]]#0 : (!fir.box<!fir.array<?xi32>>) -> index
! CHECK: %[[ELEM_I64:.*]] = fir.convert %[[ELEM]] : (index) -> i64
! CHECK: %[[LEN:.*]] = arith.muli %{{.*}}, %[[ELEM_I64]] : i64
! CHECK: %[[ENTRY:.*]] = omp.affinity_entry %{{.*}}, %[[LEN]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>

subroutine allocatable_affinity(n)
  integer, intent(in) :: n
  integer, allocatable :: a(:)

  allocate(a(n))
  !$omp task affinity(a)
    a(1) = 1
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPallocatable_affinity(
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFallocatable_affinityEa"}
! CHECK: %[[ABOX:.*]] = fir.load %[[A]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[ELEM:.*]] = fir.box_elesize %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> index
! CHECK: %[[ELEM_I64:.*]] = fir.convert %[[ELEM]] : (index) -> i64
! CHECK: %[[LEN:.*]] = arith.muli %{{.*}}, %[[ELEM_I64]] : i64
! CHECK: %[[ENTRY:.*]] = omp.affinity_entry %{{.*}}, %[[LEN]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>

subroutine pointer_affinity(n)
  integer, intent(in) :: n
  integer, target :: tgt(n)
  integer, pointer :: p(:)

  p => tgt
  !$omp task affinity(p)
    p(1) = 1
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPpointer_affinity(
! CHECK: %[[P:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFpointer_affinityEp"}
! CHECK: %[[PBOX:.*]] = fir.load %[[P]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK: %[[ELEM:.*]] = fir.box_elesize %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> index
! CHECK: %[[ELEM_I64:.*]] = fir.convert %[[ELEM]] : (index) -> i64
! CHECK: %[[LEN:.*]] = arith.muli %{{.*}}, %[[ELEM_I64]] : i64
! CHECK: %[[ENTRY:.*]] = omp.affinity_entry %{{.*}}, %[[LEN]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>

subroutine char_const_len_affinity()
  character(len=7) :: a(8)

  !$omp task affinity(a)
    a(1) = "abcdefg"
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPchar_const_len_affinity()
! CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) typeparams %c7 {uniq_name = "_QFchar_const_len_affinityEa"}
! CHECK: %[[CHARLEN_I64:.*]] = fir.convert %c7 : (index) -> i64
! CHECK: %[[ELEMSIZE:.*]] = arith.muli %[[CHARLEN_I64]], %{{.*}} : i64
! CHECK: %[[LEN:.*]] = arith.muli %{{.*}}, %[[ELEMSIZE]] : i64
! CHECK: %[[ENTRY:.*]] = omp.affinity_entry %{{.*}}, %[[LEN]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>

subroutine char_runtime_len_affinity(n, l)
  integer, intent(in) :: n, l
  character(len=l) :: a(n)

  !$omp task affinity(a)
    a(1) = repeat("x", l)
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPchar_runtime_len_affinity(
! CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) typeparams %{{.*}} {uniq_name = "_QFchar_runtime_len_affinityEa"}
! CHECK: %[[ELEM:.*]] = fir.box_elesize %[[DECLARE]]#0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK: %[[ELEM_I64:.*]] = fir.convert %[[ELEM]] : (index) -> i64
! CHECK: %[[LEN:.*]] = arith.muli %{{.*}}, %[[ELEM_I64]] : i64
! CHECK: %[[ENTRY:.*]] = omp.affinity_entry %{{.*}}, %[[LEN]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>

module task_affinity_polymorphic_mod
  type :: t
    integer :: x
  end type
contains
  subroutine task_affinity_poly()
    class(t), allocatable :: a

    allocate(a)
    !$omp task affinity(a) shared(a)
      select type (a)
      type is (t)
        a%x = 1
      end select
    !$omp end task
  end subroutine
end module

! CHECK-LABEL: func.func @_QMtask_affinity_polymorphic_modPtask_affinity_poly()
! CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMtask_affinity_polymorphic_modFtask_affinity_polyEa"}
! CHECK: %[[ALOAD:.*]] = fir.load %[[DECLARE]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMtask_affinity_polymorphic_modTt{x:i32}>>>>
! CHECK: %[[ADDR:.*]] = fir.box_addr %[[ALOAD]] : (!fir.class<!fir.heap<!fir.type<_QMtask_affinity_polymorphic_modTt{x:i32}>>>) -> !fir.heap<!fir.type<_QMtask_affinity_polymorphic_modTt{x:i32}>>
! CHECK: %[[SIZELOAD:.*]] = fir.load %[[DECLARE]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMtask_affinity_polymorphic_modTt{x:i32}>>>>
! CHECK: %[[SIZE:.*]] = fir.box_elesize %[[SIZELOAD]] : (!fir.class<!fir.heap<!fir.type<_QMtask_affinity_polymorphic_modTt{x:i32}>>>) -> index
! CHECK: %[[SIZE_I64:.*]] = fir.convert %[[SIZE]] : (index) -> i64
! CHECK: %[[ADDR_I8:.*]] = fir.convert %[[ADDR]] : (!fir.heap<!fir.type<_QMtask_affinity_polymorphic_modTt{x:i32}>>) -> !fir.ref<i8>
! CHECK: %[[ENTRY:.*]] = omp.affinity_entry %[[ADDR_I8]], %[[SIZE_I64]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK: omp.task affinity(%[[ENTRY]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>) {

! Iterator tests

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
! CHECK:   %[[SHAPE:.*]] = fir.shape %c16 : (index) -> !fir.shape<1>
! CHECK:   %[[COOR:.*]] = fir.array_coor {{.*}}(%[[SHAPE]]) %[[IV_I64]] : (!fir.ref<!fir.array<16xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[C4:.*]] = arith.constant 4 : i64
! CHECK:   %[[ADDRI8:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !fir.ref<i8>
! CHECK:   %[[ENTRY:.*]] = omp.affinity_entry %[[ADDRI8]], %[[C4]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:   omp.yield(%[[ENTRY]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>)
! CHECK: } -> !omp.iterated<!omp.affinity_entry_ty<!fir.ref<i8>, i64>>
! CHECK: omp.task affinity(%{{.*}} : !omp.iterated<!omp.affinity_entry_ty<!fir.ref<i8>, i64>>) {

subroutine task_affinity_iterator_nondefault_lb()
  implicit none
  integer, parameter :: n = 8
  integer :: a(0:n)
  integer :: i

  !$omp parallel
  !$omp single
  !$omp task affinity(iterator(i = 0:n) : a(i))
    a(0) = 1
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtask_affinity_iterator_nondefault_lb()
! CHECK: %[[ITERATED_NDLB:.*]] = omp.iterator(%[[IV_NDLB:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV_NDLB_I32:.*]] = fir.convert %[[IV_NDLB]] : (index) -> i32
! CHECK:   %[[IV_NDLB_I64:.*]] = fir.convert %[[IV_NDLB_I32]] : (i32) -> i64
! CHECK:   %[[SHIFT_NDLB:.*]] = fir.shape_shift %c0, %c9 : (index, index) -> !fir.shapeshift<1>
! CHECK:   %[[COOR_NDLB:.*]] = fir.array_coor {{.*}}(%[[SHIFT_NDLB]]) %[[IV_NDLB_I64]] : (!fir.box<!fir.array<9xi32>>, !fir.shapeshift<1>, i64) -> !fir.ref<i32>
! CHECK:   %[[ELEM_NDLB:.*]] = fir.box_elesize %{{.*}} : (!fir.box<!fir.array<9xi32>>) -> index
! CHECK:   %[[ELEM_NDLB_I64:.*]] = fir.convert %[[ELEM_NDLB]] : (index) -> i64
! CHECK:   %[[ADDRI8_NDLB:.*]] = fir.convert %[[COOR_NDLB]] : (!fir.ref<i32>) -> !fir.ref<i8>
! CHECK:   %[[ENTRY_NDLB:.*]] = omp.affinity_entry %[[ADDRI8_NDLB]], %[[ELEM_NDLB_I64]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:   omp.yield(%[[ENTRY_NDLB]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>)
! CHECK: } -> !omp.iterated<!omp.affinity_entry_ty<!fir.ref<i8>, i64>>
! CHECK: omp.task affinity(%[[ITERATED_NDLB]] : !omp.iterated<!omp.affinity_entry_ty<!fir.ref<i8>, i64>>) {

subroutine task_affinity_iterator_nondefault_lb_2d()
  implicit none
  integer, parameter :: n = 4, m = 6
  integer :: a(0:n, -1:m)
  integer :: i, j

  !$omp parallel
  !$omp single
  !$omp task affinity(iterator(i = 0:n, j = -1:m) : a(i, j))
    a(0, -1) = 1
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtask_affinity_iterator_nondefault_lb_2d()
! CHECK: %[[ITERATED_NDLB2:.*]] = omp.iterator(%[[IV0_NDLB2:.*]]: index, %[[IV1_NDLB2:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}, {{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[IV0_NDLB2_I32:.*]] = fir.convert %[[IV0_NDLB2]] : (index) -> i32
! CHECK:   %[[IV1_NDLB2_I32:.*]] = fir.convert %[[IV1_NDLB2]] : (index) -> i32
! CHECK:   %[[IV0_NDLB2_I64:.*]] = fir.convert %[[IV0_NDLB2_I32]] : (i32) -> i64
! CHECK:   %[[IV1_NDLB2_I64:.*]] = fir.convert %[[IV1_NDLB2_I32]] : (i32) -> i64
! CHECK:   %[[SHIFT_NDLB2:.*]] = fir.shape_shift %c0, %c5, %c-1, %c8 : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:   %[[COOR_NDLB2:.*]] = fir.array_coor {{.*}}(%[[SHIFT_NDLB2]]) %[[IV0_NDLB2_I64]], %[[IV1_NDLB2_I64]] : (!fir.box<!fir.array<5x8xi32>>, !fir.shapeshift<2>, i64, i64) -> !fir.ref<i32>
! CHECK:   %[[ELEM_NDLB2:.*]] = fir.box_elesize %{{.*}} : (!fir.box<!fir.array<5x8xi32>>) -> index
! CHECK:   %[[ELEM_NDLB2_I64:.*]] = fir.convert %[[ELEM_NDLB2]] : (index) -> i64
! CHECK:   %[[ADDRI8_NDLB2:.*]] = fir.convert %[[COOR_NDLB2]] : (!fir.ref<i32>) -> !fir.ref<i8>
! CHECK:   %[[ENTRY_NDLB2:.*]] = omp.affinity_entry %[[ADDRI8_NDLB2]], %[[ELEM_NDLB2_I64]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:   omp.yield(%[[ENTRY_NDLB2]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>)
! CHECK: } -> !omp.iterated<!omp.affinity_entry_ty<!fir.ref<i8>, i64>>
! CHECK: omp.task affinity(%[[ITERATED_NDLB2]] : !omp.iterated<!omp.affinity_entry_ty<!fir.ref<i8>, i64>>) {

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
! CHECK:   %[[IV1_I64:.*]] = fir.convert %[[IV1_I32]] : (i32) -> i64
! CHECK:   %[[SHAPE:.*]] = fir.shape %c4, %c6 : (index, index) -> !fir.shape<2>
! CHECK:   %[[COOR:.*]] = fir.array_coor {{.*}}(%[[SHAPE]]) %[[IV0_I64]], %[[IV1_I64]] : (!fir.ref<!fir.array<4x6xi32>>, !fir.shape<2>, i64, i64) -> !fir.ref<i32>
! CHECK:   %[[C4:.*]] = arith.constant 4 : i64
! CHECK:   %[[ADDRI8:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !fir.ref<i8>
! CHECK:   %[[ENTRY:.*]] = omp.affinity_entry %[[ADDRI8]], %[[C4]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
! CHECK:   omp.yield(%[[ENTRY]] : !omp.affinity_entry_ty<!fir.ref<i8>, i64>)
! CHECK: } -> !omp.iterated<!omp.affinity_entry_ty<!fir.ref<i8>, i64>>
! CHECK: omp.task affinity(%[[ITER]] : !omp.iterated<!omp.affinity_entry_ty<!fir.ref<i8>, i64>>)

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
! CHECK:   %[[RO_IV0_I64:.*]] = fir.convert %[[RO_IV0_I32]] : (i32) -> i64
! CHECK:   %[[SHAPE:.*]] = fir.shape %c4, %c6 : (index, index) -> !fir.shape<2>
! CHECK:   %[[COOR:.*]] = fir.array_coor {{.*}}(%[[SHAPE]]) %[[RO_IV1_I64]], %[[RO_IV0_I64]] : (!fir.ref<!fir.array<4x6xi32>>, !fir.shape<2>, i64, i64) -> !fir.ref<i32>

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
! CHECK:   %[[IVB_I64:.*]] = fir.convert %[[IVB_I32]] : (i32) -> i64
! CHECK:   %[[SHAPE2:.*]] = fir.shape %c5, %c6 : (index, index) -> !fir.shape<2>
! CHECK:   %[[COOR2:.*]] = fir.array_coor {{.*}}(%[[SHAPE2]]) %[[IP1_I64]], %[[IVB_I64]] : (!fir.ref<!fir.array<5x6xi32>>, !fir.shape<2>, i64, i64) -> !fir.ref<i32>

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
! CHECK:   %[[C2_I32:.*]] = arith.constant 2 : i32
! CHECK:   %[[JP2_I32:.*]] = arith.addi %[[IVS1_I32]], %[[C2_I32]] : i32
! CHECK:   %[[JP2_I64:.*]] = fir.convert %[[JP2_I32]] : (i32) -> i64
! CHECK:   %[[SHAPE3:.*]] = fir.shape %c5, %c6 : (index, index) -> !fir.shape<2>
! CHECK:   %[[COOR3:.*]] = fir.array_coor {{.*}}(%[[SHAPE3]]) %[[IVS0_I64]], %[[JP2_I64]] : (!fir.ref<!fir.array<5x6xi32>>, !fir.shape<2>, i64, i64) -> !fir.ref<i32>

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
! CHECK:   %[[SHAPE4:.*]] = fir.shape %c5, %c6 : (index, index) -> !fir.shape<2>
! CHECK:   %[[COOR4:.*]] = fir.array_coor {{.*}}(%[[SHAPE4]]) %[[C1_IDX]], %[[JP2_I64_2]] : (!fir.ref<!fir.array<5x6xi32>>, !fir.shape<2>, index, i64) -> !fir.ref<i32>

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
! CHECK:   %[[SHAPE5:.*]] = fir.shape {{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[COOR5:.*]] = fir.array_coor {{.*}}(%[[SHAPE5]]) %[[IVC_I64]] : ({{.*}}, !fir.shape<1>, i64) -> !fir.ref<!fir.char<1,7>>
! CHECK:   %[[C1_I64:.*]] = arith.constant 1 : i64
! CHECK:   %[[C7_I64:.*]] = fir.convert %c7 : (index) -> i64
! CHECK:   %[[ELEM5:.*]] = arith.muli %[[C7_I64]], %[[C1_I64]] : i64
! CHECK:   %[[ADDR5:.*]] = fir.convert %[[COOR5]] : (!fir.ref<!fir.char<1,7>>) -> !fir.ref<i8>
! CHECK:   %[[ENTRY5:.*]] = omp.affinity_entry %[[ADDR5]], %[[ELEM5]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>

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
! CHECK:   %[[SHAPE6:.*]] = fir.shape {{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[COOR6:.*]] = fir.array_coor {{.*}}(%[[SHAPE6]]) %[[IP1C_I64]] : ({{.*}}, !fir.shape<1>, i64) -> !fir.ref<!fir.char<1,7>>
! CHECK:   %[[C1_I64_2:.*]] = arith.constant 1 : i64
! CHECK:   %[[C7_I64_2:.*]] = fir.convert %c7 : (index) -> i64
! CHECK:   %[[ELEM6:.*]] = arith.muli %[[C7_I64_2]], %[[C1_I64_2]] : i64
! CHECK:   %[[ADDR6:.*]] = fir.convert %[[COOR6]] : (!fir.ref<!fir.char<1,7>>) -> !fir.ref<i8>
! CHECK:   %[[ENTRY6:.*]] = omp.affinity_entry %[[ADDR6]], %[[ELEM6]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>

subroutine task_affinity_iterator_char_runtime(n, l)
  integer, intent(in) :: n, l
  character(len=l) :: a(n)
  integer :: i

  !$omp parallel
  !$omp single
  !$omp task affinity(iterator(i = 1:n) : a(i))
    a(1) = repeat("x", l)
  !$omp end task
  !$omp end single
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtask_affinity_iterator_char_runtime(
! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) typeparams %{{.*}} {uniq_name = "_QFtask_affinity_iterator_char_runtimeEa"}
! CHECK: %[[ITER:.*]] = omp.iterator(%[[IV:.*]]: index) = ({{.*}} to {{.*}} step {{.*}}) {
! CHECK:   %[[COOR:.*]] = fir.array_coor %[[A]]#0({{.*}}) {{.*}} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i64) -> !fir.ref<!fir.char<1,?>>
! CHECK:   %[[ELEM:.*]] = fir.box_elesize %[[A]]#0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:   %[[ELEM_I64:.*]] = fir.convert %[[ELEM]] : (index) -> i64
! CHECK:   %[[ADDR:.*]] = fir.convert %[[COOR]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:   %[[ENTRY:.*]] = omp.affinity_entry %[[ADDR]], %[[ELEM_I64]] : (!fir.ref<i8>, i64) -> !omp.affinity_entry_ty<!fir.ref<i8>, i64>
