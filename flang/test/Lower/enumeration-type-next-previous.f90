! Test lowering of the NEXT and PREVIOUS intrinsics for enumeration types.
! RUN: %flang_fc1 -fenumeration-type -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fenumeration-type -emit-fir %s -o /dev/null

module enum_np_mod
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
end module

! -----------------------------------------------------------------------------
!            Test NEXT() with a scalar argument and STAT
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_next(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>>
subroutine test_next(c)
  use enum_np_mod
  type(color), intent(in) :: c
  type(color) :: result
  integer :: stat
  ! CHECK: %[[C:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name = "_QFtest_nextEc"}
  ! CHECK: %[[RESULT:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_nextEresult"}
  ! CHECK: %[[STAT:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_nextEstat"}
  ! CHECK: %[[F:.*]] = hlfir.designate %[[C]]#0{"__ordinal"}
  ! CHECK: %[[ORD:.*]] = fir.load %[[F]] : !fir.ref<i32>
  ! Result ordinal is min(ordinal + 1, 3).
  ! CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
  ! CHECK-DAG: %[[MAX:.*]] = arith.constant 3 : i32
  ! CHECK: %[[INC:.*]] = arith.addi %[[ORD]], %[[ONE]] : i32
  ! CHECK: %[[CMP:.*]] = arith.cmpi sle, %[[INC]], %[[MAX]] : i32
  ! CHECK: %[[NEXT:.*]] = arith.select %[[CMP]], %[[INC]], %[[MAX]] : i32
  ! CHECK: %[[BOUND:.*]] = arith.cmpi eq, %[[ORD]], %[[MAX]] : i32
  ! A non-optional local STAT needs no presence check.
  ! CHECK-NOT: fir.is_present
  ! CHECK-DAG: %[[C112:.*]] = arith.constant 112 : i32
  ! CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  ! CHECK: %[[S:.*]] = arith.select %[[BOUND]], %[[C112]], %[[C0]] : i32
  ! CHECK: hlfir.assign %[[S]] to %[[STAT]]#0 : i32, !fir.ref<i32>
  ! CHECK: %[[TMP:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp.intrinsic_result"}
  ! CHECK: %[[TF:.*]] = hlfir.designate %[[TMP]]#0{"__ordinal"}
  ! CHECK: hlfir.assign %[[NEXT]] to %[[TF]] : i32, !fir.ref<i32>
  ! CHECK: %[[E:.*]] = hlfir.as_expr %[[TMP]]#0
  ! CHECK: hlfir.assign %[[E]] to %[[RESULT]]#0
  ! CHECK: hlfir.destroy %[[E]]
  result = next(c, stat=stat)
end subroutine

! -----------------------------------------------------------------------------
!            Test PREVIOUS() with a scalar argument and STAT
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_previous(
subroutine test_previous(c)
  use enum_np_mod
  type(color), intent(in) :: c
  type(color) :: result
  integer :: stat
  ! CHECK: %[[C:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name = "_QFtest_previousEc"}
  ! CHECK: %[[F:.*]] = hlfir.designate %[[C]]#0{"__ordinal"}
  ! CHECK: %[[ORD:.*]] = fir.load %[[F]] : !fir.ref<i32>
  ! Result ordinal is max(ordinal - 1, 1).
  ! CHECK: %[[ONE:.*]] = arith.constant 1 : i32
  ! CHECK: %[[DEC:.*]] = arith.subi %[[ORD]], %[[ONE]] : i32
  ! CHECK: %[[CMP:.*]] = arith.cmpi sge, %[[DEC]], %[[ONE]] : i32
  ! CHECK: %[[PREV:.*]] = arith.select %[[CMP]], %[[DEC]], %[[ONE]] : i32
  ! CHECK: %[[BOUND:.*]] = arith.cmpi eq, %[[ORD]], %[[ONE]] : i32
  ! CHECK: arith.select %[[BOUND]]
  ! CHECK: hlfir.assign
  ! CHECK: %[[TMP:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp.intrinsic_result"}
  ! CHECK: %[[TF:.*]] = hlfir.designate %[[TMP]]#0{"__ordinal"}
  ! CHECK: hlfir.assign %[[PREV]] to %[[TF]] : i32, !fir.ref<i32>
  result = previous(c, stat=stat)
end subroutine

! -----------------------------------------------------------------------------
!            Test NEXT() without STAT
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_next_no_stat(
subroutine test_next_no_stat(c)
  use enum_np_mod
  type(color), intent(in) :: c
  type(color) :: result
  ! CHECK: %[[BOUND:.*]] = arith.cmpi eq
  ! CHECK: fir.if %[[BOUND]] {
  ! CHECK:   fir.call @_FortranAReportFatalUserError
  ! CHECK: }
  ! CHECK-NOT: arith.constant 112
  ! CHECK: hlfir.declare %{{.*}} {uniq_name = ".tmp.intrinsic_result"}
  result = next(c)
end subroutine

! -----------------------------------------------------------------------------
!            Test NEXT() with a STAT that may be absent at run time
! -----------------------------------------------------------------------------

! An absent optional dummy, an unallocated allocatable, or a disassociated
! pointer forwarded as STAT= is not present: STAT is not written and the
! boundary is a fatal error.

! CHECK-LABEL: func.func @_QPtest_next_optional_stat(
subroutine test_next_optional_stat(c, stat)
  use enum_np_mod
  type(color), intent(in) :: c
  integer, optional, intent(out) :: stat
  type(color) :: result
  ! CHECK: %[[STAT:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name = "_QFtest_next_optional_statEstat"}
  ! CHECK: %[[PRES:.*]] = fir.is_present %[[STAT]]#0 : (!fir.ref<i32>) -> i1
  ! CHECK: %[[BOUND:.*]] = arith.cmpi eq
  ! CHECK: fir.if %[[PRES]] {
  ! CHECK:   arith.select %[[BOUND]]
  ! CHECK:   hlfir.assign %{{.*}} to %[[STAT]]#0
  ! CHECK: } else {
  ! CHECK:   fir.if %[[BOUND]] {
  ! CHECK:     fir.call @_FortranAReportFatalUserError
  result = next(c, stat=stat)
end subroutine

! CHECK-LABEL: func.func @_QPtest_next_allocatable_stat(
subroutine test_next_allocatable_stat(c, stat)
  use enum_np_mod
  type(color), intent(in) :: c
  integer, allocatable, intent(inout) :: stat
  type(color) :: result
  ! CHECK: fir.box_addr
  ! CHECK: %[[PRES:.*]] = arith.cmpi ne
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %{{.*}} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  ! CHECK: %[[BOUND:.*]] = arith.cmpi eq
  ! CHECK: fir.if %[[PRES]] {
  ! CHECK:   arith.select %[[BOUND]]
  ! CHECK:   hlfir.assign %{{.*}} to %[[ADDR]] : i32, !fir.heap<i32>
  ! CHECK: } else {
  ! CHECK:   fir.if %[[BOUND]] {
  ! CHECK:     fir.call @_FortranAReportFatalUserError
  result = next(c, stat=stat)
end subroutine

! CHECK-LABEL: func.func @_QPtest_next_pointer_stat(
subroutine test_next_pointer_stat(c, stat)
  use enum_np_mod
  type(color), intent(in) :: c
  integer, pointer, intent(in) :: stat
  type(color) :: nc
  ! CHECK: fir.box_addr
  ! CHECK: %[[PRES:.*]] = arith.cmpi ne
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %{{.*}} : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
  ! CHECK: %[[BOUND:.*]] = arith.cmpi eq
  ! CHECK: fir.if %[[PRES]] {
  ! CHECK:   arith.select %[[BOUND]]
  ! CHECK:   hlfir.assign %{{.*}} to %[[ADDR]] : i32, !fir.ptr<i32>
  ! CHECK: } else {
  ! CHECK:   fir.if %[[BOUND]] {
  ! CHECK:     fir.call @_FortranAReportFatalUserError
  nc = next(c, stat=stat)
end subroutine

! -----------------------------------------------------------------------------
!            Test NEXT() and PREVIOUS() with an allocatable or pointer A
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_next_allocatable_a(
subroutine test_next_allocatable_a(a)
  use enum_np_mod
  type(color), allocatable, intent(in) :: a
  type(color) :: nc
  ! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name = "_QFtest_next_allocatable_aEa"}
  ! CHECK: %[[BOX:.*]] = fir.load %[[A]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>>>>
  ! CHECK: %[[ADDR:.*]] = fir.box_addr %[[BOX]]
  ! CHECK: %[[F:.*]] = hlfir.designate %[[ADDR]]{"__ordinal"} : (!fir.heap<!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>>) -> !fir.ref<i32>
  ! CHECK: fir.load %[[F]] : !fir.ref<i32>
  nc = next(a)
end subroutine

! CHECK-LABEL: func.func @_QPtest_next_allocatable_array_a(
subroutine test_next_allocatable_array_a(a)
  use enum_np_mod
  type(color), allocatable, intent(in) :: a(:)
  type(color) :: narr(3)
  integer :: stat(3)
  ! CHECK: %[[A:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name = "_QFtest_next_allocatable_array_aEa"}
  ! CHECK: %[[STAT:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_next_allocatable_array_aEstat"}
  ! CHECK: %[[BOX:.*]] = fir.load %[[A]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>>>>>
  ! CHECK: %[[DIMS:.*]]:3 = fir.box_dims %[[BOX]], %{{.*}}
  ! CHECK: %[[SHAPE:.*]] = fir.shape %[[DIMS]]#1
  ! CHECK: hlfir.elemental %[[SHAPE]] : (!fir.shape<1>) -> !hlfir.expr<?x!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>> {
  ! CHECK: ^bb0(%[[I:.*]]: index):
  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[BOX]] (%{{.*}})
  ! CHECK: hlfir.designate %[[ELT]]{"__ordinal"}
  ! CHECK: %[[SE:.*]] = hlfir.designate %[[STAT]]#0 (%[[I]])
  ! CHECK: hlfir.assign %{{.*}} to %[[SE]] : i32, !fir.ref<i32>
  narr = next(a, stat=stat)
end subroutine

! CHECK-LABEL: func.func @_QPtest_previous_pointer_a(
subroutine test_previous_pointer_a(p)
  use enum_np_mod
  type(color), pointer, intent(in) :: p(:)
  type(color) :: parr(3)
  ! CHECK: %[[P:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name = "_QFtest_previous_pointer_aEp"}
  ! CHECK: %[[BOX:.*]] = fir.load %[[P]]#0
  ! CHECK: %[[DIMS:.*]]:3 = fir.box_dims %[[BOX]], %{{.*}}
  ! CHECK: %[[SHAPE:.*]] = fir.shape %[[DIMS]]#1
  ! CHECK: hlfir.elemental %[[SHAPE]] unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>> {
  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[BOX]] (%{{.*}})
  ! CHECK: hlfir.designate %[[ELT]]{"__ordinal"}
  ! CHECK: arith.subi
  parr = previous(p)
end subroutine

! -----------------------------------------------------------------------------
!            Test NEXT() and PREVIOUS() over whole arrays
! -----------------------------------------------------------------------------

! An array call with STAT is an ordered hlfir.elemental over the record type
! (no "unordered"), since STAT is an elemental INTENT(OUT) argument.

! CHECK-LABEL: func.func @_QPtest_next_array(
subroutine test_next_array(arr)
  use enum_np_mod
  type(color), intent(in) :: arr(3)
  type(color) :: narr(3)
  integer :: stat(3)
  ! CHECK: %[[STAT:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_next_arrayEstat"}
  ! CHECK: %[[RES:.*]] = hlfir.elemental %{{[0-9]+}} : (!fir.shape<1>) -> !hlfir.expr<3x!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>> {
  ! CHECK: ^bb0(%[[I:.*]]: index):
  ! CHECK: %[[ELT:.*]] = hlfir.designate %{{.*}} (%[[I]])
  ! CHECK: %[[F:.*]] = hlfir.designate %[[ELT]]{"__ordinal"}
  ! CHECK: %[[ORD:.*]] = fir.load %[[F]] : !fir.ref<i32>
  ! CHECK: arith.addi %[[ORD]]
  ! CHECK: %[[BOUND:.*]] = arith.cmpi eq, %[[ORD]]
  ! CHECK: %[[SE:.*]] = hlfir.designate %[[STAT]]#0 (%[[I]])
  ! CHECK: %[[S:.*]] = arith.select %[[BOUND]]
  ! CHECK: hlfir.assign %[[S]] to %[[SE]] : i32, !fir.ref<i32>
  ! CHECK: hlfir.yield_element %{{.*}} : !hlfir.expr<!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>>
  ! CHECK: }
  ! CHECK: hlfir.assign %[[RES]] to
  ! CHECK: hlfir.destroy %[[RES]]
  narr = next(arr, stat=stat)
end subroutine

! CHECK-LABEL: func.func @_QPtest_previous_array(
subroutine test_previous_array(arr)
  use enum_np_mod
  type(color), intent(in) :: arr(3)
  type(color) :: parr(3)
  integer :: stat(3)
  ! CHECK: hlfir.elemental %{{[0-9]+}} : (!fir.shape<1>) -> !hlfir.expr<3x!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>> {
  ! CHECK: %[[ORD:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK: %[[ONE:.*]] = arith.constant 1 : i32
  ! CHECK: %[[DEC:.*]] = arith.subi %[[ORD]], %[[ONE]] : i32
  ! CHECK: %[[CMP:.*]] = arith.cmpi sge, %[[DEC]], %[[ONE]] : i32
  ! CHECK: arith.select %[[CMP]], %[[DEC]], %[[ONE]] : i32
  ! CHECK: hlfir.yield_element
  parr = previous(arr, stat=stat)
end subroutine

! CHECK-LABEL: func.func @_QPtest_next_array_optional_stat(
subroutine test_next_array_optional_stat(arr, stat)
  use enum_np_mod
  type(color), intent(in) :: arr(3)
  integer, optional, intent(out) :: stat(3)
  type(color) :: narr(3)
  ! CHECK: %[[STAT:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name = "_QFtest_next_array_optional_statEstat"}
  ! CHECK: %[[PRES:.*]] = fir.is_present %[[STAT]]#0 : (!fir.ref<!fir.array<3xi32>>) -> i1
  ! CHECK: hlfir.elemental %{{[0-9]+}} : (!fir.shape<1>)
  ! CHECK: ^bb0(%[[I:.*]]: index):
  ! CHECK: %[[BOUND:.*]] = arith.cmpi eq
  ! CHECK: fir.if %[[PRES]] {
  ! CHECK:   %[[SE:.*]] = hlfir.designate %[[STAT]]#0 (%[[I]])
  ! CHECK:   arith.select %[[BOUND]]
  ! CHECK:   hlfir.assign %{{.*}} to %[[SE]] : i32, !fir.ref<i32>
  ! CHECK: } else {
  ! CHECK:   fir.if %[[BOUND]] {
  ! CHECK:     fir.call @_FortranAReportFatalUserError
  narr = next(arr, stat=stat)
end subroutine

! -----------------------------------------------------------------------------
!            Test NEXT() inside WHERE
! -----------------------------------------------------------------------------

! Without STAT the call is pure and unordered; WHERE only evaluates it where
! the mask is true, so a boundary in a masked-off element does not terminate.

! CHECK-LABEL: func.func @_QPtest_next_where(
subroutine test_next_where(arr, mask)
  use enum_np_mod
  type(color), intent(in) :: arr(3)
  logical, intent(in) :: mask(3)
  type(color) :: narr(3)
  ! CHECK: hlfir.where {
  ! CHECK: } do {
  ! CHECK: hlfir.region_assign {
  ! CHECK: hlfir.elemental %{{[0-9]+}} unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>> {
  ! CHECK: fir.call @_FortranAReportFatalUserError
  ! CHECK: hlfir.yield_element
  where (mask) narr = next(arr)
end subroutine

! Composed calls read NEXT element by element (hlfir.apply) instead of
! materializing it, so WHERE can inline it under the mask.

! CHECK-LABEL: func.func @_QPtest_int_next_where(
subroutine test_int_next_where(arr, mask)
  use enum_np_mod
  type(color), intent(in) :: arr(3)
  logical, intent(in) :: mask(3)
  integer :: r(3)
  ! CHECK: hlfir.region_assign {
  ! CHECK: %[[N:.*]] = hlfir.elemental %{{[0-9]+}} unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>> {
  ! CHECK-NOT: hlfir.associate %[[N]](
  ! CHECK: hlfir.elemental %{{[0-9]+}} unordered : (!fir.shape<1>) -> !hlfir.expr<3xi32> {
  ! CHECK: hlfir.apply %[[N]], %{{.*}}
  ! CHECK: hlfir.yield_element %{{.*}} : i32
  where (mask) r = int(next(arr))
end subroutine

! CHECK-LABEL: func.func @_QPtest_next_next_where(
subroutine test_next_next_where(arr, mask)
  use enum_np_mod
  type(color), intent(in) :: arr(3)
  logical, intent(in) :: mask(3)
  type(color) :: narr(3)
  ! CHECK: hlfir.region_assign {
  ! CHECK: %[[N:.*]] = hlfir.elemental %{{[0-9]+}} unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>> {
  ! CHECK-NOT: hlfir.associate %[[N]](
  ! CHECK: hlfir.elemental %{{[0-9]+}} unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>> {
  ! CHECK: hlfir.apply %[[N]], %{{.*}}
  ! CHECK: hlfir.yield_element
  where (mask) narr = next(next(arr))
end subroutine

! An ordered call (STAT present) still reads its argument through hlfir.apply,
! so nested NEXT calls with STAT stay under the WHERE mask.

! CHECK-LABEL: func.func @_QPtest_next_stat_nested_where(
subroutine test_next_stat_nested_where(arr, mask)
  use enum_np_mod
  type(color), intent(in) :: arr(3)
  logical, intent(in) :: mask(3)
  type(color) :: narr(3)
  integer :: st(3), st2(3)
  ! CHECK: hlfir.region_assign {
  ! CHECK: %[[N1:.*]] = hlfir.elemental %{{[0-9]+}} unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>> {
  ! CHECK-NOT: hlfir.associate %[[N1]](
  ! CHECK: %[[N2:.*]] = hlfir.elemental %{{[0-9]+}} : (!fir.shape<1>) -> !hlfir.expr<3x!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>> {
  ! CHECK: hlfir.apply %[[N1]], %{{.*}}
  ! CHECK-NOT: hlfir.associate %[[N2]](
  ! CHECK: hlfir.elemental %{{[0-9]+}} : (!fir.shape<1>) -> !hlfir.expr<3x!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>> {
  ! CHECK: hlfir.apply %[[N2]], %{{.*}}
  where (mask) narr = next(next(next(arr), stat=st2), stat=st)
end subroutine

! The STAT write is inside the masked elemental, so STAT elements where the
! mask is false are left unchanged.

! CHECK-LABEL: func.func @_QPtest_next_where_stat(
subroutine test_next_where_stat(arr, mask)
  use enum_np_mod
  type(color), intent(in) :: arr(3)
  logical, intent(in) :: mask(3)
  type(color) :: narr(3)
  integer :: stat(3)
  ! CHECK: %[[STAT:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_next_where_statEstat"}
  ! CHECK: hlfir.where {
  ! CHECK: } do {
  ! CHECK: hlfir.region_assign {
  ! CHECK: hlfir.elemental %{{[0-9]+}} : (!fir.shape<1>) -> !hlfir.expr<3x!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>> {
  ! CHECK: ^bb0(%[[I:.*]]: index):
  ! CHECK: %[[BOUND:.*]] = arith.cmpi eq
  ! CHECK: %[[SE:.*]] = hlfir.designate %[[STAT]]#0 (%[[I]])
  ! CHECK: %[[S:.*]] = arith.select %[[BOUND]]
  ! CHECK: hlfir.assign %[[S]] to %[[SE]] : i32, !fir.ref<i32>
  ! CHECK: hlfir.yield_element
  ! CHECK: } to {
  where (mask) narr = next(arr, stat=stat)
end subroutine

! -----------------------------------------------------------------------------
!            Test NEXT() with a scalar A and an array STAT
! -----------------------------------------------------------------------------

! The result shape comes from STAT, and A is read in every iteration.

! CHECK-LABEL: func.func @_QPtest_next_scalar_a_array_stat(
subroutine test_next_scalar_a_array_stat(c)
  use enum_np_mod
  type(color), intent(in) :: c
  type(color) :: narr(3)
  integer :: stat(3)
  ! CHECK: %[[C:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name = "_QFtest_next_scalar_a_array_statEc"}
  ! CHECK: %[[STAT:.*]]:2 = hlfir.declare %{{.*}}(%[[SHAPE:.*]]) {uniq_name = "_QFtest_next_scalar_a_array_statEstat"}
  ! CHECK: hlfir.elemental %[[SHAPE]] : (!fir.shape<1>) -> !hlfir.expr<3x!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>> {
  ! CHECK: ^bb0(%[[I:.*]]: index):
  ! CHECK: hlfir.designate %[[C]]#0{"__ordinal"}
  ! CHECK: %[[SE:.*]] = hlfir.designate %[[STAT]]#0 (%[[I]])
  ! CHECK: hlfir.assign %{{.*}} to %[[SE]] : i32, !fir.ref<i32>
  narr = next(c, stat=stat)
end subroutine

! -----------------------------------------------------------------------------
!            Test NEXT() with a vector-subscripted STAT
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_next_vector_subscript_stat(
subroutine test_next_vector_subscript_stat(arr, idx)
  use enum_np_mod
  type(color), intent(in) :: arr(3)
  integer, intent(in) :: idx(3)
  type(color) :: narr(3)
  integer :: stat(5)
  ! CHECK: %[[STAT:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_next_vector_subscript_statEstat"}
  ! CHECK: %[[IDX:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<3xi64> {
  ! CHECK: hlfir.elemental %{{[0-9]+}} : (!fir.shape<1>) -> !hlfir.expr<3x!fir.type<_QMenum_np_modTcolor{__ordinal:i32}>> {
  ! CHECK: ^bb0(%[[I:.*]]: index):
  ! CHECK: %[[BOUND:.*]] = arith.cmpi eq
  ! CHECK: %[[J:.*]] = hlfir.apply %[[IDX]], %[[I]] : (!hlfir.expr<3xi64>, index) -> i64
  ! CHECK: %[[SE:.*]] = hlfir.designate %[[STAT]]#0 (%[[J]])
  ! CHECK: %[[S:.*]] = arith.select %[[BOUND]]
  ! CHECK: hlfir.assign %[[S]] to %[[SE]] : i32, !fir.ref<i32>
  ! CHECK: hlfir.destroy %[[IDX]]
  narr = next(arr, stat=stat(idx))
end subroutine

! -----------------------------------------------------------------------------
!            Test STAT of non-default integer kinds
! -----------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPtest_next_stat_kinds(
subroutine test_next_stat_kinds(c, arr)
  use enum_np_mod
  type(color), intent(in) :: c, arr(3)
  type(color) :: nc, parr(3)
  integer(8) :: stat8
  integer(2) :: stat2(3)
  ! CHECK: %[[S2:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_next_stat_kindsEstat2"} : (!fir.ref<!fir.array<3xi16>>
  ! CHECK: %[[S8:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtest_next_stat_kindsEstat8"} : (!fir.ref<i64>)
  ! CHECK-DAG: %[[C112_8:.*]] = arith.constant 112 : i64
  ! CHECK-DAG: %[[C0_8:.*]] = arith.constant 0 : i64
  ! CHECK: %[[V8:.*]] = arith.select %{{.*}}, %[[C112_8]], %[[C0_8]] : i64
  ! CHECK: hlfir.assign %[[V8]] to %[[S8]]#0 : i64, !fir.ref<i64>
  nc = next(c, stat=stat8)
  ! CHECK: hlfir.elemental
  ! CHECK: %[[E2:.*]] = hlfir.designate %[[S2]]#0 (%{{.*}}) : (!fir.ref<!fir.array<3xi16>>, index) -> !fir.ref<i16>
  ! CHECK-DAG: %[[C112_2:.*]] = arith.constant 112 : i16
  ! CHECK-DAG: %[[C0_2:.*]] = arith.constant 0 : i16
  ! CHECK: %[[V2:.*]] = arith.select %{{.*}}, %[[C112_2]], %[[C0_2]] : i16
  ! CHECK: hlfir.assign %[[V2]] to %[[E2]] : i16, !fir.ref<i16>
  parr = previous(arr, stat=stat2)
end subroutine

! CHECK: fir.string_lit "NEXT or PREVIOUS of enumeration type at boundary without STAT=\00"
