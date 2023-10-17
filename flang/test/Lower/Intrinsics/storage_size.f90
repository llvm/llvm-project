! RUN: bbc -emit-fir -polymorphic-type %s -o - | FileCheck %s

module storage_size_test
  type :: p1
    integer :: a
  end type

  type, extends(p1) :: p2
    integer :: b
  end type

  type :: p3
    class(p1), pointer :: p(:)
  end type

contains

  integer function unlimited_polymorphic_pointer(p) result(size)
    class(*), pointer :: p
    size = storage_size(p)
  end function

! CHECK-LABEL: func.func @_QMstorage_size_testPunlimited_polymorphic_pointer(
! CHECK-SAME: %[[P:.*]]: !fir.ref<!fir.class<!fir.ptr<none>>> {fir.bindc_name = "p"}) -> i32 {
! CHECK: %[[SIZE:.*]] = fir.alloca i32 {bindc_name = "size", uniq_name = "_QMstorage_size_testFunlimited_polymorphic_pointerEsize"}
! CHECK: %[[LOAD_P:.*]] = fir.load %[[P]] : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK: %[[P_ADDR:.*]] = fir.box_addr %[[LOAD_P]] : (!fir.class<!fir.ptr<none>>) -> !fir.ptr<none>
! CHECK: %[[P_ADDR_I64:.*]] = fir.convert %[[P_ADDR]] : (!fir.ptr<none>) -> i64
! CHECK: %[[C0:.*]] = arith.constant 0 : i64
! CHECK: %[[IS_NULL_ADDR:.*]] = arith.cmpi eq, %[[P_ADDR_I64]], %[[C0]] : i64
! CHECK: fir.if %[[IS_NULL_ADDR]] {
! CHECK:   %{{.*}} = fir.call @_FortranAReportFatalUserError(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<i8>, !fir.ref<i8>, i32) -> none
! CHECK: }
! CHECK: %[[LOAD_P:.*]] = fir.load %[[P]] : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK: %[[ELE_SIZE:.*]] = fir.box_elesize %[[LOAD_P]] : (!fir.class<!fir.ptr<none>>) -> i32
! CHECK: %[[C8:.*]] = arith.constant 8 : i32
! CHECK: %[[BITS:.*]] = arith.muli %[[ELE_SIZE]], %[[C8]] : i32
! CHECK: fir.store %[[BITS]] to %[[SIZE]] : !fir.ref<i32>
! CHECK: %[[RES:.*]] = fir.load %[[SIZE]] : !fir.ref<i32>
! CHECK: return %[[RES]] : i32

  integer function unlimited_polymorphic_allocatable(p) result(size)
    class(*), allocatable :: p
    size = storage_size(p)
  end function

! CHECK-LABEL: func.func @_QMstorage_size_testPunlimited_polymorphic_allocatable(
! CHECK-SAME: %[[P:.*]]: !fir.ref<!fir.class<!fir.heap<none>>> {fir.bindc_name = "p"}) -> i32 {
! CHECK: %[[SIZE:.*]] = fir.alloca i32 {bindc_name = "size", uniq_name = "_QMstorage_size_testFunlimited_polymorphic_allocatableEsize"}
! CHECK: %[[LOAD_P:.*]] = fir.load %[[P]] : !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK: %[[P_ADDR:.*]] = fir.box_addr %[[LOAD_P]] : (!fir.class<!fir.heap<none>>) -> !fir.heap<none>
! CHECK: %[[P_ADDR_I64:.*]] = fir.convert %[[P_ADDR]] : (!fir.heap<none>) -> i64
! CHECK: %[[C0:.*]] = arith.constant 0 : i64
! CHECK: %[[IS_NULL_ADDR:.*]] = arith.cmpi eq, %[[P_ADDR_I64]], %[[C0]] : i64
! CHECK: fir.if %[[IS_NULL_ADDR]] {
! CHECK:   %{{.*}} = fir.call @_FortranAReportFatalUserError(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<i8>, !fir.ref<i8>, i32) -> none
! CHECK: }
! CHECK: %[[LOAD_P:.*]] = fir.load %[[P]] : !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK: %[[ELE_SIZE:.*]] = fir.box_elesize %[[LOAD_P]] : (!fir.class<!fir.heap<none>>) -> i32
! CHECK: %[[C8:.*]] = arith.constant 8 : i32
! CHECK: %[[BITS:.*]] = arith.muli %[[ELE_SIZE]], %[[C8]] : i32
! CHECK: fir.store %[[BITS]] to %[[SIZE]] : !fir.ref<i32>
! CHECK: %[[RES:.*]] = fir.load %[[SIZE]] : !fir.ref<i32>
! CHECK: return %[[RES]] : i32

  integer function polymorphic_pointer(p) result(size)
    class(p1), pointer :: p
    size = storage_size(p)
  end function

! CHECK-LABEL: func.func @_QMstorage_size_testPpolymorphic_pointer(
! CHECK-SAME: %[[P:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMstorage_size_testTp1{a:i32}>>>> {fir.bindc_name = "p"}) -> i32 {
! CHECK: %[[SIZE:.*]] = fir.alloca i32 {bindc_name = "size", uniq_name = "_QMstorage_size_testFpolymorphic_pointerEsize"}
! CHECK: %[[LOAD_P:.*]] = fir.load %[[P]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMstorage_size_testTp1{a:i32}>>>>
! CHECK: %[[ELE_SIZE:.*]] = fir.box_elesize %[[LOAD_P]] : (!fir.class<!fir.ptr<!fir.type<_QMstorage_size_testTp1{a:i32}>>>) -> i32
! CHECK: %[[C8:.*]] = arith.constant 8 : i32
! CHECK: %[[BITS:.*]] = arith.muli %[[ELE_SIZE]], %[[C8]] : i32
! CHECK: fir.store %[[BITS]] to %[[SIZE]] : !fir.ref<i32>
! CHECK: %[[RES:.*]] = fir.load %[[SIZE]] : !fir.ref<i32>
! CHECK: return %[[RES]] : i32

  integer function polymorphic(p) result(size)
    class(p1) :: p
    size = storage_size(p)
  end function

! CHECK-LABEL: func.func @_QMstorage_size_testPpolymorphic(
! CHECK-SAME: %[[P:.*]]: !fir.class<!fir.type<_QMstorage_size_testTp1{a:i32}>> {fir.bindc_name = "p"}) -> i32 {
! CHECK: %[[SIZE:.*]] = fir.alloca i32 {bindc_name = "size", uniq_name = "_QMstorage_size_testFpolymorphicEsize"}
! CHECK: %[[ELE_SIZE:.*]] = fir.box_elesize %[[P]] : (!fir.class<!fir.type<_QMstorage_size_testTp1{a:i32}>>) -> i32
! CHECK: %[[C8:.*]] = arith.constant 8 : i32
! CHECK: %[[BITS:.*]] = arith.muli %[[ELE_SIZE]], %[[C8]] : i32
! CHECK: fir.store %[[BITS]] to %[[SIZE]] : !fir.ref<i32>
! CHECK: %[[RES:.*]] = fir.load %[[SIZE]] : !fir.ref<i32>
! CHECK: return %[[RES]] : i32

  integer(8) function polymorphic_rank(p) result(size)
    class(p1) :: p
    size = storage_size(p, 8)
  end function

! CHECK-LABEL: func.func @_QMstorage_size_testPpolymorphic_rank(
! CHECK-SAME: %[[P:.*]]: !fir.class<!fir.type<_QMstorage_size_testTp1{a:i32}>> {fir.bindc_name = "p"}) -> i64 {
! CHECK: %[[SIZE:.*]] = fir.alloca i64 {bindc_name = "size", uniq_name = "_QMstorage_size_testFpolymorphic_rankEsize"}
! CHECK: %[[ELE_SIZE:.*]] = fir.box_elesize %[[P]] : (!fir.class<!fir.type<_QMstorage_size_testTp1{a:i32}>>) -> i64
! CHECK: %[[C8:.*]] = arith.constant 8 : i64
! CHECK: %[[BITS:.*]] = arith.muli %[[ELE_SIZE]], %[[C8]] : i64
! CHECK: fir.store %[[BITS]] to %[[SIZE]] : !fir.ref<i64>
! CHECK: %[[RES:.*]] = fir.load %[[SIZE]] : !fir.ref<i64>
! CHECK: return %[[RES]] : i64

  integer function polymorphic_value(t) result(size)
    type(p3) :: t
    size = storage_size(t%p(1))
  end function

! CHECK-LABEL: func.func @_QMstorage_size_testPpolymorphic_value(
! CHECK-SAME: %[[T:.*]]: !fir.ref<!fir.type<_QMstorage_size_testTp3{p:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMstorage_size_testTp1{a:i32}>>>>}>> {fir.bindc_name = "t"}) -> i32 {
! CHECK: %[[ALLOCA:.*]] = fir.alloca i32 {bindc_name = "size", uniq_name = "_QMstorage_size_testFpolymorphic_valueEsize"}
! CHECK: %[[FIELD_P:.*]] = fir.field_index p, !fir.type<_QMstorage_size_testTp3{p:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMstorage_size_testTp1{a:i32}>>>>}>
! CHECK: %[[COORD_P:.*]] = fir.coordinate_of %[[T]], %[[FIELD_P]] : (!fir.ref<!fir.type<_QMstorage_size_testTp3{p:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMstorage_size_testTp1{a:i32}>>>>}>>, !fir.field) -> !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMstorage_size_testTp1{a:i32}>>>>>
! CHECK: %[[LOAD_COORD_P:.*]] = fir.load %[[COORD_P]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMstorage_size_testTp1{a:i32}>>>>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[LOAD_COORD_P]], %[[C0]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMstorage_size_testTp1{a:i32}>>>>, index) -> (index, index, index)
! CHECK: %[[C1:.*]] = arith.constant 1 : i64
! CHECK: %[[DIMI64:.*]] = fir.convert %[[BOX_DIMS]]#0 : (index) -> i64
! CHECK: %[[IDX:.*]] = arith.subi %[[C1]], %[[DIMI64]] : i64
! CHECK: %[[COORD_OF:.*]] = fir.coordinate_of %[[LOAD_COORD_P]], %[[IDX]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMstorage_size_testTp1{a:i32}>>>>, i64) -> !fir.ref<!fir.type<_QMstorage_size_testTp1{a:i32}>>
! CHECK: %[[BOXED:.*]] = fir.embox %[[COORD_OF]] source_box %[[LOAD_COORD_P]] : (!fir.ref<!fir.type<_QMstorage_size_testTp1{a:i32}>>, !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMstorage_size_testTp1{a:i32}>>>>) -> !fir.class<!fir.type<_QMstorage_size_testTp1{a:i32}>>
! CHECK: %[[ELE_SIZE:.*]] = fir.box_elesize %[[BOXED]] : (!fir.class<!fir.type<_QMstorage_size_testTp1{a:i32}>>) -> i32
! CHECK: %[[C8:.*]] = arith.constant 8 : i32
! CHECK: %[[SIZE:.*]] = arith.muli %[[ELE_SIZE]], %[[C8]] : i32
! CHECK: fir.store %[[SIZE]] to %[[ALLOCA]] : !fir.ref<i32>
! CHECK: %[[RET:.*]] = fir.load %[[ALLOCA]] : !fir.ref<i32>
! CHECK: return %[[RET]] : i32

end module
