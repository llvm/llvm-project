! RUN: bbc -emit-fir -polymorphic-type %s -o - | FileCheck %s

module storage_size_test
  type :: p1
    integer :: a
  end type

  type, extends(p1) :: p2
    integer :: b
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
! CHECK: %[[P_ADDR:.*]] = fir.box_addr %[[LOAD_P]] : (!fir.class<!fir.ptr<!fir.type<_QMstorage_size_testTp1{a:i32}>>>) -> !fir.ptr<!fir.type<_QMstorage_size_testTp1{a:i32}>>
! CHECK: %[[P_ADDR_I64:.*]] = fir.convert %[[P_ADDR]] : (!fir.ptr<!fir.type<_QMstorage_size_testTp1{a:i32}>>) -> i64
! CHECK: %[[C0:.*]] = arith.constant 0 : i64
! CHECK: %[[IS_NULL_ADDR:.*]] = arith.cmpi eq, %[[P_ADDR_I64]], %[[C0]] : i64
! CHECK: fir.if %[[IS_NULL_ADDR]] {
! CHECK:   %{{.*}} = fir.call @_FortranAReportFatalUserError(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<i8>, !fir.ref<i8>, i32) -> none
! CHECK: }
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

end module
