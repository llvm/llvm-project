! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

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
! CHECK-SAME: %[[P:.*]]: !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK: %[[pDecl:.*]]:2 = hlfir.declare %[[P]]
! CHECK: %[[LOAD_P:.*]] = fir.load %[[pDecl]]#0 : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK: %[[P_ADDR:.*]] = fir.box_addr %[[LOAD_P]] : (!fir.class<!fir.ptr<none>>) -> !fir.ptr<none>
! CHECK: %[[P_ADDR_I64:.*]] = fir.convert %[[P_ADDR]] : (!fir.ptr<none>) -> i64
! CHECK: %[[C0:.*]] = arith.constant 0 : i64
! CHECK: %[[IS_NULL_ADDR:.*]] = arith.cmpi eq, %[[P_ADDR_I64]], %[[C0]] : i64
! CHECK: fir.if %[[IS_NULL_ADDR]] {
! CHECK:   fir.call @_FortranAReportFatalUserError
! CHECK: }
! CHECK: %[[LOAD_P2:.*]] = fir.load %[[pDecl]]#0 : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK: %[[ELE_SIZE:.*]] = fir.box_elesize %[[LOAD_P2]] : (!fir.class<!fir.ptr<none>>) -> i32
! CHECK: %[[C8:.*]] = arith.constant 8 : i32
! CHECK: %[[BITS:.*]] = arith.muli %[[ELE_SIZE]], %[[C8]] : i32

  integer function unlimited_polymorphic_allocatable(p) result(size)
    class(*), allocatable :: p
    size = storage_size(p)
  end function

! CHECK-LABEL: func.func @_QMstorage_size_testPunlimited_polymorphic_allocatable(
! CHECK-SAME: %[[P:.*]]: !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK: %[[pDecl:.*]]:2 = hlfir.declare %[[P]]
! CHECK: %[[LOAD_P:.*]] = fir.load %[[pDecl]]#0 : !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK: %[[P_ADDR:.*]] = fir.box_addr %[[LOAD_P]] : (!fir.class<!fir.heap<none>>) -> !fir.heap<none>
! CHECK: %[[P_ADDR_I64:.*]] = fir.convert %[[P_ADDR]] : (!fir.heap<none>) -> i64
! CHECK: %[[C0:.*]] = arith.constant 0 : i64
! CHECK: %[[IS_NULL_ADDR:.*]] = arith.cmpi eq, %[[P_ADDR_I64]], %[[C0]] : i64
! CHECK: fir.if %[[IS_NULL_ADDR]] {
! CHECK:   fir.call @_FortranAReportFatalUserError
! CHECK: }
! CHECK: %[[LOAD_P2:.*]] = fir.load %[[pDecl]]#0 : !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK: %[[ELE_SIZE:.*]] = fir.box_elesize %[[LOAD_P2]] : (!fir.class<!fir.heap<none>>) -> i32
! CHECK: %[[C8:.*]] = arith.constant 8 : i32
! CHECK: %[[BITS:.*]] = arith.muli %[[ELE_SIZE]], %[[C8]] : i32

  integer function polymorphic_pointer(p) result(size)
    class(p1), pointer :: p
    size = storage_size(p)
  end function

! CHECK-LABEL: func.func @_QMstorage_size_testPpolymorphic_pointer(
! CHECK-SAME: %[[P:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMstorage_size_testTp1{a:i32}>>>>
! CHECK: %[[pDecl:.*]]:2 = hlfir.declare %[[P]]
! CHECK: %[[LOAD_P:.*]] = fir.load %[[pDecl]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMstorage_size_testTp1{a:i32}>>>>
! CHECK: %[[ELE_SIZE:.*]] = fir.box_elesize %[[LOAD_P]] : (!fir.class<!fir.ptr<!fir.type<_QMstorage_size_testTp1{a:i32}>>>) -> i32
! CHECK: %[[C8:.*]] = arith.constant 8 : i32
! CHECK: %[[BITS:.*]] = arith.muli %[[ELE_SIZE]], %[[C8]] : i32

  integer function polymorphic(p) result(size)
    class(p1) :: p
    size = storage_size(p)
  end function

! CHECK-LABEL: func.func @_QMstorage_size_testPpolymorphic(
! CHECK-SAME: %[[P:.*]]: !fir.class<!fir.type<_QMstorage_size_testTp1{a:i32}>>
! CHECK: %[[pDecl:.*]]:2 = hlfir.declare %[[P]]
! CHECK: %[[ELE_SIZE:.*]] = fir.box_elesize %[[pDecl]]#1 : (!fir.class<!fir.type<_QMstorage_size_testTp1{a:i32}>>) -> i32
! CHECK: %[[C8:.*]] = arith.constant 8 : i32
! CHECK: %[[BITS:.*]] = arith.muli %[[ELE_SIZE]], %[[C8]] : i32

  integer(8) function polymorphic_rank(p) result(size)
    class(p1) :: p
    size = storage_size(p, 8)
  end function

! CHECK-LABEL: func.func @_QMstorage_size_testPpolymorphic_rank(
! CHECK-SAME: %[[P:.*]]: !fir.class<!fir.type<_QMstorage_size_testTp1{a:i32}>>
! CHECK: %[[pDecl:.*]]:2 = hlfir.declare %[[P]]
! CHECK: %[[ELE_SIZE:.*]] = fir.box_elesize %[[pDecl]]#1 : (!fir.class<!fir.type<_QMstorage_size_testTp1{a:i32}>>) -> i64
! CHECK: %[[C8:.*]] = arith.constant 8 : i64
! CHECK: %[[BITS:.*]] = arith.muli %[[ELE_SIZE]], %[[C8]] : i64

  integer function polymorphic_value(t) result(size)
    type(p3) :: t
    size = storage_size(t%p(1))
  end function

! CHECK-LABEL: func.func @_QMstorage_size_testPpolymorphic_value(
! CHECK-SAME: %[[T:.*]]: !fir.ref<!fir.type<_QMstorage_size_testTp3{p:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMstorage_size_testTp1{a:i32}>>>>}>>
! CHECK: %[[tDecl:.*]]:2 = hlfir.declare %[[T]]
! CHECK: %[[FIELD_P:.*]] = hlfir.designate %[[tDecl]]#0{"p"}
! CHECK: %[[LOAD_P:.*]] = fir.load %[[FIELD_P]]
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[ELEM:.*]] = hlfir.designate %[[LOAD_P]] (%[[C1]])
! CHECK: %[[ELE_SIZE:.*]] = fir.box_elesize %[[ELEM]] : (!fir.class<!fir.type<_QMstorage_size_testTp1{a:i32}>>) -> i32
! CHECK: %[[C8:.*]] = arith.constant 8 : i32
! CHECK: %[[SIZE:.*]] = arith.muli %[[ELE_SIZE]], %[[C8]] : i32

end module
