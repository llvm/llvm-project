! Test correct deallocation of intent(out) allocatables.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

module mod1
  type, bind(c) :: t1
    integer :: i
  end type

  interface
    subroutine sub3(a) bind(c)
      integer, intent(out), allocatable :: a(:)
    end subroutine
  end interface

  interface
    subroutine sub7(t) bind(c)
      import :: t1
      type(t1), allocatable, intent(out) :: t
    end subroutine
  end interface

contains
  subroutine sub0()
    integer, allocatable :: a(:)
    allocate(a(10))
    call sub1(a)
  end subroutine

  subroutine sub1(a)
    integer, intent(out), allocatable :: a(:)
  end subroutine

! Make sure there is no deallocation of the allocatable intent(out) on the
! caller side.

! CHECK-LABEL: func.func @_QMmod1Psub0()
! CHECK-NOT: fir.freemem
! CHECK: fir.call @_QMmod1Psub1

! Check inline deallocation of allocatable intent(out) on the callee side.

! CHECK-LABEL: func.func @_QMmod1Psub1(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "a"})
! CHECK: %[[BOX:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK: fir.freemem %[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! CHECK: %[[EMBOX:.*]] = fir.embox %[[ZERO]](%[[SHAPE]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK: fir.store %[[EMBOX]] to %[[ARG0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

  subroutine sub2()
    integer, allocatable :: a(:)
    allocate(a(10))
    call sub3(a)
  end subroutine

! Check inlined deallocation of allocatble intent(out) on the caller side for BIND(C).

! CHECK-LABEL: func.func @_QMmod1Psub2()
! CHECK: %[[BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "a", uniq_name = "_QMmod1Fsub2Ea"}
! CHECK: %[[BOX_ALLOC:.*]] = fir.alloca !fir.heap<!fir.array<?xi32>> {uniq_name = "_QMmod1Fsub2Ea.addr"}
! CHECK: %[[LOAD:.*]] = fir.load %[[BOX_ALLOC]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK: %{{.*}} = fir.embox %[[LOAD]](%{{.*}}) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<
! CHECK: %[[LOAD:.*]] = fir.load %[[BOX_ALLOC]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK: fir.freemem %[[LOAD]] : !fir.heap<!fir.array<?xi32>>
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK: fir.store %[[ZERO]] to %[[BOX_ALLOC]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK: fir.call @sub3(%[[BOX]]) : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> ()

  subroutine sub4()
    type(t1), allocatable :: t
    call sub5(t)
  end subroutine

  subroutine sub5(t)
    type(t1), allocatable, intent(out) :: t
  end subroutine

! Make sure there is no deallocation runtime call of the allocatable intent(out)
! on the caller side.

! CHECK-LABEL: func.func @_QMmod1Psub4()
! CHECK: %[[BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>> {bindc_name = "t", uniq_name = "_QMmod1Fsub4Et"}
! CHECK-NOT: fir.call @_FortranAAllocatableDeallocate
! CHECK: fir.call @_QMmod1Psub5(%[[BOX]]) : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>>>) -> ()

! Check deallocation of allocatble intent(out) on the callee side. Deallocation
! is done with a runtime call.

! CHECK-LABEL: func.func @_QMmod1Psub5(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>>> {fir.bindc_name = "t"})
! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[ARG0]] : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableDeallocate(%[[BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

  subroutine sub6()
    type(t1), allocatable :: t
    call sub7(t)
  end subroutine

! Check deallocation of allocatble intent(out) on the caller side for BIND(C).
! Deallocation is done with a runtime call.

! CHECK-LABEL: func.func @_QMmod1Psub6()
! CHECK: %[[BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>> {bindc_name = "t", uniq_name = "_QMmod1Fsub6Et"}
! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[BOX]] : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableDeallocate(%[[BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK: fir.call @sub7(%[[BOX]]) : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>>>) -> ()

  subroutine sub8()
    integer, allocatable :: a(:)
    allocate(a(10))
    call sub9(a)
  end subroutine

  subroutine sub9(a)
    integer, intent(out), allocatable, optional :: a(:)
  end subroutine

! Make sure there is no deallocation of the allocatable intent(out) on the
! caller side.

! CHECK-LABEL: func.func @_QMmod1Psub8()
! CHECK-NOT: fir.freemem
! CHECK: fir.call @_QMmod1Psub9

! Check inline deallocation of optional allocatable intent(out) on the callee side.

! CHECK-LABEL: func.func @_QMmod1Psub9(
! CHECK-SAME:  %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "a", fir.optional})
! CHECK: %[[IS_PRESENT:.*]] = fir.is_present %[[ARG0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> i1
! CHECK: fir.if %[[IS_PRESENT]] {
! CHECK:   %[[BOX:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:   %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:   fir.freemem %[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>
! CHECK:   %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:   %[[C0:.*]] = arith.constant 0 : index
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! CHECK:   %[[EMBOX:.*]] = fir.embox %[[ZERO]](%[[SHAPE]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:   fir.store %[[EMBOX]] to %[[ARG0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: }

  subroutine sub10(a)
    integer, intent(out), allocatable :: a(:)

  entry sub11
  end subroutine

! CHECK-LABEL: func.func @_QMmod1Psub10(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "a"}) {
! CHECK: %[[LOAD:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK: fir.freemem %[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! CHECK: %[[EMBOX:.*]] = fir.embox %[[ZERO]](%[[SHAPE]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK: fir.store %[[EMBOX]] to %[[ARG0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

! CHECK-LABEL: func.func @_QMmod1Psub11() {
! CHECK-NOT: fir.freemem

  subroutine sub12(a)
    integer, intent(out), allocatable :: a(:)
  entry sub13(a)
  end subroutine

! CHECK-LABEL: func.func @_QMmod1Psub12(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "a"}) {
! CHECK: %[[LOAD:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK: fir.freemem %[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! CHECK: %[[EMBOX:.*]] = fir.embox %[[ZERO]](%[[SHAPE]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK: fir.store %[[EMBOX]] to %[[ARG0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

! CHECK-LABEL: func.func @_QMmod1Psub13(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "a"}) {
! CHECK: %[[LOAD:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK: fir.freemem %[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! CHECK: %[[EMBOX:.*]] = fir.embox %[[ZERO]](%[[SHAPE]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK: fir.store %[[EMBOX]] to %[[ARG0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>


end module

