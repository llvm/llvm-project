! Test correct deallocation of intent(out) allocatables.
! RUN: bbc --use-desc-for-alloc=false -emit-fir -polymorphic-type %s -o - | FileCheck %s --check-prefixes=CHECK,FIR
! RUN: bbc -emit-hlfir -polymorphic-type %s -o - -I nw | FileCheck %s --check-prefixes=CHECK,HLFIR

module mod1
  type, bind(c) :: t1
    integer :: i
  end type

  type :: t
    integer :: a
  end type

  type, extends(t) :: t2
    integer :: b
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
! FIR-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "a"})
! HLFIR: %[[ARG0:.*]]:2 = hlfir.declare {{.*}}"_QMmod1Fsub1Ea"
! CHECK: %[[BOX:.*]] = fir.load %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK: %[[BOX_ADDR_PTR:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK: %[[C0:.*]] = arith.constant 0 : i64
! CHECK: %[[IS_ALLOCATED:.*]] = arith.cmpi ne, %[[BOX_ADDR_PTR]], %[[C0]] : i64
! CHECK: fir.if %[[IS_ALLOCATED]] {
! CHECK:   %[[BOX:.*]] = fir.load %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:   %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:   fir.freemem %[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>
! CHECK:   %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:   %[[C0:.*]] = arith.constant 0 : index
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! CHECK:   %[[EMBOX:.*]] = fir.embox %[[ZERO]](%[[SHAPE]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:   fir.store %[[EMBOX]] to %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: }

  subroutine sub2()
    integer, allocatable :: a(:)
    allocate(a(10))
    call sub3(a)
  end subroutine

! Check inlined deallocation of allocatble intent(out) on the caller side for BIND(C).

! FIR-LABEL: func.func @_QMmod1Psub2()
! FIR: %[[BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "a", uniq_name = "_QMmod1Fsub2Ea"}
! FIR: %[[BOX_ALLOC:.*]] = fir.alloca !fir.heap<!fir.array<?xi32>> {uniq_name = "_QMmod1Fsub2Ea.addr"}
! FIR: %[[BOX_ADDR:.*]] = fir.load %[[BOX_ALLOC]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! FIR: %[[BOX_ADDR_PTR:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! FIR: %[[C0:.*]] = arith.constant 0 : i64
! FIR: %[[IS_ALLOCATED:.*]] = arith.cmpi ne, %[[BOX_ADDR_PTR]], %[[C0]] : i64
! FIR: fir.if %[[IS_ALLOCATED]] {
! FIR:   %[[LOAD:.*]] = fir.load %[[BOX_ALLOC]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! FIR:   fir.freemem %[[LOAD]] : !fir.heap<!fir.array<?xi32>>
! FIR:   %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! FIR:   fir.store %[[ZERO]] to %[[BOX_ALLOC]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! FIR: }
! FIR: %[[LOAD:.*]] = fir.load %[[BOX_ALLOC]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! FIR: %{{.*}} = fir.embox %[[LOAD]](%{{.*}}) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<
! FIR: fir.call @sub3(%[[BOX]]) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> ()

! HLFIR-LABEL: func.func @_QMmod1Psub2(
! HLFIR: %[[ARG0:.*]]:2 = hlfir.declare {{.*}}"_QMmod1Fsub2Ea"
! HLFIR: %[[BOX:.*]] = fir.load %[[ARG0]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! HLFIR: %[[BOX_ADDR_PTR:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! HLFIR: %[[C0:.*]] = arith.constant 0 : i64
! HLFIR: %[[IS_ALLOCATED:.*]] = arith.cmpi ne, %[[BOX_ADDR_PTR]], %[[C0]] : i64
! HLFIR: fir.if %[[IS_ALLOCATED]] {
! HLFIR:   %[[BOX:.*]] = fir.load %[[ARG0]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR:   %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! HLFIR:   fir.freemem %[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>
! HLFIR:   %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! HLFIR:   %[[C0:.*]] = arith.constant 0 : index
! HLFIR:   %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! HLFIR:   %[[EMBOX:.*]] = fir.embox %[[ZERO]](%[[SHAPE]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! HLFIR:   fir.store %[[EMBOX]] to %[[ARG0]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR: fir.call @sub3(%[[ARG0]]#0) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> ()

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
! FIR: %[[BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>> {bindc_name = "t", uniq_name = "_QMmod1Fsub4Et"}
! HLFIR: %[[BOX:.*]]:2 = hlfir.declare {{.*}}"_QMmod1Fsub4Et"
! CHECK-NOT: fir.call @_FortranAAllocatableDeallocate
! CHECK: fir.call @_QMmod1Psub5(%[[BOX]]{{[#0]*}}) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>>>) -> ()

! Check deallocation of allocatble intent(out) on the callee side. Deallocation
! is done with a runtime call.

! CHECK-LABEL: func.func @_QMmod1Psub5(
! FIR-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>>> {fir.bindc_name = "t"})
! HLFIR: %[[ARG0:.*]]:2 = hlfir.declare {{.*}}"_QMmod1Fsub5Et"
! CHECK: %[[BOX:.*]] = fir.load %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>>>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>>) -> !fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>
! CHECK: %[[BOX_ADDR_PTR:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>) -> i64
! CHECK: %[[C0:.*]] = arith.constant 0 : i64
! CHECK: %[[IS_ALLOCATED:.*]] = arith.cmpi ne, %[[BOX_ADDR_PTR]], %[[C0]] : i64
! CHECK: fir.if %[[IS_ALLOCATED]] {
! CHECK:   %[[BOX_NONE:.*]] = fir.convert %[[ARG0]]{{[#1]*}} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:   %{{.*}} = fir.call @_FortranAAllocatableDeallocate(%[[BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

  subroutine sub6()
    type(t1), allocatable :: t
    call sub7(t)
  end subroutine

! Check deallocation of allocatble intent(out) on the caller side for BIND(C).
! Deallocation is done with a runtime call.

! CHECK-LABEL: func.func @_QMmod1Psub6()
! FIR: %[[BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>> {bindc_name = "t", uniq_name = "_QMmod1Fsub6Et"}
! HLFIR: %[[BOX:.*]]:2 = hlfir.declare {{.*}}"_QMmod1Fsub6Et"
! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[BOX]]{{[#1]*}} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableDeallocate(%[[BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK: fir.call @sub7(%[[BOX]]{{[#0]*}}) {{.*}}: (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMmod1Tt1{i:i32}>>>>) -> ()

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
! FIR-SAME:  %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "a", fir.optional})
! HLFIR: %[[ARG0:.*]]:2 = hlfir.declare {{.*}}"_QMmod1Fsub9Ea"
! CHECK: %[[IS_PRESENT:.*]] = fir.is_present %[[ARG0]]{{[#1]*}} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> i1
! CHECK: fir.if %[[IS_PRESENT]] {
! CHECK:   %[[BOX:.*]] = fir.load %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:   %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:   %[[BOX_ADDR_PTR:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK:   %[[C0:.*]] = arith.constant 0 : i64
! CHECK:   %[[IS_ALLOCATED:.*]] = arith.cmpi ne, %[[BOX_ADDR_PTR]], %[[C0]] : i64
! CHECK:   fir.if %[[IS_ALLOCATED]] {
! CHECK:     %[[BOX:.*]] = fir.load %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:     %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:     fir.freemem %[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>
! CHECK:     %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:     %[[C0:.*]] = arith.constant 0 : index
! CHECK:     %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! CHECK:     %[[EMBOX:.*]] = fir.embox %[[ZERO]](%[[SHAPE]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:     fir.store %[[EMBOX]] to %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:   }
! CHECK: }

  subroutine sub10(a)
    integer, intent(out), allocatable :: a(:)

  entry sub11
  end subroutine

! CHECK-LABEL: func.func @_QMmod1Psub10(
! FIR-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "a"}) {
! HLFIR: %[[ARG0:.*]]:2 = hlfir.declare {{.*}}"_QMmod1Fsub10Ea"
! CHECK: %[[LOAD:.*]] = fir.load %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK: %[[BOX_ADDR_PTR:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK: %[[C0:.*]] = arith.constant 0 : i64
! CHECK: %[[IS_ALLOCATED:.*]] = arith.cmpi ne, %[[BOX_ADDR_PTR]], %[[C0]] : i64
! CHECK: fir.if %[[IS_ALLOCATED]] {
! CHECK:   %[[BOX:.*]] = fir.load %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:   %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:   fir.freemem %[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>
! CHECK:   %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:   %[[C0:.*]] = arith.constant 0 : index
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! CHECK:   %[[EMBOX:.*]] = fir.embox %[[ZERO]](%[[SHAPE]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:   fir.store %[[EMBOX]] to %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: }

! CHECK-LABEL: func.func @_QMmod1Psub11() {
! CHECK-NOT: fir.freemem

  subroutine sub12(a)
    integer, intent(out), allocatable :: a(:)
  entry sub13(a)
  end subroutine

! CHECK-LABEL: func.func @_QMmod1Psub12(
! FIR-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "a"}) {
! HLFIR: %[[ARG0:.*]]:2 = hlfir.declare {{.*}}"_QMmod1Fsub12Ea"
! CHECK: %[[LOAD:.*]] = fir.load %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK: %[[BOX_ADDR_PTR:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK: %[[C0:.*]] = arith.constant 0 : i64
! CHECK: %[[IS_ALLOCATED:.*]] = arith.cmpi ne, %[[BOX_ADDR_PTR]], %[[C0]] : i64
! CHECK: fir.if %[[IS_ALLOCATED]] {
! CHECK:   %[[BOX:.*]] = fir.load %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:   %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:   fir.freemem %[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>
! CHECK:   %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:   %[[C0:.*]] = arith.constant 0 : index
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! CHECK:   %[[EMBOX:.*]] = fir.embox %[[ZERO]](%[[SHAPE]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:   fir.store %[[EMBOX]] to %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: }

! CHECK-LABEL: func.func @_QMmod1Psub13(
! FIR-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "a"}) {
! HLFIR: %[[ARG0:.*]]:2 = hlfir.declare {{.*}}"_QMmod1Fsub12Ea"
! CHECK: %[[LOAD:.*]] = fir.load %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK: %[[BOX_ADDR_PTR:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK: %[[C0:.*]] = arith.constant 0 : i64
! CHECK: %[[IS_ALLOCATED:.*]] = arith.cmpi ne, %[[BOX_ADDR_PTR]], %[[C0]] : i64
! CHECK: fir.if %[[IS_ALLOCATED]] {
! CHECK:   %[[BOX:.*]] = fir.load %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:   %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:   fir.freemem %[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>
! CHECK:   %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:   %[[C0:.*]] = arith.constant 0 : index
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! CHECK:   %[[EMBOX:.*]] = fir.embox %[[ZERO]](%[[SHAPE]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:   fir.store %[[EMBOX]] to %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: }


  subroutine sub14(p)
    class(t), intent(out), allocatable :: p
  end subroutine

! CHECK-LABEL: func.func @_QMmod1Psub14(
! FIR-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.type<_QMmod1Tt{a:i32}>>>> {fir.bindc_name = "p"}) {
! HLFIR: %[[ARG0:.*]]:2 = hlfir.declare {{.*}}"_QMmod1Fsub14Ep"
! CHECK: %[[BOX:.*]] = fir.load %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMmod1Tt{a:i32}>>>>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.class<!fir.heap<!fir.type<_QMmod1Tt{a:i32}>>>) -> !fir.heap<!fir.type<_QMmod1Tt{a:i32}>>
! CHECK: %[[BOX_ADDR_PTR:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.heap<!fir.type<_QMmod1Tt{a:i32}>>) -> i64
! CHECK: %[[C0:.*]] = arith.constant 0 : i64
! CHECK: %[[IS_ALLOCATED:.*]] = arith.cmpi ne, %[[BOX_ADDR_PTR]], %[[C0]] : i64
! CHECK: fir.if %[[IS_ALLOCATED]] {
! CHECK:   %[[TYPE_DESC:.*]] = fir.type_desc !fir.type<_QMmod1Tt{a:i32}>
! CHECK:   %[[BOX_NONE:.*]] = fir.convert %[[ARG0]]{{[#1]*}} : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMmod1Tt{a:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:   %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC]] : (!fir.tdesc<!fir.type<_QMmod1Tt{a:i32}>>) -> !fir.ref<none>
! CHECK:   %{{.*}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%[[BOX_NONE]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK: }

  subroutine sub15(p)
    class(*), intent(out), allocatable :: p
  end subroutine

! CHECK-LABEL: func.func @_QMmod1Psub15(
! FIR-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.heap<none>>> {fir.bindc_name = "p"}) {
! HLFIR: %[[ARG0:.*]]:2 = hlfir.declare {{.*}}"_QMmod1Fsub15Ep"
! CHECK: %[[BOX:.*]] = fir.load %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.class<!fir.heap<none>>) -> !fir.heap<none>
! CHECK: %[[BOX_ADDR_PTR:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.heap<none>) -> i64
! CHECK: %[[C0:.*]] = arith.constant 0 : i64
! CHECK: %[[IS_ALLOCATED:.*]] = arith.cmpi ne, %[[BOX_ADDR_PTR]], %[[C0]] : i64
! CHECK: fir.if %[[IS_ALLOCATED]] {
! CHECK:   %[[NULL_TYPE_DESC:.*]] = fir.zero_bits !fir.ref<none>
! CHECK:   %[[BOX_NONE:.*]] = fir.convert %[[ARG0]]{{[#1]*}} : (!fir.ref<!fir.class<!fir.heap<none>>>) -> !fir.ref<!fir.box<none>>
! CHECK:   %{{.*}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%[[BOX_NONE]], %[[NULL_TYPE_DESC]], %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK: }

  subroutine sub16(p)
    class(t), optional, intent(out), allocatable :: p
  end subroutine

! CHECK-LABEL: func.func @_QMmod1Psub16(
! FIR-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.type<_QMmod1Tt{a:i32}>>>> {fir.bindc_name = "p", fir.optional}) {
! HLFIR: %[[ARG0:.*]]:2 = hlfir.declare {{.*}}"_QMmod1Fsub16Ep"
! CHECK: %[[IS_PRESENT:.*]] = fir.is_present %[[ARG0]]{{[#1]*}} : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMmod1Tt{a:i32}>>>>) -> i1
! CHECK: fir.if %[[IS_PRESENT]] {
! CHECK:   %[[BOX:.*]] = fir.load %[[ARG0]]{{[#1]*}} : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMmod1Tt{a:i32}>>>>
! CHECK:   %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.class<!fir.heap<!fir.type<_QMmod1Tt{a:i32}>>>) -> !fir.heap<!fir.type<_QMmod1Tt{a:i32}>>
! CHECK:   %[[BOX_ADDR_PTR:.*]] = fir.convert %[[BOX_ADDR]] : (!fir.heap<!fir.type<_QMmod1Tt{a:i32}>>) -> i64
! CHECK:   %[[C0:.*]] = arith.constant 0 : i64
! CHECK:   %[[IS_ALLOCATED:.*]] = arith.cmpi ne, %[[BOX_ADDR_PTR]], %[[C0]] : i64
! CHECK:   fir.if %[[IS_ALLOCATED]] {
! CHECK:     %[[TYPE_DESC:.*]] = fir.type_desc !fir.type<_QMmod1Tt{a:i32}>
! CHECK:     %[[BOX_NONE:.*]] = fir.convert %[[ARG0]]{{[#1]*}} : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMmod1Tt{a:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:     %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC]] : (!fir.tdesc<!fir.type<_QMmod1Tt{a:i32}>>) -> !fir.ref<none>
! CHECK:     %{{.*}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%[[BOX_NONE]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:   }
! CHECK: }

end module
