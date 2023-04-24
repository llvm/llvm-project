! RUN: bbc --use-desc-for-alloc=false -polymorphic-type -emit-fir %s -o - | FileCheck %s
! RUN: bbc --use-desc-for-alloc=false -polymorphic-type -emit-fir %s -o - | tco | FileCheck %s --check-prefix=LLVM

module poly
  type p1
    integer :: a
    integer :: b
  contains
    procedure, nopass :: proc1 => proc1_p1
    procedure :: proc2 => proc2_p1
  end type

  type, extends(p1) :: p2
    integer :: c
  contains
    procedure, nopass :: proc1 => proc1_p2
    procedure :: proc2 => proc2_p2
  end type

  type with_alloc
    class(p1), pointer :: element
  end type

contains
  subroutine proc1_p1()
    print*, 'call proc1_p1'
  end subroutine

  subroutine proc1_p2()
    print*, 'call proc1_p2'
  end subroutine

  subroutine proc2_p1(this)
    class(p1) :: this
    print*, 'call proc2_p1'
  end subroutine

  subroutine proc2_p2(this)
    class(p2) :: this
    print*, 'call proc2_p2'
  end subroutine


! ------------------------------------------------------------------------------
! Test lowering of ALLOCATE statement for polymoprhic pointer
! ------------------------------------------------------------------------------

  subroutine test_pointer()
    class(p1), pointer :: p
    class(p1), pointer :: c1, c2
    class(p1), pointer, dimension(:) :: c3, c4
    integer :: i

    print*, '---------------------------------------'
    print*, 'test allocation of polymorphic pointers'
    print*, '---------------------------------------'

    allocate(p)
    call p%proc1()

    allocate(p1::c1)
    allocate(p2::c2)

    call c1%proc1()
    call c2%proc1()

    call c1%proc2()
    call c2%proc2()

    allocate(p1::c3(10))
    allocate(p2::c4(20))

    do i = 1, 10
      call c3(i)%proc2()
    end do

    do i = 1, 20
      call c4(i)%proc2()
    end do

    deallocate(p)
    deallocate(c1)
    deallocate(c2)
    deallocate(c3)
    deallocate(c4)

  end subroutine

! CHECK-LABEL: func.func @_QMpolyPtest_pointer()
! CHECK: %[[C1_DESC:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>> {bindc_name = "c1", uniq_name = "_QMpolyFtest_pointerEc1"}
! CHECK: %[[C2_DESC:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>> {bindc_name = "c2", uniq_name = "_QMpolyFtest_pointerEc2"}
! CHECK: %[[C3_DESC:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>> {bindc_name = "c3", uniq_name = "_QMpolyFtest_pointerEc3"}
! CHECK: %[[C4_DESC:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>> {bindc_name = "c4", uniq_name = "_QMpolyFtest_pointerEc4"}
! CHECK: %[[P_DESC:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>> {bindc_name = "p", uniq_name = "_QMpolyFtest_pointerEp"}

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}>
! CHECK: %[[P_DESC_CAST:.*]] = fir.convert %[[P_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[CORANK:.*]] = arith.constant 0 : i32
! CHECK: %{{.*}} = fir.call @_FortranAPointerNullifyDerived(%[[P_DESC_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[CORANK]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[P_DESC_CAST:.*]] = fir.convert %[[P_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAllocate(%[[P_DESC_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! call p%proc1()
! CHECK: %[[P_LOAD:.*]] = fir.load %[[P_DESC:.*]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: fir.dispatch "proc1"(%[[P_LOAD]] : !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>)

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}>
! CHECK: %[[C1_DESC_CAST:.*]] = fir.convert %[[C1_DESC:.*]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[CORANK:.*]] = arith.constant 0 : i32
! CHECK: %{{.*}} = fir.call @_FortranAPointerNullifyDerived(%[[C1_DESC_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[CORANK]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C1_DESC_CAST:.*]] = fir.convert %[[C1_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAllocate(%[[C1_DESC_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P2:.*]] = fir.type_desc !fir.type<_QMpolyTp2{a:i32,b:i32,c:i32}>
! CHECK: %[[C2_DESC_CAST:.*]] = fir.convert %[[C2_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P2_CAST:.*]] = fir.convert %[[TYPE_DESC_P2]] : (!fir.tdesc<!fir.type<_QMpolyTp2{a:i32,b:i32,c:i32}>>) -> !fir.ref<none>
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[CORANK:.*]] = arith.constant 0 : i32
! CHECK: %{{.*}} = fir.call @_FortranAPointerNullifyDerived(%[[C2_DESC_CAST]], %[[TYPE_DESC_P2_CAST]], %[[RANK]], %[[CORANK]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C2_DESC_CAST:.*]] = fir.convert %[[C2_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAllocate(%[[C2_DESC_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! call c1%proc1()
! CHECK: %[[C1_DESC_LOAD:.*]] = fir.load %[[C1_DESC]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: fir.dispatch "proc1"(%[[C1_DESC_LOAD]] : !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>)

! call c2%proc1()
! CHECK: %[[C2_DESC_LOAD:.*]] = fir.load %[[C2_DESC]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: fir.dispatch "proc1"(%[[C2_DESC_LOAD]] : !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>)

! call c1%proc2()
! CHECK: %[[C1_LOAD:.*]] = fir.load %[[C1_DESC]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[C1_REBOX:.*]] = fir.rebox %[[C1_LOAD]] : (!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc2"(%[[C1_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[C1_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! call c2%proc2()
! CHECK: %[[C2_LOAD:.*]] = fir.load %[[C2_DESC]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[C2_REBOX:.*]] = fir.rebox %[[C2_LOAD]] : (!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc2"(%[[C2_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[C2_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}>
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %[[RANK:.*]] = arith.constant 1 : i32
! CHECK: %[[CORANK:.*]] = arith.constant 0 : i32
! CHECK: %{{.*}} = fir.call @_FortranAPointerNullifyDerived(%[[C3_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[CORANK]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerSetBounds(%[[C3_CAST]], %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAllocate(%[[C3_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P2:.*]] = fir.type_desc !fir.type<_QMpolyTp2{a:i32,b:i32,c:i32}>
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P2_CAST:.*]] = fir.convert %[[TYPE_DESC_P2]] : (!fir.tdesc<!fir.type<_QMpolyTp2{a:i32,b:i32,c:i32}>>) -> !fir.ref<none>
! CHECK: %[[RANK:.*]] = arith.constant 1 : i32
! CHECK: %[[CORANK:.*]] = arith.constant 0 : i32
! CHECK: %{{.*}} = fir.call @_FortranAPointerNullifyDerived(%[[C4_CAST]], %[[TYPE_DESC_P2_CAST]], %[[RANK]], %[[CORANK]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerSetBounds(%[[C4_CAST]], %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAllocate(%[[C4_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK-LABEL: fir.do_loop
! CHECK: %[[C3_LOAD:.*]] = fir.load %[[C3_DESC]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C3_COORD:.*]] = fir.coordinate_of %[[C3_LOAD]], %{{.*}} : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, i64) -> !fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[C3_BOXED:.*]] = fir.embox %[[C3_COORD]] source_box %[[C3_LOAD]] : (!fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>, !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc2"(%[[C3_BOXED]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[C3_BOXED]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK-LABEL: fir.do_loop
! CHECK: %[[C4_LOAD:.*]] = fir.load %[[C4_DESC]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C4_COORD:.*]] = fir.coordinate_of %[[C4_LOAD]], %{{.*}} : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, i64) -> !fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[C4_BOXED:.*]] = fir.embox %[[C4_COORD]] source_box %[[C4_LOAD]] : (!fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>, !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc2"(%[[C4_BOXED]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[C4_BOXED]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}


! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}>
! CHECK: %[[P_CAST:.*]] = fir.convert %[[P_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerDeallocatePolymorphic(%[[P_CAST]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}>
! CHECK: %[[C1_DESC_CAST:.*]] = fir.convert %[[C1_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerDeallocatePolymorphic(%[[C1_DESC_CAST]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}>
! CHECK: %[[C2_DESC_CAST:.*]] = fir.convert %[[C2_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerDeallocatePolymorphic(%[[C2_DESC_CAST]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}>
! CHECK: %[[C3_DESC_CAST:.*]] = fir.convert %[[C3_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerDeallocatePolymorphic(%[[C3_DESC_CAST]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}>
! CHECK: %[[C4_DESC_CAST:.*]] = fir.convert %[[C4_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerDeallocatePolymorphic(%[[C4_DESC_CAST]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! ------------------------------------------------------------------------------
! Test lowering of ALLOCATE statement for polymoprhic allocatable
! ------------------------------------------------------------------------------

  subroutine test_allocatable()
    class(p1), allocatable :: p
    class(p1), allocatable :: c1, c2
    class(p1), allocatable, dimension(:) :: c3, c4
    integer :: i

    print*, '------------------------------------------'
    print*, 'test allocation of polymorphic allocatable'
    print*, '------------------------------------------'

    allocate(p) ! allocate as p1

    allocate(p1::c1)
    allocate(p2::c2)

    allocate(p1::c3(10))
    allocate(p2::c4(20))

    call c1%proc1()
    call c2%proc1()

    call c1%proc2()
    call c2%proc2()

    do i = 1, 10
      call c3(i)%proc2()
    end do

    do i = 1, 20
      call c4(i)%proc2()
    end do

    deallocate(p)
    deallocate(c1)
    deallocate(c2)
    deallocate(c3)
    deallocate(c4)
  end subroutine

! CHECK-LABEL: func.func @_QMpolyPtest_allocatable()

! CHECK-DAG: %[[C1:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>> {bindc_name = "c1", uniq_name = "_QMpolyFtest_allocatableEc1"}
! CHECK-DAG: %[[C2:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>> {bindc_name = "c2", uniq_name = "_QMpolyFtest_allocatableEc2"}
! CHECK-DAG: %[[C3:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>> {bindc_name = "c3", uniq_name = "_QMpolyFtest_allocatableEc3"}
! CHECK-DAG: %[[C4:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>> {bindc_name = "c4", uniq_name = "_QMpolyFtest_allocatableEc4"}
! CHECK-DAG: %[[P:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>> {bindc_name = "p", uniq_name = "_QMpolyFtest_allocatableEp"}

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}>
! CHECK: %[[P_CAST:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerivedForAllocate(%[[P_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[C0]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[P_CAST:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[P_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}>
! CHECK: %[[C1_CAST:.*]] = fir.convert %0 : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerivedForAllocate(%[[C1_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[C0]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C1_CAST:.*]] = fir.convert %[[C1]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[C1_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P2:.*]] = fir.type_desc !fir.type<_QMpolyTp2{a:i32,b:i32,c:i32}>
! CHECK: %[[C2_CAST:.*]] = fir.convert %[[C2]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P2_CAST:.*]] = fir.convert %[[TYPE_DESC_P2]] : (!fir.tdesc<!fir.type<_QMpolyTp2{a:i32,b:i32,c:i32}>>) -> !fir.ref<none>
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerivedForAllocate(%[[C2_CAST]], %[[TYPE_DESC_P2_CAST]], %[[RANK]], %[[C0]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C2_CAST:.*]] = fir.convert %[[C2]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[C2_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}>
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %[[RANK:.*]] = arith.constant 1 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerivedForAllocate(%[[C3_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[C0]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C10:.*]] = arith.constant 10 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[C1_I64:.*]] = fir.convert %c1 : (index) -> i64
! CHECK: %[[C10_I64:.*]] = fir.convert %[[C10]] : (i32) -> i64
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableSetBounds(%[[C3_CAST]], %[[C0]], %[[C1_I64]], %[[C10_I64]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[C3_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P2:.*]] = fir.type_desc !fir.type<_QMpolyTp2{a:i32,b:i32,c:i32}>
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P2_CAST:.*]] = fir.convert %[[TYPE_DESC_P2]] : (!fir.tdesc<!fir.type<_QMpolyTp2{a:i32,b:i32,c:i32}>>) -> !fir.ref<none>
! CHECK: %[[RANK:.*]] = arith.constant 1 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerivedForAllocate(%[[C4_CAST]], %[[TYPE_DESC_P2_CAST]], %[[RANK]], %[[C0]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[CST1:.*]] = arith.constant 1 : index
! CHECK: %[[C20:.*]] = arith.constant 20 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[C1_I64:.*]] = fir.convert %[[CST1]] : (index) -> i64
! CHECK: %[[C20_I64:.*]] = fir.convert %[[C20]] : (i32) -> i64
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableSetBounds(%[[C4_CAST]], %[[C0]], %[[C1_I64]], %[[C20_I64]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[C4_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[C1_LOAD1:.*]] = fir.load %[[C1_DESC]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: fir.dispatch "proc1"(%[[C1_LOAD1]] : !fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>)

! CHECK: %[[C2_LOAD1:.*]] = fir.load %[[C2_DESC]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: fir.dispatch "proc1"(%[[C2_LOAD1]] : !fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>)

! CHECK: %[[C1_LOAD2:.*]] = fir.load %[[C1_DESC]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[C1_REBOX:.*]] = fir.rebox %[[C1_LOAD2]] : (!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc2"(%[[C1_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[C1_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK: %[[C2_LOAD2:.*]] = fir.load %[[C2_DESC]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[C2_REBOX:.*]] = fir.rebox %[[C2_LOAD2]] : (!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc2"(%[[C2_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[C2_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK-LABEL: %{{.*}} = fir.do_loop
! CHECK: %[[C3_LOAD:.*]] = fir.load %[[C3_DESC]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C3_COORD:.*]] = fir.coordinate_of %[[C3_LOAD]], %{{.*}} : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, i64) -> !fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[C3_EMBOX:.*]] = fir.embox %[[C3_COORD]] source_box %[[C3_LOAD]] : (!fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc2"(%[[C3_EMBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[C3_EMBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK-LABEL: %{{.*}} = fir.do_loop
! CHECK: %[[C4_LOAD:.*]] = fir.load %[[C4]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C4_COORD:.*]] = fir.coordinate_of %[[C4_LOAD]], %{{.*}} : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, i64) -> !fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[C4_EMBOX:.*]] = fir.embox %[[C4_COORD]] source_box %[[C4_LOAD]] : (!fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc2"(%[[C4_EMBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[C4_EMBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}>
! CHECK: %[[P_CAST:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%[[P_CAST]], %[[TYPE_NONE]], %{{.*}}, %1{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}>
! CHECK: %[[C1_CAST:.*]] = fir.convert %[[C1]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%[[C1_CAST]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}>
! CHECK: %[[C2_CAST:.*]] = fir.convert %[[C2]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%[[C2_CAST]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}>
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%[[C3_CAST]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}>
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%[[C4_CAST]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

  subroutine test_unlimited_polymorphic_with_intrinsic_type_spec()
    class(*), allocatable :: p
    class(*), pointer :: ptr
    allocate(integer::p)
    deallocate(p)

    allocate(real::ptr)
    deallocate(ptr)

  end subroutine

! CHECK-LABEL: func.func @_QMpolyPtest_unlimited_polymorphic_with_intrinsic_type_spec() {
! CHECK: %[[P:.*]] = fir.alloca !fir.class<!fir.heap<none>> {bindc_name = "p", uniq_name = "_QMpolyFtest_unlimited_polymorphic_with_intrinsic_type_specEp"}
! CHECK: %[[PTR:.*]] = fir.alloca !fir.class<!fir.ptr<none>> {bindc_name = "ptr", uniq_name = "_QMpolyFtest_unlimited_polymorphic_with_intrinsic_type_specEptr"}
! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.heap<none>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[CAT:.*]] = arith.constant 0 : i32
! CHECK: %[[KIND:.*]] = arith.constant 4 : i32
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[CORANK:.*]] = arith.constant 0 : i32
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableInitIntrinsicForAllocate(%[[BOX_NONE]], %[[CAT]], %[[KIND]], %[[RANK]], %[[CORANK]]) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, i32, i32) -> none
! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.heap<none>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[PTR]] : (!fir.ref<!fir.class<!fir.ptr<none>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[CAT:.*]] = arith.constant 1 : i32
! CHECK: %[[KIND:.*]] = arith.constant 4 : i32
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[CORANK:.*]] = arith.constant 0 : i32
! CHECK: %{{.*}} = fir.call @_FortranAPointerNullifyIntrinsic(%[[BOX_NONE]], %[[CAT]], %[[KIND]], %[[RANK]], %[[CORANK]]) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, i32, i32) -> none
! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[PTR]] : (!fir.ref<!fir.class<!fir.ptr<none>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAllocate(%[[BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[NULL_TYPE_DESC:.*]] = fir.zero_bits !fir.ref<none>
! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[PTR]] : (!fir.ref<!fir.class<!fir.ptr<none>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerDeallocatePolymorphic(%[[BOX_NONE]], %[[NULL_TYPE_DESC]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

  ! Test code generation of deallocate
  subroutine test_deallocate()
    class(p1), allocatable :: p

    allocate(p)
    deallocate(p)
  end subroutine

  subroutine test_type_with_polymorphic_pointer_component()
    type(with_alloc), pointer :: a
    allocate(a)
    allocate(a%element)
  end subroutine

! CHECK-LABEL: func.func @_QMpolyPtest_type_with_polymorphic_pointer_component()
! CHECK: %[[TYPE_PTR:.*]] = fir.alloca !fir.ptr<!fir.type<_QMpolyTwith_alloc{element:!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>}>> {uniq_name = "_QMpolyFtest_type_with_polymorphic_pointer_componentEa.addr"}
! CHECK: %[[TYPE_PTR_LOAD:.*]] = fir.load %[[TYPE_PTR]] : !fir.ref<!fir.ptr<!fir.type<_QMpolyTwith_alloc{element:!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>}>>>
! CHECK: %[[ELEMENT:.*]] = fir.field_index element, !fir.type<_QMpolyTwith_alloc{element:!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>}>
! CHECK: %[[ELEMENT_DESC:.*]] = fir.coordinate_of %[[TYPE_PTR_LOAD]], %[[ELEMENT]] : (!fir.ptr<!fir.type<_QMpolyTwith_alloc{element:!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>}>>, !fir.field) -> !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[ZERO_DESC:.*]] = fir.embox %[[ZERO]] : (!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>
! CHECK: fir.store %[[ZERO_DESC]] to %[[ELEMENT_DESC]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[TYPE_DESC_P1:.*]] = fir.type_desc !fir.type<_QMpolyTp1{a:i32,b:i32}>
! CHECK: %[[ELEMENT_DESC_CAST:.*]] = fir.convert %[[ELEMENT_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.ref<none>
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[CORANK:.*]] = arith.constant 0 : i32
! CHECK: %{{.*}} = fir.call @_FortranAPointerNullifyDerived(%[[ELEMENT_DESC_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[CORANK]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[ELEMENT_DESC_CAST:.*]] = fir.convert %[[ELEMENT_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAllocate(%[[ELEMENT_DESC_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

  subroutine test_allocate_with_mold()
    type(p2) :: x(10)
    class(p1), pointer :: p(:)
    integer(4) :: i(20)
    class(*), pointer :: up(:)

    allocate(p, mold=x)
    allocate(up, mold=i)
  end subroutine

! CHECK-LABEL: func.func @_QMpolyPtest_allocate_with_mold() {
! CHECK: %[[I:.*]] = fir.alloca !fir.array<20xi32> {bindc_name = "i", uniq_name = "_QMpolyFtest_allocate_with_moldEi"}
! CHECK: %[[P:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>> {bindc_name = "p", uniq_name = "_QMpolyFtest_allocate_with_moldEp"}
! CHECK: %[[UP:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>> {bindc_name = "up", uniq_name = "_QMpolyFtest_allocate_with_moldEup"}
! CHECK: %[[X:.*]] = fir.alloca !fir.array<10x!fir.type<_QMpolyTp2{a:i32,b:i32,c:i32}>> {bindc_name = "x", uniq_name = "_QMpolyFtest_allocate_with_moldEx"}
! CHECK: %[[EMBOX_X:.*]] = fir.embox %[[X]](%{{.*}}) : (!fir.ref<!fir.array<10x!fir.type<_QMpolyTp2{a:i32,b:i32,c:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.type<_QMpolyTp2{a:i32,b:i32,c:i32}>>>
! CHECK: %[[RANK:.*]] = arith.constant 1 : i32 
! CHECK: %[[P_BOX_NONE:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[X_BOX_NONE:.*]] = fir.convert %[[EMBOX_X]] : (!fir.box<!fir.array<10x!fir.type<_QMpolyTp2{a:i32,b:i32,c:i32}>>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerApplyMold(%[[P_BOX_NONE]], %[[X_BOX_NONE]], %[[RANK]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> none
! CHECK: %[[P_BOX_NONE:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAllocate(%[[P_BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK: %[[EMBOX_I:.*]] = fir.embox %[[I]](%{{.*}}) : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<20xi32>>
! CHECK: %[[RANK:.*]] = arith.constant 1 : i32
! CHECK: %[[UP_BOX_NONE:.*]] = fir.convert %[[UP]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[I_BOX_NONE:.*]] = fir.convert %[[EMBOX_I]] : (!fir.box<!fir.array<20xi32>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerApplyMold(%[[UP_BOX_NONE]], %[[I_BOX_NONE]], %[[RANK]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> none
! CHECK: %[[UP_BOX_NONE:.*]] = fir.convert %[[UP]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAllocate(%[[UP_BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

  subroutine test_allocate_with_source()
    type(p2) :: x(10)
    class(p1), pointer :: p(:)
    integer(4) :: i(20)
    class(*), pointer :: up(:)

    allocate(p, source=x)
    allocate(up, source=i)
  end subroutine

! CHECK-LABEL: func.func @_QMpolyPtest_allocate_with_source() {
! CHECK: %[[I:.*]] = fir.alloca !fir.array<20xi32> {bindc_name = "i", uniq_name = "_QMpolyFtest_allocate_with_sourceEi"}
! CHECK: %[[P:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>> {bindc_name = "p", uniq_name = "_QMpolyFtest_allocate_with_sourceEp"}
! CHECK: %[[UP:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>> {bindc_name = "up", uniq_name = "_QMpolyFtest_allocate_with_sourceEup"}
! CHECK: %[[X:.*]] = fir.alloca !fir.array<10x!fir.type<_QMpolyTp2{a:i32,b:i32,c:i32}>> {bindc_name = "x", uniq_name = "_QMpolyFtest_allocate_with_sourceEx"}
! CHECK: %[[EMBOX_X:.*]] = fir.embox %[[X]](%{{.*}}) : (!fir.ref<!fir.array<10x!fir.type<_QMpolyTp2{a:i32,b:i32,c:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.type<_QMpolyTp2{a:i32,b:i32,c:i32}>>>
! CHECK: %[[RANK:.*]] = arith.constant 1 : i32
! CHECK: %[[P_BOX_NONE:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[X_BOX_NONE:.*]] = fir.convert %[[EMBOX_X]] : (!fir.box<!fir.array<10x!fir.type<_QMpolyTp2{a:i32,b:i32,c:i32}>>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerApplyMold(%[[P_BOX_NONE]], %[[X_BOX_NONE]], %[[RANK]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> none
! CHECK: %{{.*}} = fir.call @_FortranAPointerSetBounds
! CHECK: %[[BOX_NONE_P:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[BOX_NONE_X:.*]] = fir.convert %[[EMBOX_X]] : (!fir.box<!fir.array<10x!fir.type<_QMpolyTp2{a:i32,b:i32,c:i32}>>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAllocateSource(%[[BOX_NONE_P]], %[[BOX_NONE_X]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK: %[[EMBOX_I:.*]] = fir.embox %[[I]](%{{.*}}) : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<20xi32>>
! CHECK: %[[RANK:.*]] = arith.constant 1 : i32
! CHECK: %[[UP_BOX_NONE:.*]] = fir.convert %[[UP]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[I_BOX_NONE:.*]] = fir.convert %[[EMBOX_I]] : (!fir.box<!fir.array<20xi32>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerApplyMold(%[[UP_BOX_NONE]], %[[I_BOX_NONE]], %[[RANK]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> none
! CHECK: %{{.*}} = fir.call @_FortranAPointerSetBounds
! CHECK: %[[UP_BOX_NONE:.*]] = fir.convert %[[UP]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[I_BOX_NONE:.*]] = fir.convert %[[EMBOX_I]] : (!fir.box<!fir.array<20xi32>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAllocateSource(%[[UP_BOX_NONE]], %[[I_BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

  subroutine test_allocatable_up_from_up_mold(a, b)
    class(*), allocatable :: a
    class(*), pointer :: b
    allocate(a, source = b)
  end subroutine

! CHECK-LABEL: func.func @_QMpolyPtest_allocatable_up_from_up_mold(
! CHECK-SAME: %[[A:.*]]: !fir.ref<!fir.class<!fir.heap<none>>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<!fir.class<!fir.ptr<none>>> {fir.bindc_name = "b"}) {
! CHECK: %[[LOAD_B:.*]] = fir.load %[[B]] : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[A_BOX_NONE:.*]] = fir.convert %[[A]] : (!fir.ref<!fir.class<!fir.heap<none>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[B_BOX_NONE:.*]] = fir.convert %[[LOAD_B]] : (!fir.class<!fir.ptr<none>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableApplyMold(%[[A_BOX_NONE]], %[[B_BOX_NONE]], %[[RANK]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> none
! CHECK: %[[A_BOX_NONE:.*]] = fir.convert %[[A]] : (!fir.ref<!fir.class<!fir.heap<none>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[B_BOX_NONE:.*]] = fir.convert %[[LOAD_B]] : (!fir.class<!fir.ptr<none>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocateSource(%[[A_BOX_NONE]], %[[B_BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

  subroutine test_allocatable_up_from_mold_rank(a)
    class(*), allocatable :: a(:)
    allocate(a(20), source = 10)
  end subroutine

! CHECK-LABEL: func.func @_QMpolyPtest_allocatable_up_from_mold_rank(
! CHECK-SAME: %[[A:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>> {fir.bindc_name = "a"}) {
! CHECK: %[[VALUE_10:.*]] = fir.alloca i32 {adapt.valuebyref}
! CHECK: %[[C10:.*]] = arith.constant 10 : i32
! CHECK: fir.store %[[C10]] to %[[VALUE_10]] : !fir.ref<i32>
! CHECK: %[[EMBOX_10:.*]] = fir.embox %[[VALUE_10]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK: %[[RANK:.*]] = arith.constant 1 : i32
! CHECK: %[[A_BOX_NONE:.*]] = fir.convert %[[A]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[BOX_NONE_10:.*]] = fir.convert %[[EMBOX_10]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableApplyMold(%[[A_BOX_NONE]], %[[BOX_NONE_10]], %[[RANK]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> none
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C2:.*]] = arith.constant 20 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: %[[A_BOX_NONE:.*]] = fir.convert %[[A]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[C1_I64:.*]] = fir.convert %[[C1]] : (index) -> i64
! CHECK: %[[C20_I64:.*]] = fir.convert %[[C20]] : (i32) -> i64
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableSetBounds(%[[A_BOX_NONE]], %[[C0]], %[[C1_I64]], %[[C20_I64]]) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK: %[[A_BOX_NONE:.*]] = fir.convert %[[A]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[BOX_NONE_10:.*]] = fir.convert %[[EMBOX_10]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocateSource(%[[A_BOX_NONE]], %[[BOX_NONE_10]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

  subroutine test_allocatable_up_character()
    class(*), allocatable :: a
    allocate(character*10::a)
  end subroutine

! CHECK-LABEL: func.func @_QMpolyPtest_allocatable_up_character() {
! CHECK: %[[A:.*]] = fir.alloca !fir.class<!fir.heap<none>> {bindc_name = "a", uniq_name = "_QMpolyFtest_allocatable_up_characterEa"}
! CHECK: %[[LEN:.*]] = arith.constant 10 : i64
! CHECK: %[[A_NONE:.*]] = fir.convert %[[A]] : (!fir.ref<!fir.class<!fir.heap<none>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[KIND:.*]] = arith.constant 1 : i32
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[CORANK:.*]] = arith.constant 0 : i32
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableInitCharacterForAllocate(%[[A_NONE]], %[[LEN]], %[[KIND]], %[[RANK]], %[[CORANK]]) {{.*}} : (!fir.ref<!fir.box<none>>, i64, i32, i32, i32) -> none
! CHECK: %[[A_NONE:.*]] = fir.convert %[[A:.*]] : (!fir.ref<!fir.class<!fir.heap<none>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[A_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

end module


program test_alloc
  use poly

  call test_allocatable()
  call test_pointer()
end

! Check code generation of allocate runtime calls for polymorphic entities. This
! is done from Fortran so we don't have a file full of auto-generated type info
! in order to perform the checks.

! LLVM-LABEL: define void @_QMpolyPtest_allocatable()

! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerivedForAllocate(ptr %{{.*}}, ptr @_QMpolyE.dt.p1, i32 0, i32 0)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr %{{.*}}, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerivedForAllocate(ptr %{{.*}}, ptr @_QMpolyE.dt.p1, i32 0, i32 0)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr %{{.*}}, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerivedForAllocate(ptr %{{.*}}, ptr @_QMpolyE.dt.p2, i32 0, i32 0)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr %{{.*}}, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerivedForAllocate(ptr %{{.*}}, ptr @_QMpolyE.dt.p1, i32 1, i32 0)
! LLVM: %{{.*}} = call {} @_FortranAAllocatableSetBounds(ptr %{{.*}}, i32 0, i64 1, i64 10)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr %{{.*}}, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerivedForAllocate(ptr %{{.*}}, ptr @_QMpolyE.dt.p2, i32 1, i32 0)
! LLVM: %{{.*}} = call {} @_FortranAAllocatableSetBounds(ptr %{{.*}}, i32 0, i64 1, i64 20)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr %{{.*}}, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM-COUNT-2:  call void %{{.*}}()

! LLVM: %[[C1_LOAD:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.*}}
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[C1_LOAD]], ptr %{{.*}}
! LLVM: %[[GEP_TDESC_C1:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 7
! LLVM: %[[TDESC_C1:.*]] = load ptr, ptr %[[GEP_TDESC_C1]]
! LLVM: %[[ELEM_SIZE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 1
! LLVM: %[[ELEM_SIZE:.*]] = load i64, ptr %[[ELEM_SIZE_GEP]]
! LLVM: %[[TYPE_CODE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 4
! LLVM: %[[TYPE_CODE:.*]] = load i32, ptr %[[TYPE_CODE_GEP]]
! LLVM: %{{.*}} = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } undef, i64 %[[ELEM_SIZE]], 1
! LLVM: %[[TRUNC_TYPE_CODE:.*]] = trunc i32 %[[TYPE_CODE]] to i8 
! LLVM: %{{.*}} = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %{{.*}}, i8 %[[TRUNC_TYPE_CODE]], 4
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %{{.*}}, ptr %[[TMP:.*]]
! LLVM: call void %{{.*}}(ptr %{{.*}}) 

! LLVM: %[[LOAD_C2:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.*}}
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOAD_C2]], ptr %{{.*}}
! LLVM: %[[GEP_TDESC_C2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 7
! LLVM: %[[TDESC_C2:.*]] = load ptr, ptr %[[GEP_TDESC_C2]]
! LLVM: %[[ELEM_SIZE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 1
! LLVM: %[[ELEM_SIZE:.*]] = load i64, ptr %[[ELEM_SIZE_GEP]]
! LLVM: %[[TYPE_CODE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 4
! LLVM: %[[TYPE_CODE:.*]] = load i32, ptr %[[TYPE_CODE_GEP]]
! LLVM: %{{.*}} = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } undef, i64 %[[ELEM_SIZE]], 1
! LLVM: %[[TRUNC_TYPE_CODE:.*]] = trunc i32 %[[TYPE_CODE]] to i8 
! LLVM: %{{.*}} = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %{{.*}}, i8 %[[TRUNC_TYPE_CODE]], 4
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %{{.*}}, ptr %{{.*}}
! LLVM: call void %{{.*}}(ptr %{{.*}})

! LLVM: %[[C3_LOAD:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } %[[C3_LOAD]], ptr %{{.*}}

! LLVM: %[[GEP_TDESC_C3:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 8
! LLVM: %[[TDESC_C3:.*]] = load ptr, ptr %[[GEP_TDESC_C3]]
! LLVM: %[[ELE_SIZE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 1
! LLVM: %[[ELE_SIZE:.*]] = load i64, ptr %[[ELE_SIZE_GEP]]
! LLVM: %[[TYPE_CODE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 4
! LLVM: %[[TYPE_CODE:.*]] = load i32, ptr %[[TYPE_CODE_GEP]]
! LLVM: %[[BOX0:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } undef, i64 %[[ELE_SIZE]], 1
! LLVM: %[[BOX1:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX0]], i32 20180515, 2
! LLVM: %[[BOX2:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX1]], i8 0, 3
! LLVM: %[[TYPE_CODE_TRUNC:.*]] = trunc i32 %[[TYPE_CODE]] to i8
! LLVM: %[[BOX3:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX2]], i8 %[[TYPE_CODE_TRUNC]], 4
! LLVM: %[[BOX4:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX3]], i8 0, 5
! LLVM: %[[BOX5:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX4]], i8 1, 6
! LLVM: %[[BOX6:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX5]], ptr %[[TDESC_C3]], 7
! LLVM: %[[BOX7:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX6]], ptr %{{.*}}, 0
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX7]], ptr %{{.*}}
! LLVM: call void %{{.*}}(ptr %{{.*}})

! LLVM: %[[C4_LOAD:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } %[[C4_LOAD]], ptr %{{.*}}
! LLVM: %[[GEP_TDESC_C4:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 8
! LLVM: %[[TDESC_C4:.*]] = load ptr, ptr %[[GEP_TDESC_C4]]
! LLVM: %[[ELE_SIZE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 1
! LLVM: %[[ELE_SIZE:.*]] = load i64, ptr %[[ELE_SIZE_GEP]]
! LLVM: %[[TYPE_CODE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 4
! LLVM: %[[TYPE_CODE:.*]] = load i32, ptr %[[TYPE_CODE_GEP]]
! LLVM: %[[BOX0:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } undef, i64 %[[ELE_SIZE]], 1
! LLVM: %[[BOX1:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX0]], i32 20180515, 2
! LLVM: %[[BOX2:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX1]], i8 0, 3
! LLVM: %[[TYPE_CODE_TRUNC:.*]] = trunc i32 %[[TYPE_CODE]] to i8
! LLVM: %[[BOX3:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX2]], i8 %[[TYPE_CODE_TRUNC]], 4
! LLVM: %[[BOX4:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX3]], i8 0, 5
! LLVM: %[[BOX5:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX4]], i8 1, 6
! LLVM: %[[BOX6:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX5]], ptr %[[TDESC_C4]], 7
! LLVM: %[[BOX7:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX6]], ptr %{{.*}}, 0
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX7]], ptr %{{.*}}
! LLVM: call void %{{.*}}(ptr %{{.*}})


! Check code generation of a allocate/deallocate sequence on polymorphic
! allocatable.

! LLVM-LABEL: define void @_QMpolyPtest_deallocate()
! LLVM: %[[ALLOCA1:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }
! LLVM: %[[ALLOCA2:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, i64 1
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } { ptr null, i64 ptrtoint (ptr getelementptr (%_QMpolyTp1, ptr null, i32 1) to i64), i32 20180515, i8 0, i8 42, i8 2, i8 1, ptr @_QMpolyE.dt.p1, [1 x i64] undef }, ptr %[[ALLOCA1]]
! LLVM: %[[LOAD:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[ALLOCA1]]
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOAD]], ptr %[[ALLOCA2]]
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerivedForAllocate(ptr %[[ALLOCA2]], ptr @_QMpolyE.dt.p1, i32 0, i32 0)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr %[[ALLOCA2]], i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableDeallocatePolymorphic(ptr %[[ALLOCA2]], ptr {{.*}}, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
