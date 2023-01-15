! RUN: bbc -polymorphic-type -emit-fir %s -o - | FileCheck %s
! RUN: bbc -polymorphic-type -emit-fir %s -o - | tco | FileCheck %s --check-prefix=LLVM

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

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CHECK: %[[P_DESC_CAST:.*]] = fir.convert %[[P_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[CORANK:.*]] = arith.constant 0 : i32
! CHECK: %{{.*}} = fir.call @_FortranAPointerNullifyDerived(%[[P_DESC_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[CORANK]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[P_DESC_CAST:.*]] = fir.convert %[[P_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAllocate(%[[P_DESC_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! call p%proc1()
! CHECK: %[[P_CAST:.*]] = fir.convert %[[P_DESC:.*]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>
! CHECK: fir.dispatch "proc1"(%[[P_CAST]] : !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>)

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CHECK: %[[C1_DESC_CAST:.*]] = fir.convert %[[C1_DESC:.*]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[CORANK:.*]] = arith.constant 0 : i32
! CHECK: %{{.*}} = fir.call @_FortranAPointerNullifyDerived(%[[C1_DESC_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[CORANK]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C1_DESC_CAST:.*]] = fir.convert %[[C1_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAllocate(%[[C1_DESC_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P2:.*]] = fir.address_of(@_QMpolyE.dt.p2) : !fir.ref<!fir.type<{{.*}}>>
! CHECK: %[[C2_DESC_CAST:.*]] = fir.convert %[[C2_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P2_CAST:.*]] = fir.convert %[[TYPE_DESC_P2]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[CORANK:.*]] = arith.constant 0 : i32
! CHECK: %{{.*}} = fir.call @_FortranAPointerNullifyDerived(%[[C2_DESC_CAST]], %[[TYPE_DESC_P2_CAST]], %[[RANK]], %[[CORANK]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C2_DESC_CAST:.*]] = fir.convert %[[C2_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAllocate(%[[C2_DESC_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! call c1%proc1()
! CHECK: %[[C1_DESC_CAST:.*]] = fir.convert %[[C1_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>
! CHECK: fir.dispatch "proc1"(%[[C1_DESC_CAST]] : !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>)

! call c2%proc1()
! CHECK: %[[C2_DESC_CAST:.*]] = fir.convert %[[C2_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>
! CHECK: fir.dispatch "proc1"(%[[C2_DESC_CAST]] : !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>)

! call c1%proc2()
! CHECK: %[[C1_LOAD:.*]] = fir.load %[[C1_DESC]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[C1_REBOX:.*]] = fir.rebox %[[C1_LOAD]] : (!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc2"(%[[C1_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[C1_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! call c2%proc2()
! CHECK: %[[C2_LOAD:.*]] = fir.load %[[C2_DESC]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[C2_REBOX:.*]] = fir.rebox %[[C2_LOAD]] : (!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc2"(%[[C2_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[C2_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<{{.*}}>>) -> !fir.ref<none>
! CHECK: %[[RANK:.*]] = arith.constant 1 : i32
! CHECK: %[[CORANK:.*]] = arith.constant 0 : i32
! CHECK: %{{.*}} = fir.call @_FortranAPointerNullifyDerived(%[[C3_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[CORANK]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerSetBounds(%[[C3_CAST]], %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAllocate(%[[C3_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P2:.*]] = fir.address_of(@_QMpolyE.dt.p2) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<{{.*}}>>
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P2_CAST:.*]] = fir.convert %[[TYPE_DESC_P2]] : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<{{.*}}>>) -> !fir.ref<none>
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


! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CHECK: %[[P_CAST:.*]] = fir.convert %[[P_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerDeallocatePolymorphic(%[[P_CAST]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CHECK: %[[C1_DESC_CAST:.*]] = fir.convert %[[C1_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerDeallocatePolymorphic(%[[C1_DESC_CAST]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CHECK: %[[C2_DESC_CAST:.*]] = fir.convert %[[C2_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerDeallocatePolymorphic(%[[C2_DESC_CAST]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CHECK: %[[C3_DESC_CAST:.*]] = fir.convert %[[C3_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAPointerDeallocatePolymorphic(%[[C3_DESC_CAST]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CHECK: %[[C4_DESC_CAST:.*]] = fir.convert %[[C4_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
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

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[P_CAST:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerived(%[[P_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[C0]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[P_CAST:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[P_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[C1_CAST:.*]] = fir.convert %0 : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerived(%[[C1_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[C0]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C1_CAST:.*]] = fir.convert %[[C1]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[C1_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P2:.*]] = fir.address_of(@_QMpolyE.dt.p2) : !fir.ref<!fir.type<
! CHECK: %[[C2_CAST:.*]] = fir.convert %[[C2]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P2_CAST:.*]] = fir.convert %[[TYPE_DESC_P2]] : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerived(%[[C2_CAST]], %[[TYPE_DESC_P2_CAST]], %[[RANK]], %[[C0]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C2_CAST:.*]] = fir.convert %[[C2]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[C2_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[RANK:.*]] = arith.constant 1 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerived(%[[C3_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[C0]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C10:.*]] = arith.constant 10 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[C1_I64:.*]] = fir.convert %c1 : (index) -> i64
! CHECK: %[[C10_I64:.*]] = fir.convert %[[C10]] : (i32) -> i64
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableSetBounds(%[[C3_CAST]], %[[C0]], %[[C1_I64]], %[[C10_I64]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[C3_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P2:.*]] = fir.address_of(@_QMpolyE.dt.p2) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P2_CAST:.*]] = fir.convert %[[TYPE_DESC_P2]] : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[RANK:.*]] = arith.constant 1 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerived(%[[C4_CAST]], %[[TYPE_DESC_P2_CAST]], %[[RANK]], %[[C0]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[CST1:.*]] = arith.constant 1 : index
! CHECK: %[[C20:.*]] = arith.constant 20 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[C1_I64:.*]] = fir.convert %[[CST1]] : (index) -> i64
! CHECK: %[[C20_I64:.*]] = fir.convert %[[C20]] : (i32) -> i64
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableSetBounds(%[[C4_CAST]], %[[C0]], %[[C1_I64]], %[[C20_I64]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[C4_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[C1_CAST:.*]] = fir.convert %[[C1_DESC]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>
! CHECK: fir.dispatch "proc1"(%[[C1_CAST]] : !fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>)

! CHECK: %[[C2_CAST:.*]] = fir.convert %[[C2_DESC]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>
! CHECK: fir.dispatch "proc1"(%[[C2_CAST]] : !fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>)

! CHECK: %[[C1_LOAD:.*]] = fir.load %[[C1_DESC]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[C1_REBOX:.*]] = fir.rebox %[[C1_LOAD]] : (!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc2"(%[[C1_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[C1_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK: %[[C2_LOAD:.*]] = fir.load %[[C2_DESC]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[C2_REBOX:.*]] = fir.rebox %[[C2_LOAD]] : (!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
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

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CHECK: %[[P_CAST:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%[[P_CAST]], %[[TYPE_NONE]], %{{.*}}, %1{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CHECK: %[[C1_CAST:.*]] = fir.convert %[[C1]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%[[C1_CAST]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CHECK: %[[C2_CAST:.*]] = fir.convert %[[C2]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%[[C2_CAST]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%[[C3_CAST]], %[[TYPE_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_ADDR:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_NONE:.*]] = fir.convert %[[TYPE_DESC_ADDR]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none>
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
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableInitIntrinsic(%[[BOX_NONE]], %[[CAT]], %[[KIND]], %[[RANK]], %[[CORANK]]) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, i32, i32) -> none
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
end module

! CHECK-LABEL: func.func @_QMpolyPtest_type_with_polymorphic_pointer_component()
! CHECK: %[[TYPE_PTR:.*]] = fir.alloca !fir.ptr<!fir.type<_QMpolyTwith_alloc{element:!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>}>> {uniq_name = "_QMpolyFtest_type_with_polymorphic_pointer_componentEa.addr"}
! CHECK: %[[TYPE_PTR_LOAD:.*]] = fir.load %[[TYPE_PTR]] : !fir.ref<!fir.ptr<!fir.type<_QMpolyTwith_alloc{element:!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>}>>>
! CHECK: %[[ELEMENT:.*]] = fir.field_index element, !fir.type<_QMpolyTwith_alloc{element:!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>}>
! CHECK: %[[ELEMENT_DESC:.*]] = fir.coordinate_of %[[TYPE_PTR_LOAD]], %[[ELEMENT]] : (!fir.ptr<!fir.type<_QMpolyTwith_alloc{element:!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>}>>, !fir.field) -> !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[ZERO_DESC:.*]] = fir.embox %[[ZERO]] : (!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>
! CHECK: fir.store %[[ZERO_DESC]] to %[[ELEMENT_DESC]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[TYPE_DESC_P1:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<{{.*}}>>
! CHECK: %[[ELEMENT_DESC_CAST:.*]] = fir.convert %[[ELEMENT_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.ref<!fir.type<{{.*}}>>) -> !fir.ref<none> 
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[CORANK:.*]] = arith.constant 0 : i32
! CHECK: %{{.*}} = fir.call @_FortranAPointerNullifyDerived(%[[ELEMENT_DESC_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[CORANK]]) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[ELEMENT_DESC_CAST:.*]] = fir.convert %[[ELEMENT_DESC]] : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAPointerAllocate(%[[ELEMENT_DESC_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

program test_alloc
  use poly

  call test_allocatable()
  call test_pointer()
end

! Check code generation of allocate runtime calls for polymorphic entities. This
! is done from Fortran so we don't have a file full of auto-generated type info
! in order to perform the checks.

! LLVM-LABEL: define void @_QMpolyPtest_allocatable()

! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerived(ptr %{{.*}}, ptr @_QMpolyE.dt.p1, i32 0, i32 0)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr %{{.*}}, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerived(ptr %{{.*}}, ptr @_QMpolyE.dt.p1, i32 0, i32 0)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr %{{.*}}, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerived(ptr %{{.*}}, ptr @_QMpolyE.dt.p2, i32 0, i32 0)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr %{{.*}}, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerived(ptr %{{.*}}, ptr @_QMpolyE.dt.p1, i32 1, i32 0)
! LLVM: %{{.*}} = call {} @_FortranAAllocatableSetBounds(ptr %{{.*}}, i32 0, i64 1, i64 10)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr %{{.*}}, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerived(ptr %{{.*}}, ptr @_QMpolyE.dt.p2, i32 1, i32 0)
! LLVM: %{{.*}} = call {} @_FortranAAllocatableSetBounds(ptr %{{.*}}, i32 0, i64 1, i64 20)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr %{{.*}}, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM-COUNT-2:  call void %{{.*}}()

! LLVM: %[[C1_LOAD:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.*}}
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[C1_LOAD]], ptr %{{.*}}
! LLVM: %[[GEP_TDESC_C1:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 7
! LLVM: %[[TDESC_C1:.*]] = load ptr, ptr %[[GEP_TDESC_C1]]
! LLVM: %[[ELEM_SIZE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.}}, i32 0, i32 1
! LLVM: %[[ELEM_SIZE:.*]] = load i64, ptr %[[ELEM_SIZE_GEP]]
! LLVM: %[[TYPE_CODE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.}}, i32 0, i32 4
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
! LLVM: %[[ELEM_SIZE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.}}, i32 0, i32 1
! LLVM: %[[ELEM_SIZE:.*]] = load i64, ptr %[[ELEM_SIZE_GEP]]
! LLVM: %[[TYPE_CODE_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.}}, i32 0, i32 4
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

! LLVM: %[[BOX0:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } { ptr undef, i64 ptrtoint (ptr getelementptr (%_QMpolyTp1, ptr null, i32 1) to i64), i32 20180515, i8 0, i8 42, i8 0, i8 1, ptr undef, [1 x i64] undef }, ptr %[[TDESC_C3]], 7
! LLVM: %[[BOX1:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX0]], ptr %{{.*}}, 0
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX1]], ptr %{{.*}}
! LLVM: call void %{{.*}}(ptr %{{.*}})

! LLVM: %[[C4_LOAD:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } %[[C4_LOAD]], ptr %{{.*}}
! LLVM: %[[GEP_TDESC_C4:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 8
! LLVM: %[[TDESC_C4:.*]] = load ptr, ptr %[[GEP_TDESC_C4]]
! LLVM: %[[BOX0:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } { ptr undef, i64 ptrtoint (ptr getelementptr (%_QMpolyTp1, ptr null, i32 1) to i64), i32 20180515, i8 0, i8 42, i8 0, i8 1, ptr undef, [1 x i64] undef }, ptr %[[TDESC_C4]], 7
! LLVM: %[[BOX1:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX0]], ptr %{{.*}}, 0
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX1]], ptr %{{.*}}
! LLVM: call void %{{.*}}(ptr %{{.*}})


! Check code generation of a allocate/deallocate sequence on polymorphic
! allocatable.

! LLVM-LABEL: define void @_QMpolyPtest_deallocate()
! LLVM: %[[ALLOCA1:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }
! LLVM: %[[ALLOCA2:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, i64 1
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } { ptr null, i64 ptrtoint (ptr getelementptr (%_QMpolyTp1, ptr null, i32 1) to i64), i32 20180515, i8 0, i8 42, i8 2, i8 1, ptr @_QMpolyE.dt.p1, [1 x i64] undef }, ptr %[[ALLOCA1]]
! LLVM: %[[LOAD:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[ALLOCA1]]
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOAD]], ptr %[[ALLOCA2]]
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerived(ptr %[[ALLOCA2]], ptr @_QMpolyE.dt.p1, i32 0, i32 0)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr %[[ALLOCA2]], i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableDeallocatePolymorphic(ptr %[[ALLOCA2]], ptr {{.*}}, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
