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
end module

program test_allocatable
  use poly

  class(p1), allocatable :: p
  class(p1), allocatable :: c1, c2
  class(p1), allocatable, dimension(:) :: c3, c4
  integer :: i

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
end

! CHECK-LABEL: func.func @_QQmain()

! CHECK-DAG: %[[C1:.*]] = fir.address_of(@_QFEc1) : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK-DAG: %[[C2:.*]] = fir.address_of(@_QFEc2) : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK-DAG: %[[C3:.*]] = fir.address_of(@_QFEc3) : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK-DAG: %[[C4:.*]] = fir.address_of(@_QFEc4) : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK-DAG: %[[P:.*]] = fir.address_of(@_QFEp) : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[P_CAST:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerived(%[[P_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[C0]]) : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[P_CAST:.*]] = fir.convert %[[P]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[P_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[C1_CAST:.*]] = fir.convert %0 : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerived(%[[C1_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[C0]]) : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C1_CAST:.*]] = fir.convert %[[C1]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[C1_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P2:.*]] = fir.address_of(@_QMpolyE.dt.p2) : !fir.ref<!fir.type<
! CHECK: %[[C2_CAST:.*]] = fir.convert %[[C2]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P2_CAST:.*]] = fir.convert %[[TYPE_DESC_P2]] : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[RANK:.*]] = arith.constant 0 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerived(%[[C2_CAST]], %[[TYPE_DESC_P2_CAST]], %[[RANK]], %[[C0]]) : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C2_CAST:.*]] = fir.convert %[[C2]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[C2_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P1:.*]] = fir.address_of(@_QMpolyE.dt.p1) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P1_CAST:.*]] = fir.convert %[[TYPE_DESC_P1]] : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[RANK:.*]] = arith.constant 1 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerived(%[[C3_CAST]], %[[TYPE_DESC_P1_CAST]], %[[RANK]], %[[C0]]) : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[C10:.*]] = arith.constant 10 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[C1_I64:.*]] = fir.convert %c1 : (index) -> i64
! CHECK: %[[C10_I64:.*]] = fir.convert %[[C10]] : (i32) -> i64
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableSetBounds(%[[C3_CAST]], %[[C0]], %[[C1_I64]], %[[C10_I64]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK: %[[C3_CAST:.*]] = fir.convert %[[C3]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[C3_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! CHECK: %[[TYPE_DESC_P2:.*]] = fir.address_of(@_QMpolyE.dt.p2) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[TYPE_DESC_P2_CAST:.*]] = fir.convert %[[TYPE_DESC_P2]] : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype
! CHECK: %[[RANK:.*]] = arith.constant 1 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: fir.call @_FortranAAllocatableInitDerived(%[[C4_CAST]], %[[TYPE_DESC_P2_CAST]], %[[RANK]], %[[C0]]) : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> none
! CHECK: %[[CST1:.*]] = arith.constant 1 : index
! CHECK: %[[C20:.*]] = arith.constant 20 : i32
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[C1_I64:.*]] = fir.convert %[[CST1]] : (index) -> i64
! CHECK: %[[C20_I64:.*]] = fir.convert %[[C20]] : (i32) -> i64
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableSetBounds(%[[C4_CAST]], %[[C0]], %[[C1_I64]], %[[C20_I64]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK: %[[C4_CAST:.*]] = fir.convert %[[C4]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[C4_CAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! Check fir.rebox for fir.class
! CHECK: %[[C1_LOAD:.*]] = fir.load %[[C1]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[C1_REBOX:.*]] = fir.rebox %[[C1_LOAD]] : (!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc2"(%[[C1_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[C1_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK: %[[C2_LOAD:.*]] = fir.load %[[C2]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>
! CHECK: %[[C2_REBOX:.*]] = fir.rebox %[[C2_LOAD]] : (!fir.class<!fir.heap<!fir.type<_QMpolyTp1{a:i32,b:i32}>>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc2"(%[[C2_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[C2_REBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK-LABEL: %{{.*}} = fir.do_loop
! CHECK: %[[C3_LOAD:.*]] = fir.load %[[C3]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C3_COORD:.*]] = fir.coordinate_of %[[C3_LOAD]], %{{.*}} : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, i64) -> !fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[C3_TDESC:.*]] = fir.box_tdesc %[[C3_LOAD]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[C3_EMBOX:.*]] = fir.embox %[[C3_COORD]] tdesc %[[C3_TDESC]] : (!fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>, !fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc2"(%[[C3_EMBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[C3_EMBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! CHECK-LABEL: %{{.*}} = fir.do_loop
! CHECK: %[[C4_LOAD:.*]] = fir.load %[[C4]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>>
! CHECK: %[[C4_COORD:.*]] = fir.coordinate_of %[[C4_LOAD]], %{{.*}} : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>, i64) -> !fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[C4_TDESC:.*]] = fir.box_tdesc %[[C4_LOAD]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpolyTp1{a:i32,b:i32}>>>>) -> !fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: %[[C4_EMBOX:.*]] = fir.embox %[[C4_COORD]] tdesc %[[C4_TDESC]] : (!fir.ref<!fir.type<_QMpolyTp1{a:i32,b:i32}>>, !fir.tdesc<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>
! CHECK: fir.dispatch "proc2"(%[[C4_EMBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) (%[[C4_EMBOX]] : !fir.class<!fir.type<_QMpolyTp1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

! Check code generation of allocate runtime calls for polymoprhic entities. This
! is done from Fortran so we don't have a file full of auto-generated type info
! in order to perform the checks.

! LLVM-LABEL: define void @_QQmain()
! LLVM: %[[TMP1:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }
! LLVM: %[[TMP2:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }
! LLVM: %[[TMP3:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }
! LLVM: %[[TMP4:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }
! LLVM: %[[TMP5:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }
! LLVM: %[[TMP6:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }
! LLVM: %[[TMP7:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }
! LLVM: %[[TMP8:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }

! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerived(ptr @_QFEp, ptr @_QMpolyE.dt.p1, i32 0, i32 0)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr @_QFEp, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerived(ptr @_QFEc1, ptr @_QMpolyE.dt.p1, i32 0, i32 0)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr @_QFEc1, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerived(ptr @_QFEc2, ptr @_QMpolyE.dt.p2, i32 0, i32 0)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr @_QFEc2, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerived(ptr @_QFEc3, ptr @_QMpolyE.dt.p1, i32 1, i32 0)
! LLVM: %{{.*}} = call {} @_FortranAAllocatableSetBounds(ptr @_QFEc3, i32 0, i64 1, i64 10)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr @_QFEc3, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM: %{{.*}} = call {} @_FortranAAllocatableInitDerived(ptr @_QFEc4, ptr @_QMpolyE.dt.p2, i32 1, i32 0)
! LLVM: %{{.*}} = call {} @_FortranAAllocatableSetBounds(ptr @_QFEc4, i32 0, i64 1, i64 20)
! LLVM: %{{.*}} = call i32 @_FortranAAllocatableAllocate(ptr @_QFEc4, i1 false, ptr null, ptr @_QQcl.{{.*}}, i32 {{.*}})
! LLVM-COUNT-2:  call void %{{.*}}()

! LLVM: %[[C1_LOAD:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr @_QFEc1
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[C1_LOAD]], ptr %[[TMP8]]
! LLVM: %[[GEP_TDESC_C1:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[TMP8]], i32 0, i32 7
! LLVM: %[[TDESC_C1:.*]] = load ptr, ptr %[[GEP_TDESC_C1]]
! LLVM: %[[BOX0:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } { ptr undef, i64 ptrtoint (ptr getelementptr (%_QMpolyTp1, ptr null, i32 1) to i64), i32 20180515, i8 0, i8 42, i8 0, i8 1, ptr undef, [1 x i64] undef }, ptr %[[TDESC_C1]], 7
! LLVM: %[[BOX1:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX0]], ptr %{{.*}}, 0
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX1]], ptr %[[TMP7]]

! LLVM: %[[LOAD_C2:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr @_QFEc2
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[LOAD_C2]], ptr %[[TMP6]]
! LLVM: %[[GEP_TDESC_C2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[TMP6]], i32 0, i32 7
! LLVM: %[[TDESC_C2:.*]] = load ptr, ptr %[[GEP_TDESC_C2]]
! LLVM: %[[BOX0:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } { ptr undef, i64 ptrtoint (ptr getelementptr (%_QMpolyTp1, ptr null, i32 1) to i64), i32 20180515, i8 0, i8 42, i8 0, i8 1, ptr undef, [1 x i64] undef }, ptr %[[TDESC_C2]], 7
! LLVM: %[[BOX1:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX0]], ptr %{{.*}}, 0
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX1]], ptr %[[TMP5]]

! LLVM: %[[C3_LOAD:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr @_QFEc3
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } %[[C3_LOAD]], ptr %[[TMP2]]
! LLVM: %[[GEP_TDESC_C3:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[TMP2]], i32 0, i32 8
! LLVM: %[[TDESC_C3:.*]] = load ptr, ptr %[[GEP_TDESC_C3]]
! LLVM: %[[BOX0:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } { ptr undef, i64 ptrtoint (ptr getelementptr (%_QMpolyTp1, ptr null, i32 1) to i64), i32 20180515, i8 0, i8 42, i8 0, i8 1, ptr undef, [1 x i64] undef }, ptr %[[TDESC_C3]], 7
! LLVM: %[[BOX1:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX0]], ptr %{{.*}}, 0
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX1]], ptr %[[TMP1]]

! LLVM: %[[C4_LOAD:.*]] = load { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr @_QFEc4
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] } %[[C4_LOAD]], ptr %[[TMP4]]
! LLVM: %[[GEP_TDESC_C4:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[TMP4]], i32 0, i32 8
! LLVM: %[[TDESC_C4:.*]] = load ptr, ptr %[[GEP_TDESC_C4]]
! LLVM: %[[BOX0:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } { ptr undef, i64 ptrtoint (ptr getelementptr (%_QMpolyTp1, ptr null, i32 1) to i64), i32 20180515, i8 0, i8 42, i8 0, i8 1, ptr undef, [1 x i64] undef }, ptr %[[TDESC_C4]], 7
! LLVM: %[[BOX1:.*]] = insertvalue { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX0]], ptr %{{.*}}, 0
! LLVM: store { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] } %[[BOX1]], ptr %[[TMP3]]
