! Test default initialization of global variables (static init)
! RUN: bbc -hlfir=false %s -o - | FileCheck %s --check-prefixes=%if system-aix %{"CHECK","CHECK-BE"%} \
! RUN:                                         %else %{"CHECK","CHECK-LE"%}

module tinit
  real, target :: ziel(100)

  type tno_init
    integer :: k
  end type

  type t0
    integer :: k = 66
  end type

  ! Test type that combines all kinds of components with and without
  ! default initialization.
  type t1
    ! Simple type component with default init
    integer :: i = 42
    ! Simple type component without default init
    integer :: j

    ! Pointer component with a default initial target
    real, pointer :: x(:) => ziel
    ! Pointer component with no init
    real, pointer :: y(:)
    ! Pointer component with null init
    real, pointer :: z(:) => NULL()

    ! Character component with init
    character(10) c1 = "hello"
    ! Character component without init
    character(10) c2

    ! Component with a derived type that has default init
    type(t0) :: somet0
    ! Component with a derived type that has no default init.
    type(tno_init) :: sometno_init
    ! Component whose type default init is overridden by
    ! default init for the component.
    type(t0) :: somet0_2 = t0(33)
    ! Array component with a derived type that has default init
    type(t0) :: somet0_array
  end type

  ! Test type that extends type with default init.
  type, extends(t0) :: textendst0
    integer :: l
  end type

  ! Test type with default init in equivalences
  type tseq
    sequence
    integer :: i = 2
    integer :: j = 3
  end type

  type tv
    real, pointer :: v(:)
  end type

  real, pointer :: mv(:)

  ! Test scalar with default init
  type(t0) :: at0
! CHECK-LABEL: fir.global @_QMtinitEat0 : !fir.type<_QMtinitTt0{k:i32}> {
  ! CHECK: %[[VAL_0:.*]] = arith.constant 66 : i32
  ! CHECK: %[[VAL_1:.*]] = fir.undefined !fir.type<_QMtinitTt0{k:i32}>
  ! CHECK: %[[VAL_2:.*]] = fir.insert_value %[[VAL_1]], %[[VAL_0]], ["k", !fir.type<_QMtinitTt0{k:i32}>] : (!fir.type<_QMtinitTt0{k:i32}>, i32) -> !fir.type<_QMtinitTt0{k:i32}>
  ! CHECK: fir.has_value %[[VAL_2]] : !fir.type<_QMtinitTt0{k:i32}>

  ! Test array with default init
  type(t0) :: bt0(100)
! CHECK-LABEL: @_QMtinitEbt0 : !fir.array<100x!fir.type<_QMtinitTt0{k:i32}>> {
  ! CHECK: %[[VAL_3:.*]] = arith.constant 66 : i32
  ! CHECK: %[[VAL_4:.*]] = fir.undefined !fir.type<_QMtinitTt0{k:i32}>
  ! CHECK: %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_3]], ["k", !fir.type<_QMtinitTt0{k:i32}>] : (!fir.type<_QMtinitTt0{k:i32}>, i32) -> !fir.type<_QMtinitTt0{k:i32}>
  ! CHECK: %[[VAL_6:.*]] = fir.undefined !fir.array<100x!fir.type<_QMtinitTt0{k:i32}>>
  ! CHECK: %[[VAL_7:.*]] = fir.insert_on_range %[[VAL_6]], %[[VAL_5]] from (0) to (99) : (!fir.array<100x!fir.type<_QMtinitTt0{k:i32}>>, !fir.type<_QMtinitTt0{k:i32}>) -> !fir.array<100x!fir.type<_QMtinitTt0{k:i32}>>
  ! CHECK: fir.has_value %[[VAL_7]] : !fir.array<100x!fir.type<_QMtinitTt0{k:i32}>>

  ! Test default init overridden by explicit init
  type(t0) :: ct0 = t0(42)
! CHECK-LABEL: fir.global @_QMtinitEct0 : !fir.type<_QMtinitTt0{k:i32}> {
  ! CHECK: %[[VAL_8:.*]] = arith.constant 42 : i32
  ! CHECK: %[[VAL_9:.*]] = fir.undefined !fir.type<_QMtinitTt0{k:i32}>
  ! CHECK: %[[VAL_10:.*]] = fir.insert_value %[[VAL_9]], %[[VAL_8]], ["k", !fir.type<_QMtinitTt0{k:i32}>] : (!fir.type<_QMtinitTt0{k:i32}>, i32) -> !fir.type<_QMtinitTt0{k:i32}>
  ! CHECK: fir.has_value %[[VAL_10]] : !fir.type<_QMtinitTt0{k:i32}>

  ! Test a non trivial derived type mixing all sorts of default initialization
  type(t1) :: dt1
! CHECK-LABEL: @_QMtinitEdt1 : !fir.type<_QMtinitTt1{{.*}}> {
  ! CHECK-DAG: %[[VAL_11:.*]] = arith.constant 42 : i32
  ! CHECK-DAG: %[[VAL_12:.*]] = arith.constant 100 : index
  ! CHECK-DAG: %[[VAL_13:.*]] = arith.constant 0 : index
  ! CHECK-DAG: %[[VAL_14:.*]] = arith.constant 33 : i32
  ! CHECK-DAG: %[[VAL_15:.*]] = arith.constant 66 : i32
  ! CHECK: %[[VAL_16:.*]] = fir.undefined !fir.type<_QMtinitTt1{{.*}}>
  ! CHECK: %[[VAL_17:.*]] = fir.insert_value %[[VAL_16]], %[[VAL_11]], ["i", !fir.type<_QMtinitTt1{{.*}}>] : (!fir.type<_QMtinitTt1{{.*}}>, i32) -> !fir.type<_QMtinitTt1{{.*}}>
  ! CHECK: %[[VAL_18:.*]] = fir.zero_bits i32
  ! CHECK: %[[VAL_19:.*]] = fir.insert_value %[[VAL_17]], %[[VAL_18]], ["j", !fir.type<_QMtinitTt1{{.*}}>] : (!fir.type<_QMtinitTt1{{.*}}>, i32) -> !fir.type<_QMtinitTt1{{.*}}>
  ! CHECK: %[[VAL_20:.*]] = fir.address_of(@_QMtinitEziel) : !fir.ref<!fir.array<100xf32>>
  ! CHECK: %[[VAL_21:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_22:.*]] = fir.embox %[[VAL_20]](%[[VAL_21]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<100xf32>>
  ! CHECK: %[[VAL_22_B:.*]] = fir.rebox %[[VAL_22]] : (!fir.box<!fir.array<100xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: %[[VAL_23:.*]] = fir.insert_value %[[VAL_19]], %[[VAL_22_B]], ["x", !fir.type<_QMtinitTt1{{.*}}>] : (!fir.type<_QMtinitTt1{{.*}}>, !fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.type<_QMtinitTt1{{.*}}>
  ! CHECK: %[[VAL_24:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
  ! CHECK: %[[VAL_25:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_26:.*]] = fir.embox %[[VAL_24]](%[[VAL_25]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: %[[VAL_27:.*]] = fir.insert_value %[[VAL_23]], %[[VAL_26]], ["y", !fir.type<_QMtinitTt1{{.*}}>] : (!fir.type<_QMtinitTt1{{.*}}>, !fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.type<_QMtinitTt1{{.*}}>
  ! CHECK: %[[VAL_28:.*]] = fir.insert_value %[[VAL_27]], %[[VAL_26]], ["z", !fir.type<_QMtinitTt1{{.*}}>] : (!fir.type<_QMtinitTt1{{.*}}>, !fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.type<_QMtinitTt1{{.*}}>
  ! CHECK: %[[VAL_29:.*]] = fir.string_lit "hello     "(10) : !fir.char<1,10>
  ! CHECK: %[[VAL_30:.*]] = fir.insert_value %[[VAL_28]], %[[VAL_29]], ["c1", !fir.type<_QMtinitTt1{{.*}}>] : (!fir.type<_QMtinitTt1{{.*}}>, !fir.char<1,10>) -> !fir.type<_QMtinitTt1{{.*}}>
  ! CHECK: %[[VAL_31:.*]] = fir.zero_bits !fir.char<1,10>
  ! CHECK: %[[VAL_32:.*]] = fir.insert_value %[[VAL_30]], %[[VAL_31]], ["c2", !fir.type<_QMtinitTt1{{.*}}>] : (!fir.type<_QMtinitTt1{{.*}}>, !fir.char<1,10>) -> !fir.type<_QMtinitTt1{{.*}}>
  ! CHECK: %[[VAL_33:.*]] = fir.undefined !fir.type<_QMtinitTt0{k:i32}>
  ! CHECK: %[[VAL_34:.*]] = fir.insert_value %[[VAL_33]], %[[VAL_15]], ["k", !fir.type<_QMtinitTt0{k:i32}>] : (!fir.type<_QMtinitTt0{k:i32}>, i32) -> !fir.type<_QMtinitTt0{k:i32}>
  ! CHECK: %[[VAL_35:.*]] = fir.insert_value %[[VAL_32]], %[[VAL_34]], ["somet0", !fir.type<_QMtinitTt1{{.*}}>] : (!fir.type<_QMtinitTt1{{.*}}>, !fir.type<_QMtinitTt0{k:i32}>) -> !fir.type<_QMtinitTt1{{.*}}>
  ! CHECK: %[[VAL_36:.*]] = fir.zero_bits !fir.type<_QMtinitTtno_init{k:i32}>
  ! CHECK: %[[VAL_37:.*]] = fir.insert_value %[[VAL_35]], %[[VAL_36]], ["sometno_init", !fir.type<_QMtinitTt1{{.*}}>] : (!fir.type<_QMtinitTt1{{.*}}>, !fir.type<_QMtinitTtno_init{k:i32}>) -> !fir.type<_QMtinitTt1{{.*}}>
  ! CHECK: %[[VAL_38:.*]] = fir.insert_value %[[VAL_33]], %[[VAL_14]], ["k", !fir.type<_QMtinitTt0{k:i32}>] : (!fir.type<_QMtinitTt0{k:i32}>, i32) -> !fir.type<_QMtinitTt0{k:i32}>
  ! CHECK: %[[VAL_39:.*]] = fir.insert_value %[[VAL_37]], %[[VAL_38]], ["somet0_2", !fir.type<_QMtinitTt1{{.*}}>] : (!fir.type<_QMtinitTt1{{.*}}>, !fir.type<_QMtinitTt0{k:i32}>) -> !fir.type<_QMtinitTt1{{.*}}>
  ! CHECK: %[[VAL_40:.*]] = fir.insert_value %[[VAL_39]], %[[VAL_34]], ["somet0_array", !fir.type<_QMtinitTt1{{.*}}>] : (!fir.type<_QMtinitTt1{{.*}}>, !fir.type<_QMtinitTt0{k:i32}>) -> !fir.type<_QMtinitTt1{{.*}}>
  ! CHECK: fir.has_value %[[VAL_40]] : !fir.type<_QMtinitTt1{{.*}}>

  ! Test a type extending other type with a default init
  type(textendst0) :: etextendst0
! CHECK-LABEL: @_QMtinitEetextendst0 : !fir.type<_QMtinitTtextendst0{k:i32,l:i32}> {
  ! CHECK: %[[VAL_42:.*]] = arith.constant 66 : i32
  ! CHECK: %[[VAL_43:.*]] = fir.undefined !fir.type<_QMtinitTtextendst0{k:i32,l:i32}>
  ! CHECK: %[[VAL_44:.*]] = fir.insert_value %[[VAL_43]], %[[VAL_42]], ["k", !fir.type<_QMtinitTtextendst0{k:i32,l:i32}>] : (!fir.type<_QMtinitTtextendst0{k:i32,l:i32}>, i32) -> !fir.type<_QMtinitTtextendst0{k:i32,l:i32}>
  ! CHECK: %[[VAL_45:.*]] = fir.zero_bits i32
  ! CHECK: %[[VAL_46:.*]] = fir.insert_value %[[VAL_44]], %[[VAL_45]], ["l", !fir.type<_QMtinitTtextendst0{k:i32,l:i32}>] : (!fir.type<_QMtinitTtextendst0{k:i32,l:i32}>, i32) -> !fir.type<_QMtinitTtextendst0{k:i32,l:i32}>
  ! CHECK: fir.has_value %[[VAL_46]] : !fir.type<_QMtinitTtextendst0{k:i32,l:i32}>

  type(tv) :: withmold = tv(null(mv))
  ! CHECK-LABEL: fir.global @_QMtinitEwithmold
  ! CHECK: %[[C0:.*]] = arith.constant 0 : index
  ! CHECK: %[[UNDEF:.*]] = fir.undefined !fir.type<_QMtinitTtv{v:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
  ! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
  ! CHECK: %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
  ! CHECK: %[[ZEROBOX:.*]] = fir.embox %[[ZERO]](%[[SHAPE]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: %[[RET:.*]] = fir.insert_value %[[UNDEF]], %[[ZEROBOX]], ["v", !fir.type<_QMtinitTtv{v:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>] : (!fir.type<_QMtinitTtv{v:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>, !fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.type<_QMtinitTtv{v:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
  ! CHECK: fir.has_value %[[RET]] : !fir.type<_QMtinitTtv{v:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>

end module


! Test that default initialization is also applied to saved variables
subroutine saved()
  use tinit
  type(t0), save :: savedt0
! CHECK-LABEL: fir.global internal @_QFsavedEsavedt0 : !fir.type<_QMtinitTt0{k:i32}> {
  ! CHECK: %[[VAL_47:.*]] = arith.constant 66 : i32
  ! CHECK: %[[VAL_48:.*]] = fir.undefined !fir.type<_QMtinitTt0{k:i32}>
  ! CHECK: %[[VAL_49:.*]] = fir.insert_value %[[VAL_48]], %[[VAL_47]], ["k", !fir.type<_QMtinitTt0{k:i32}>] : (!fir.type<_QMtinitTt0{k:i32}>, i32) -> !fir.type<_QMtinitTt0{k:i32}>
  ! CHECK: fir.has_value %[[VAL_49]] : !fir.type<_QMtinitTt0{k:i32}>
end subroutine

! Test default initialization in equivalences
subroutine eqv()
  use tinit
  type(tseq), save :: somet
  integer :: i(2)
  equivalence (somet, i)
! CHECK-LABEL: fir.global internal @_QFeqvEi : !fir.array<2xi32> {
  ! CHECK-DAG: %[[VAL_50:.*]] = arith.constant 2 : i32
  ! CHECK-DAG: %[[VAL_51:.*]] = arith.constant 3 : i32
  ! CHECK: %[[VAL_52:.*]] = fir.undefined !fir.array<2xi32>
  ! CHECK: %[[VAL_53:.*]] = fir.insert_value %[[VAL_52]], %[[VAL_50]], [0 : index] : (!fir.array<2xi32>, i32) -> !fir.array<2xi32>
  ! CHECK: %[[VAL_54:.*]] = fir.insert_value %[[VAL_53]], %[[VAL_51]], [1 : index] : (!fir.array<2xi32>, i32) -> !fir.array<2xi32>
  ! CHECK: fir.has_value %[[VAL_54]] : !fir.array<2xi32>
end subroutine

subroutine eqv_explicit_init()
  use tinit
  type(tseq), save :: somet
  integer :: i(2) = [4, 5]
  equivalence (somet, i)
! CHECK-LABEL: fir.global internal @_QFeqv_explicit_initEi : !fir.array<2xi32> {
  ! CHECK-DAG: %[[VAL_57:.*]] = arith.constant 4 : i32
  ! CHECK-DAG: %[[VAL_58:.*]] = arith.constant 5 : i32
  ! CHECK: %[[VAL_59:.*]] = fir.undefined !fir.array<2xi32>
  ! CHECK: %[[VAL_60:.*]] = fir.insert_value %[[VAL_59]], %[[VAL_57]], [0 : index] : (!fir.array<2xi32>, i32) -> !fir.array<2xi32>
  ! CHECK: %[[VAL_61:.*]] = fir.insert_value %[[VAL_60]], %[[VAL_58]], [1 : index] : (!fir.array<2xi32>, i32) -> !fir.array<2xi32>
  ! CHECK: fir.has_value %[[VAL_61]] : !fir.array<2xi32>
end subroutine

subroutine eqv_same_default_init()
  use tinit
  type(tseq), save :: somet1(2), somet2
  equivalence (somet1(1), somet2)
! CHECK-LABEL: fir.global internal @_QFeqv_same_default_initEsomet1 : !fir.array<2xi64> {
  ! CHECK-LE: %[[VAL_62:.*]] = arith.constant 12884901890 : i64
  ! CHECK-BE: %[[VAL_62:.*]] = arith.constant 8589934595 : i64
  ! CHECK: %[[VAL_63:.*]] = fir.undefined !fir.array<2xi64>
  ! CHECK: %[[VAL_64:.*]] = fir.insert_on_range %[[VAL_63]], %[[VAL_62]] from (0) to (1) : (!fir.array<2xi64>, i64) -> !fir.array<2xi64>
  ! CHECK: fir.has_value %[[VAL_64]] : !fir.array<2xi64>
end subroutine

subroutine eqv_full_overlaps_with_explicit_init()
  use tinit
  type(tseq), save :: somet
  integer, save :: link(4)
  integer :: i(2) = [5, 6]
  integer :: j(2) = [7, 8]
  ! Equivalence: somet fully covered by explicit init.
  !   i(1)=5 | i(2)=6  |    -    |  -
  !     -    | somet%i | somet%j |
  !     -    |    -    | j(1)=7  | j(2)=8
  equivalence (i, link(1))
  equivalence (somet, link(2))
  equivalence (j, link(3))
! CHECK-LABEL: fir.global internal @_QFeqv_full_overlaps_with_explicit_initEi : !fir.array<4xi32> {
  ! CHECK-DAG: %[[VAL_73:.*]] = arith.constant 5 : i32
  ! CHECK-DAG: %[[VAL_74:.*]] = arith.constant 6 : i32
  ! CHECK-DAG: %[[VAL_75:.*]] = arith.constant 7 : i32
  ! CHECK-DAG: %[[VAL_76:.*]] = arith.constant 8 : i32
  ! CHECK-DAG: %[[VAL_77:.*]] = fir.undefined !fir.array<4xi32>
  ! CHECK-DAG: %[[VAL_78:.*]] = fir.insert_value %[[VAL_77]], %[[VAL_73]], [0 : index] : (!fir.array<4xi32>, i32) -> !fir.array<4xi32>
  ! CHECK-DAG: %[[VAL_79:.*]] = fir.insert_value %[[VAL_78]], %[[VAL_74]], [1 : index] : (!fir.array<4xi32>, i32) -> !fir.array<4xi32>
  ! CHECK-DAG: %[[VAL_80:.*]] = fir.insert_value %[[VAL_79]], %[[VAL_75]], [2 : index] : (!fir.array<4xi32>, i32) -> !fir.array<4xi32>
  ! CHECK-DAG: %[[VAL_81:.*]] = fir.insert_value %[[VAL_80]], %[[VAL_76]], [3 : index] : (!fir.array<4xi32>, i32) -> !fir.array<4xi32>
  ! CHECK-DAG: fir.has_value %[[VAL_81]] : !fir.array<4xi32>
end subroutine

subroutine eqv_partial_overlaps_with_explicit_init()
  use tinit
  type(tseq), save :: somet
  integer, save :: link(4)
  integer :: i(2) = [5, 6]
  integer :: j = 7
  ! `somet` is only partially covered by explicit init, somet%j default
  ! init value should be used in the equiv storage init to match nvfortran,
  ! ifort, and nagfor behavior (gfortran refuses this code). 19.5.3.4 point
  ! 10 specifies that explicit initialization overrides default initialization.
  !   i(1)=5 | i(2)=6  |    -    |  -
  !     -    | somet%i | somet%j |
  !     -    |    -    |    -    | j=7
  equivalence (i, link(1))
  equivalence (somet, link(2))
  equivalence (j, link(4))
! CHECK-LABEL: fir.global internal @_QFeqv_partial_overlaps_with_explicit_initEi : !fir.array<4xi32>
   ! CHECK-DAG: %[[VAL_82:.*]] = arith.constant 5 : i32
   ! CHECK-DAG: %[[VAL_83:.*]] = arith.constant 6 : i32
   ! CHECK-DAG: %[[VAL_84:.*]] = arith.constant 3 : i32
   ! CHECK-DAG: %[[VAL_85:.*]] = arith.constant 7 : i32
   ! CHECK: %[[VAL_86:.*]] = fir.undefined !fir.array<4xi32>
   ! CHECK: %[[VAL_87:.*]] = fir.insert_value %[[VAL_86]], %[[VAL_82]], [0 : index] : (!fir.array<4xi32>, i32) -> !fir.array<4xi32>
   ! CHECK: %[[VAL_88:.*]] = fir.insert_value %[[VAL_87]], %[[VAL_83]], [1 : index] : (!fir.array<4xi32>, i32) -> !fir.array<4xi32>
   ! CHECK: %[[VAL_89:.*]] = fir.insert_value %[[VAL_88]], %[[VAL_84]], [2 : index] : (!fir.array<4xi32>, i32) -> !fir.array<4xi32>
   ! CHECK: %[[VAL_90:.*]] = fir.insert_value %[[VAL_89]], %[[VAL_85]], [3 : index] : (!fir.array<4xi32>, i32) -> !fir.array<4xi32>
   ! CHECK: fir.has_value %[[VAL_90]] : !fir.array<4xi32>
end subroutine
