! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Tests the different possible type involving polymorphic entities. 

module polymorphic_types
  type p1
    integer :: a
    integer :: b
  contains
    procedure :: polymorphic_dummy
  end type
contains

! ------------------------------------------------------------------------------
! Test polymorphic entity types
! ------------------------------------------------------------------------------

  subroutine polymorphic_dummy(p)
    class(p1) :: p
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_typesPpolymorphic_dummy(
! CHECK-SAME: %{{.*}}: !fir.class<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>

  subroutine polymorphic_dummy_assumed_shape_array(pa)
    class(p1) :: pa(:)
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_typesPpolymorphic_dummy_assumed_shape_array(
! CHECK-SAME: %{{.*}}: !fir.class<!fir.array<?x!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>>

  subroutine polymorphic_dummy_explicit_shape_array(pa)
    class(p1) :: pa(10)
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_typesPpolymorphic_dummy_explicit_shape_array(
! CHECK-SAME: %{{.*}}: !fir.class<!fir.array<10x!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>>

  subroutine polymorphic_allocatable(p)
    class(p1), allocatable :: p
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_typesPpolymorphic_allocatable(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>>>

  subroutine polymorphic_pointer(p)
    class(p1), pointer :: p
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_typesPpolymorphic_pointer(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>>>

  subroutine polymorphic_allocatable_intentout(p)
    class(p1), allocatable, intent(out) :: p
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_typesPpolymorphic_allocatable_intentout(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>>>
! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[ARG0]] : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableDeallocate(%[[BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

! ------------------------------------------------------------------------------
! Test unlimited polymorphic dummy argument types
! ------------------------------------------------------------------------------

  subroutine unlimited_polymorphic_dummy(u)
    class(*) :: u
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_typesPunlimited_polymorphic_dummy(
! CHECK-SAME: %{{.*}}: !fir.class<none>

  subroutine unlimited_polymorphic_assumed_shape_array(ua)
    class(*) :: ua(:)
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_typesPunlimited_polymorphic_assumed_shape_array(
! CHECK-SAME: %{{.*}}: !fir.class<!fir.array<?xnone>>

  subroutine unlimited_polymorphic_explicit_shape_array(ua)
    class(*) :: ua(20)
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_typesPunlimited_polymorphic_explicit_shape_array(
! CHECK-SAME: %{{.*}}: !fir.class<!fir.array<20xnone>>

  subroutine unlimited_polymorphic_allocatable(p)
    class(*), allocatable :: p
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_typesPunlimited_polymorphic_allocatable(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.class<!fir.heap<none>>>

  subroutine unlimited_polymorphic_pointer(p)
    class(*), pointer :: p
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_typesPunlimited_polymorphic_pointer(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.class<!fir.ptr<none>>>

! ------------------------------------------------------------------------------
! Test polymorphic function return types
! ------------------------------------------------------------------------------

  function ret_polymorphic_allocatable() result(ret)
    class(p1), allocatable :: ret
  end function

! CHECK-LABEL: func.func @_QMpolymorphic_typesPret_polymorphic_allocatable() -> !fir.class<!fir.heap<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>>
! CHECK: %[[MEM:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>> {bindc_name = "ret", uniq_name = "_QMpolymorphic_typesFret_polymorphic_allocatableEret"}
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>
! CHECK: %[[BOX:.*]] = fir.embox %[[ZERO]] : (!fir.heap<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>) -> !fir.class<!fir.heap<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>>
! CHECK: fir.store %[[BOX]] to %[[MEM]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>>>
! CHECK: %[[LOAD:.*]] = fir.load %[[MEM]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>>>
! CHECK: return %[[LOAD]] : !fir.class<!fir.heap<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>>

  function ret_polymorphic_pointer() result(ret)
    class(p1), pointer :: ret
  end function

! CHECK-LABEL: func.func @_QMpolymorphic_typesPret_polymorphic_pointer() -> !fir.class<!fir.ptr<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>>
! CHECK: %[[MEM:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>> {bindc_name = "ret", uniq_name = "_QMpolymorphic_typesFret_polymorphic_pointerEret"}
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.ptr<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>
! CHECK: %[[BOX:.*]] = fir.embox %[[ZERO]] : (!fir.ptr<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>) -> !fir.class<!fir.ptr<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>>
! CHECK: fir.store %[[BOX]] to %[[MEM]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>>>
! CHECK: %[[LOAD:.*]] = fir.load %[[MEM]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>>>
! CHECK: return %[[LOAD]] : !fir.class<!fir.ptr<!fir.type<_QMpolymorphic_typesTp1{a:i32,b:i32}>>>

! ------------------------------------------------------------------------------
! Test unlimited polymorphic function return types
! ------------------------------------------------------------------------------

  function ret_unlimited_polymorphic_allocatable() result(ret)
    class(*), allocatable :: ret
  end function

! CHECK-LABEL: func.func @_QMpolymorphic_typesPret_unlimited_polymorphic_allocatable() -> !fir.class<!fir.heap<none>>
! CHECK: %[[MEM:.*]] = fir.alloca !fir.class<!fir.heap<none>> {bindc_name = "ret", uniq_name = "_QMpolymorphic_typesFret_unlimited_polymorphic_allocatableEret"}
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.heap<none>
! CHECK: %[[BOX:.*]] = fir.embox %[[ZERO]] : (!fir.heap<none>) -> !fir.class<!fir.heap<none>>
! CHECK: fir.store %[[BOX]] to %[[MEM]] : !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK: %[[LOAD:.*]] = fir.load %[[MEM]] : !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK: return %[[LOAD]] : !fir.class<!fir.heap<none>>

  function ret_unlimited_polymorphic_pointer() result(ret)
    class(*), pointer :: ret
  end function

! CHECK-LABEL: func.func @_QMpolymorphic_typesPret_unlimited_polymorphic_pointer() -> !fir.class<!fir.ptr<none>>
! CHECK: %[[MEM:.*]] = fir.alloca !fir.class<!fir.ptr<none>> {bindc_name = "ret", uniq_name = "_QMpolymorphic_typesFret_unlimited_polymorphic_pointerEret"}
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.ptr<none>
! CHECK: %[[BOX:.*]] = fir.embox %[[ZERO]] : (!fir.ptr<none>) -> !fir.class<!fir.ptr<none>>
! CHECK: fir.store %[[BOX]] to %[[MEM]] : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK: %[[LOAD:.*]] = fir.load %[[MEM]] : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK: return %[[LOAD]] : !fir.class<!fir.ptr<none>>

! ------------------------------------------------------------------------------
! Test assumed type argument types
! ------------------------------------------------------------------------------

  ! Follow up patch will add a `fir.assumed_type` attribute to the types in the
  ! two tests below.
  subroutine assumed_type_dummy(a) bind(c)
    type(*) :: a
  end subroutine assumed_type_dummy

  ! CHECK-LABEL: func.func @assumed_type_dummy(
  ! CHECK-SAME: %{{.*}}: !fir.class<none>

  subroutine assumed_type_dummy_array(a) bind(c)
    type(*) :: a(:)
  end subroutine assumed_type_dummy_array

  ! CHECK-LABEL: func.func @assumed_type_dummy_array(
  ! CHECK-SAME: %{{.*}}: !fir.class<!fir.array<?xnone>>
end module
