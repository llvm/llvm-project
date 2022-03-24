! Test lowering of derived type descriptors builtin data
! RUN: bbc -emit-fir %s -o - | FileCheck %s

subroutine foo()
  real, save, target :: init_values(10, 10)
  type sometype
    integer :: num = 42
    real, pointer :: values(:, :) => init_values
  end type
  type(sometype), allocatable, save :: x(:)
end subroutine

! CHECK-LABEL: fir.global linkonce_odr @_QFfooE.n.num constant : !fir.char<1,3> {
! CHECK: %[[res:.*]] = fir.string_lit "num"(3) : !fir.char<1,3>
! CHECK: fir.has_value %[[res]] : !fir.char<1,3>
! CHECK-LABEL: fir.global linkonce_odr @_QFfooE.di.sometype.num constant : i32
! CHECK-LABEL: fir.global linkonce_odr @_QFfooE.n.values constant : !fir.char<1,6> {
! CHECK: %[[res:.*]] = fir.string_lit "values"(6) : !fir.char<1,6>
! CHECK: fir.has_value %[[res]] : !fir.char<1,6>
! CHECK-LABEL: fir.global linkonce_odr @_QFfooE.n.sometype constant : !fir.char<1,8> {
! CHECK: %[[res:.*]] = fir.string_lit "sometype"(8) : !fir.char<1,8>
! CHECK: fir.has_value %[[res]] : !fir.char<1,8>

! CHECK-LABEL: fir.global linkonce_odr @_QFfooE.di.sometype.values constant : !fir.type<_QFfooT.dp.sometype.values{values:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>}> {
  ! CHECK: fir.address_of(@_QFfooEinit_values)
! CHECK: }

! CHECK-LABEL: fir.global linkonce_odr @_QFfooE.dt.sometype constant {{.*}} {
  !CHECK: fir.address_of(@_QFfooE.n.sometype)
  !CHECK: fir.address_of(@_QFfooE.c.sometype)
! CHECK:}

! CHECK-LABEL: fir.global linkonce_odr @_QFfooE.c.sometype constant {{.*}} {
  ! CHECK: fir.address_of(@_QFfooE.n.num)
  ! CHECK: fir.address_of(@_QFfooE.di.sometype.num) : !fir.ref<i32>
  ! CHECK: fir.address_of(@_QFfooE.n.values)
  ! CHECK: fir.address_of(@_QFfooE.di.sometype.values)
! CHECK: }

subroutine char_comp_init()
  implicit none  
  type t
     character(8) :: name='Empty'
  end type t
  type(t) :: a
end subroutine

! CHECK-LABEL: fir.global linkonce_odr @_QFchar_comp_initE.di.t.name constant : !fir.char<1,8> {
! CHECK: %[[res:.*]] = fir.string_lit "Empty   "(8) : !fir.char<1,8>
! CHECK: fir.has_value %[[res]] : !fir.char<1,8>

! CHECK-LABEL: fir.global linkonce_odr @_QFchar_comp_initE.c.t constant : {{.*}} {
  ! CHECK: fir.address_of(@_QFchar_comp_initE.di.t.name) : !fir.ref<!fir.char<1,8>>
! CHECK: }
