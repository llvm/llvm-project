! Test lowering of empty character array constructors with non-default
! kinds: the element type must carry the constructor's character kind
! (!fir.char<4>/!fir.char<2>), not default !fir.char<1>.
! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_empty_char4(
! CHECK: fir.address_of(@_QQro.0x0xc4.null{{.*}}) : !fir.ref<!fir.array<0x!fir.char<4,0>>>
! CHECK: hlfir.assign %{{.*}} to %{{.*}} realloc : !fir.ref<!fir.array<0x!fir.char<4,0>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<4,?>>>>>
subroutine test_empty_char4(c)
  character(kind=4, len=:), allocatable :: c(:)
  c = [character(kind=4, len=0) ::]
end subroutine

! CHECK-LABEL: func.func @_QPtest_empty_char4_len3(
! CHECK: fir.address_of(@_QQro.0x3xc4.null{{.*}}) : !fir.ref<!fir.array<0x!fir.char<4,3>>>
! CHECK: hlfir.assign %{{.*}} to %{{.*}} realloc : !fir.ref<!fir.array<0x!fir.char<4,3>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<4,?>>>>>
subroutine test_empty_char4_len3(c)
  character(kind=4, len=:), allocatable :: c(:)
  c = [character(kind=4, len=3) ::]
end subroutine

! CHECK-LABEL: func.func @_QPtest_empty_char2(
! CHECK: fir.address_of(@_QQro.0x0xc2.null{{.*}}) : !fir.ref<!fir.array<0x!fir.char<2,0>>>
! CHECK: hlfir.assign %{{.*}} to %{{.*}} realloc : !fir.ref<!fir.array<0x!fir.char<2,0>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<2,?>>>>>
subroutine test_empty_char2(c)
  character(kind=2, len=:), allocatable :: c(:)
  c = [character(kind=2, len=0) ::]
end subroutine

! CHECK-DAG: fir.global internal @_QQro.0x0xc4.null{{.*}} constant : !fir.array<0x!fir.char<4,0>>
! CHECK-DAG: fir.global internal @_QQro.0x3xc4.null{{.*}} constant : !fir.array<0x!fir.char<4,3>>
! CHECK-DAG: fir.global internal @_QQro.0x0xc2.null{{.*}} constant : !fir.array<0x!fir.char<2,0>>
