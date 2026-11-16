! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s
! Nested constants must retain expanded array components outside DATA.

module nested_constructor
  type pair
    character(3) :: c(4)
  end type
  type holder
    integer tag
    type(pair) p
  end type
  type(holder), parameter :: hp = holder(7, pair('abc'))
  character(3), parameter :: first = hp%p%c(2)(1:3)
  integer, parameter :: n = size(hp%p%c)
  type(holder) :: initialized = holder(7, pair('abc'))
  type wrapper
    type(holder) :: value = holder(7, pair('abc'))
  end type
contains
  subroutine assign(x)
    type(holder) x
    x = holder(7, pair('abc'))
  end
  subroutine inquire(c, count)
    character(3) c
    integer count
    c = first
    count = n
  end
end

! CHECK-LABEL: func.func @_QMnested_constructorPassign(
! CHECK-NOT: ctor.temp
! CHECK: fir.address_of(@[[CTOR:[^ ]+]])
! CHECK: hlfir.assign
! CHECK-LABEL: func.func @_QMnested_constructorPinquire(
! CHECK: arith.constant 4 : i32
! CHECK-DAG: fir.global internal @[[CTOR]] constant
! CHECK-DAG: fir.global @_QMnested_constructorEinitialized
! CHECK-DAG: fir.insert_on_range {{.*}} from (0) to (3) : (!fir.array<4x!fir.char<1,3>>,
