! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

!  Constant array ctor.
! CHECK-LABEL: func @_QPtest1(
subroutine test1(a, b)
  real :: a(3)
  integer :: b(4)
  integer, parameter :: constant_array(4) = [6, 7, 42, 9]

  ! Array ctors for constant arrays should be outlined as constant globals.

  !  Look at inline constructor case
  ! CHECK: %[[CONST:.*]] = fir.address_of(@_QQro.3xr4.0) : !fir.ref<!fir.array<3xf32>>
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[CONST]](%{{.*}}) {{{.*}}uniq_name = "_QQro.3xr4.0"}
  ! CHECK: hlfir.assign %[[DECL]]#0 to %{{.*}}
  a = (/ 1.0, 2.0, 3.0 /)

  !  Look at PARAMETER case
  ! CHECK: %[[CONST2:.*]] = fir.address_of(@_QQro.4xi4.1) : !fir.ref<!fir.array<4xi32>>
  ! CHECK: %[[DECL2:.*]]:2 = hlfir.declare %[[CONST2]](%{{.*}}) {{{.*}}uniq_name = "_QQro.4xi4.1"}
  ! CHECK: hlfir.assign %[[DECL2]]#0 to %{{.*}}
  b = constant_array
end subroutine test1

!  Dynamic array ctor with constant extent.
! CHECK-LABEL: func @_QPtest2(
! CHECK-SAME: %[[a:[^:]*]]: !fir.ref<!fir.array<5xf32>>{{.*}}, %[[b:[^:]*]]: !fir.ref<f32>{{.*}})
subroutine test2(a, b)
  real :: a(5), b
  real, external :: f

  !  Look for the 5 store patterns
  ! CHECK: %[[ALLOC:.*]] = fir.allocmem !fir.array<5xf32>
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ALLOC]](%{{.*}})
  ! CHECK: %[[VAL:.*]] = fir.call @_QPf
  ! CHECK: %[[DESIGNATE:.*]] = hlfir.designate %[[DECL]]#0 (%{{.*}})
  ! CHECK: hlfir.assign %[[VAL]] to %[[DESIGNATE]]
  ! CHECK: fir.call @_QPf
  ! CHECK: hlfir.designate
  ! CHECK: hlfir.assign
  ! CHECK: fir.call @_QPf
  ! CHECK: hlfir.designate
  ! CHECK: hlfir.assign
  ! CHECK: fir.call @_QPf
  ! CHECK: hlfir.designate
  ! CHECK: hlfir.assign
  ! CHECK: fir.call @_QPf
  ! CHECK: hlfir.designate
  ! CHECK: hlfir.assign

  !  After the ctor done, loop to copy result to `a`
  ! CHECK: %[[EXPR:.*]] = hlfir.as_expr %[[DECL]]#0
  ! CHECK: hlfir.assign %[[EXPR]] to %{{.*}}
  ! CHECK: hlfir.destroy %[[EXPR]]

  a = [f(b), f(b+1), f(b+2), f(b+5), f(b+11)]
end subroutine test2

!  Dynamic array ctor with dynamic extent.
! CHECK-LABEL: func @_QPtest3(
! CHECK-SAME: %[[a:.*]]: !fir.box<!fir.array<?xf32>>{{.*}})
subroutine test3(a)
  real :: a(:)
  real, allocatable :: b(:), c(:)
  interface
    subroutine test3b(x)
      real, allocatable :: x(:)
    end subroutine test3b
  end interface
  interface
    function test3c
      real, allocatable :: test3c(:)
    end function test3c
  end interface

  ! CHECK: fir.call @_QPtest3b
  call test3b(b)
  ! CHECK: fir.call @_FortranAInitArrayConstructorVector
  ! CHECK: fir.call @_FortranAPushArrayConstructorValue
  ! CHECK: fir.call @_QPtest3c
  ! CHECK: fir.save_result
  ! CHECK: fir.call @_FortranAPushArrayConstructorValue
  ! CHECK: %[[EXPR:.*]] = hlfir.as_expr %{{.*}}
  ! CHECK: hlfir.assign %[[EXPR]] to %{{.*}}
  ! CHECK: hlfir.destroy %[[EXPR]]
  a = (/ b, test3c() /)
end subroutine test3

! CHECK-LABEL: func @_QPtest4(
subroutine test4(a, b, n1, m1)
  real :: a(:)
  real :: b(:,:)
  integer, external :: f1, f2, f3

  !  Dynamic array ctor with dynamic extent using implied do loops.
  ! CHECK: fir.call @_FortranAInitArrayConstructorVector
  ! CHECK: fir.do_loop
  ! CHECK: hlfir.associate
  ! CHECK: fir.call @_QPf1
  ! CHECK: fir.call @_QPf2
  ! CHECK: fir.call @_QPf3
  ! CHECK: fir.do_loop
  ! CHECK: hlfir.designate
  ! CHECK: fir.load
  ! CHECK: fir.store
  ! CHECK: fir.call @_FortranAPushArrayConstructorSimpleScalar
  ! CHECK: hlfir.as_expr
  ! CHECK: hlfir.assign
  a = [ ((b(i,j), j=f1(i),f2(n1),f3(m1+i)), i=1,n1,m1) ]
end subroutine test4

! CHECK-LABEL: func @_QPtest5(
! CHECK-SAME: %[[a:[^:]*]]: !fir.box<!fir.array<?xf32>>{{.*}}, %[[array2:[^:]*]]: !fir.ref<!fir.array<2xf32>>{{.*}})
subroutine test5(a, array2)
  real :: a(:)
  real, parameter :: const_array1(2) = [ 1.0, 2.0 ]
  real :: array2(2)

  !  Array ctor with runtime element values and constant extents.
  !  Concatenation of array values of constant extent.
  ! CHECK: fir.call @_FortranAInitArrayConstructorVector
  ! CHECK: fir.address_of(@_QQro.{{.*}})
  ! CHECK: fir.call @_FortranAPushArrayConstructorValue
  ! CHECK: fir.call @_FortranAPushArrayConstructorValue
  ! CHECK: %[[EXPR:.*]] = hlfir.as_expr
  ! CHECK: hlfir.assign %[[EXPR]]
  ! CHECK: hlfir.destroy %[[EXPR]]
  ! CHECK: return
  a = [ const_array1, array2 ]
end subroutine test5

! CHECK-LABEL: func @_QPtest6(
subroutine test6(c, d, e)
  character(5) :: c(2)
  character(5) :: d, e
  ! CHECK: hlfir.designate
  ! CHECK: hlfir.assign
  ! CHECK: hlfir.designate
  ! CHECK: hlfir.assign
  ! CHECK: %[[EXPR:.*]] = hlfir.as_expr
  ! CHECK: hlfir.assign %[[EXPR]]
  ! CHECK: hlfir.destroy %[[EXPR]]
  c = (/ d, e /)
end subroutine test6

! CHECK-LABEL: func @_QPtest7(
! CHECK: hlfir.elemental
! CHECK: fir.insert_value
! CHECK: hlfir.yield_element
! CHECK: hlfir.assign
subroutine test7(a, n)
  character(1) :: a(n)
  a = (/ (CHAR(i), i=1,n) /)
end subroutine test7

! CHECK: fir.global internal @_QQro.3xr4.0(dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>) constant : !fir.array<3xf32>

! CHECK: fir.global internal @_QQro.4xi4.1(dense<[6, 7, 42, 9]> : tensor<4xi32>) constant : !fir.array<4xi32>
