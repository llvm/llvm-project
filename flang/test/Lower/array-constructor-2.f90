! RUN: bbc %s -o - | FileCheck %s

!  Constant array ctor.
! CHECK-LABEL: func @_QPtest1(
subroutine test1(a, b)
  real :: a(3)
  integer :: b(4)
  integer, parameter :: constant_array(4) = [6, 7, 42, 9]

  ! Array ctors for constant arrays should be outlined as constant globals.

  !  Look at inline constructor case
  ! CHECK: %{{.*}} = fir.address_of(@_QQro.3xr4.6e55f044605a4991f15fd4505d83faf4) : !fir.ref<!fir.array<3xf32>>
  a = (/ 1.0, 2.0, 3.0 /)

  !  Look at PARAMETER case
  ! CHECK: %{{.*}} = fir.address_of(@_QQro.4xi4.6a6af0eea868c84da59807d34f7e1a86) : !fir.ref<!fir.array<4xi32>>
  b = constant_array
end subroutine test1

!  Dynamic array ctor with constant extent.
! CHECK-LABEL: func @_QPtest2(
! CHECK-SAME: %[[a:[^:]*]]: !fir.ref<!fir.array<5xf32>>{{.*}}, %[[b:[^:]*]]: !fir.ref<f32>{{.*}})
subroutine test2(a, b)
  real :: a(5), b
  real, external :: f

  !  Look for the 5 store patterns
  ! CHECK: %[[tmp:.*]] = fir.allocmem !fir.array<5xf32>
  ! CHECK: %[[val:.*]] = fir.call @_QPf(%[[b]]) : (!fir.ref<f32>) -> f32
  ! CHECK: %[[loc:.*]] = fir.coordinate_of %{{.*}}, %{{.*}} : (!fir.heap<!fir.array<5xf32>>, index) -> !fir.ref<f32>
  ! CHECK: fir.store %[[val]] to %[[loc]] : !fir.ref<f32>
  ! CHECK: fir.call @_QPf(%{{.*}}) : (!fir.ref<f32>) -> f32
  ! CHECK: fir.coordinate_of %{{.*}}, %{{.*}} : (!fir.heap<!fir.array<5xf32>>, index) -> !fir.ref<f32>
  ! CHECK: fir.store
  ! CHECK: fir.call @_QPf(
  ! CHECK: fir.coordinate_of %
  ! CHECK: fir.store
  ! CHECK: fir.call @_QPf(
  ! CHECK: fir.coordinate_of %
  ! CHECK: fir.store
  ! CHECK: fir.call @_QPf(
  ! CHECK: fir.coordinate_of %
  ! CHECK: fir.store

  !  After the ctor done, loop to copy result to `a`
  ! CHECK-DAG: fir.array_coor %[[tmp:.*]](%
  ! CHECK-DAG: %[[ai:.*]] = fir.array_coor %[[a]](%
  ! CHECK: fir.store %{{.*}} to %[[ai]] : !fir.ref<f32>
  ! CHECK: fir.freemem %[[tmp]]

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
  ! CHECK: %{{.*}}:3 = fir.box_dims %{{.*}}, %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK: %{{.*}} = fir.box_addr %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK: %[[tmp:.*]] = fir.allocmem f32, %c32
  call test3b(b)
  ! CHECK: %[[hp1:.*]] = fir.allocmem !fir.array<?xf32>, %{{.*}} {uniq_name = ".array.expr"}
  ! CHECK-DAG: %[[rep:.*]] = fir.convert %{{.*}} : (!fir.heap<f32>) -> !fir.ref<i8>
  ! CHECK-DAG: %[[res:.*]] = fir.convert %{{.*}} : (index) -> i64
  ! CHECK: %{{.*}} = fir.call @realloc(%[[rep]], %[[res]]) : (!fir.ref<i8>, i64) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memcpy.p0.p0.i64(%{{.*}}, %{{.*}}, %{{.*}}, %false{{.*}}) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: fir.call @_QPtest3c
  ! CHECK: fir.save_result
  ! CHECK: %[[tmp2:.*]] = fir.allocmem !fir.array<?xf32>, %{{.*}}#1 {uniq_name = ".array.expr"}
  ! CHECK: fir.call @realloc
  ! CHECK: fir.call @llvm.memcpy.p0.p0.i64(%
  ! CHECK: fir.array_coor %[[tmp:.*]](%{{.*}}) %{{.*}} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK-NEXT: fir.load
  ! CHECK-NEXT: fir.array_coor %arg0 %{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  ! CHECK-NEXT: fir.store
  ! CHECK: fir.freemem %[[tmp]]
  ! CHECK: fir.freemem %[[tmp2]]
  ! CHECK: %[[alli:.*]] = fir.box_addr %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK: fir.freemem %[[alli]]
  ! CHECK: fir.freemem %[[hp1]]
  a = (/ b, test3c() /)
end subroutine test3

! CHECK-LABEL: func @_QPtest4(
subroutine test4(a, b, n1, m1)
  real :: a(:)
  real :: b(:,:)
  integer, external :: f1, f2, f3

  !  Dynamic array ctor with dynamic extent using implied do loops.
  ! CHECK-DAG: fir.alloca index {bindc_name = ".buff.pos"}
  ! CHECK-DAG: fir.alloca index {bindc_name = ".buff.size"}
  ! CHECK-DAG: %[[c32:.*]] = arith.constant 32 : index
  ! CHECK: fir.allocmem f32, %[[c32]]
  ! CHECK: fir.call @_QPf1(%{{.*}}) : (!fir.ref<i32>) -> i32
  ! CHECK: fir.call @_QPf2(%arg2) : (!fir.ref<i32>) -> i32
  ! CHECK: fir.call @_QPf3(%{{.*}}) : (!fir.ref<i32>) -> i32
  ! CHECK: %[[q:.*]] = fir.coordinate_of %arg1, %{{.*}}, %{{.*}} : (!fir.box<!fir.array<?x?xf32>>, i64, i64) -> !fir.ref<f32>
  ! CHECK: %[[q2:.*]] = fir.load %[[q]] : !fir.ref<f32>
  ! CHECK: fir.store %[[q2]] to %{{.*}} : !fir.ref<f32>
  ! CHECK: fir.freemem %{{.*}}
  ! CHECK-NEXT: return
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
  ! CHECK: %[[res:.*]] = fir.allocmem !fir.array<4xf32>
  ! CHECK: fir.address_of(@_QQro.2xr4.057a7f5ab69cb695657046b18832c330) : !fir.ref<!fir.array<2xf32>>
  ! CHECK: %[[tmp1:.*]] = fir.allocmem !fir.array<2xf32>
  ! CHECK: fir.call @llvm.memcpy.p0.p0.i64(%{{.*}}, %{{.*}}, %{{.*}}, %false{{.*}}) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[tmp2:.*]] = fir.allocmem !fir.array<2xf32>
  ! CHECK: = fir.array_coor %[[array2]](%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: = fir.array_coor %[[tmp2]](%{{.*}}) %{{.*}} : (!fir.heap<!fir.array<2xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.call @llvm.memcpy.p0.p0.i64(%{{.*}}, %{{.*}}, %{{.*}}, %false{{.*}}) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: = fir.array_coor %{{.*}}(%{{.*}}) %{{.*}} : (!fir.heap<!fir.array<4xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: = fir.array_coor %[[a]] %{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  ! CHECK-DAG: fir.freemem %{{.*}}
  ! CHECK-DAG: fir.freemem %[[tmp2]]
  ! CHECK-DAG: fir.freemem %[[tmp1]]
  ! CHECK: return
  a = [ const_array1, array2 ]
end subroutine test5

! CHECK-LABEL: func @_QPtest6(
subroutine test6(c, d, e)
  character(5) :: c(3)
  character(5) :: d, e
  ! CHECK: = fir.allocmem !fir.array<2x!fir.char<1,5>>
  ! CHECK: fir.call @realloc
  ! CHECK: %[[t:.*]] = fir.coordinate_of %{{.*}}, %{{.*}} : (!fir.heap<!fir.array<2x!fir.char<1,5>>>, index) -> !fir.ref<!fir.char<1,5>>
  ! CHECK: %[[to:.*]] = fir.convert %[[t]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memcpy.p0.p0.i64(%[[to]], %{{.*}}, %{{.*}}, %false) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: fir.call @realloc
  ! CHECK: %[[t:.*]] = fir.coordinate_of %{{.*}}, %{{.*}} : (!fir.heap<!fir.array<2x!fir.char<1,5>>>, index) -> !fir.ref<!fir.char<1,5>>
  ! CHECK: %[[to:.*]] = fir.convert %[[t]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memcpy.p0.p0.i64(%[[to]], %{{.*}}, %{{.*}}, %false) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: fir.freemem %{{.*}}
  c = (/ d, e /)
end subroutine test6

! CHECK-LABEL: func @_QPtest7(
! CHECK: %[[i:.*]] = fir.convert %{{.*}} : (index) -> i8
! CHECK: %[[und:.*]] = fir.undefined !fir.char<1>
! CHECK: %[[scalar:.*]] = fir.insert_value %[[und]], %[[i]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK: ^bb{{[0-9]+}}(%{{.*}}: !fir.heap<!fir.char<1>>):  // 2 preds
! CHECK: fir.store %[[scalar]] to %{{.*}} : !fir.ref<!fir.char<1>>
subroutine test7(a, n)
  character(1) :: a(n)
  a = (/ (CHAR(i), i=1,n) /)
end subroutine test7

! CHECK: fir.global internal @_QQro.3xr4.{{.*}}(dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>) constant : !fir.array<3xf32>

! CHECK: fir.global internal @_QQro.4xi4.{{.*}}(dense<[6, 7, 42, 9]> : tensor<4xi32>) constant : !fir.array<4xi32>
