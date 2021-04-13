! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest1
subroutine test1(a,b,c,n)
  integer :: n
  real, intent(out) :: a(n)
  real, intent(in) :: b(n), c(n)
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1(%
  ! CHECK-DAG: %[[C:.*]] = fir.array_load %arg2(%
  ! CHECK: %[[T:.*]] = fir.do_loop
  ! CHECK-DAG: %[[Bi:.*]] = fir.array_fetch %[[B]]
  ! CHECK-DAG: %[[Ci:.*]] = fir.array_fetch %[[C]]
  ! CHECK: %[[rv:.*]] = addf %[[Bi]], %[[Ci]]
  ! CHECK: fir.array_update %{{.*}}, %[[rv]], %
  a = b + c
  ! CHECK: fir.array_merge_store %[[A]], %[[T]] to %arg0
end subroutine test1

! CHECK-LABEL: func @_QPtest1b
subroutine test1b(a,b,c,d,n)
  integer :: n
  real, intent(out) :: a(n)
  real, intent(in) :: b(n), c(n), d(n)
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1(%
  ! CHECK-DAG: %[[C:.*]] = fir.array_load %arg2(%
  ! CHECK-DAG: %[[D:.*]] = fir.array_load %arg3(%
  ! CHECK: %[[T:.*]] = fir.do_loop
  ! CHECK-DAG: %[[Bi:.*]] = fir.array_fetch %[[B]]
  ! CHECK-DAG: %[[Ci:.*]] = fir.array_fetch %[[C]]
  ! CHECK: %[[rv1:.*]] = addf %[[Bi]], %[[Ci]]
  ! CHECK: %[[Di:.*]] = fir.array_fetch %[[D]]
  ! CHECK: %[[rv:.*]] = addf %[[rv1]], %[[Di]]
  ! CHECK: fir.array_update %{{.*}}, %[[rv]], %
  a = b + c + d
  ! CHECK: fir.array_merge_store %[[A]], %[[T]] to %arg0
end subroutine test1b

! CHECK-LABEL: func @_QPtest2(
! CHECK-SAME: %[[aarg:[^:]*]]: !fir.box<!fir.array<?xf32>>,
! CHECK-SAME: %[[barg:[^:]+]]: !fir.box<!fir.array<?xf32>>,
! CHECK-SAME: %[[carg:[^:]+]]: !fir.box<!fir.array<?xf32>>)
subroutine test2(a,b,c)
  real, intent(out) :: a(:)
  real, intent(in) :: b(:), c(:)
  ! CHECK: %[[a:.*]] = fir.array_load %[[aarg]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  ! CHECK: %[[b:.*]] = fir.array_load %[[barg]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  ! CHECK: %[[c:.*]] = fir.array_load %[[carg]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  ! CHECK: %{{[^:]+}}:3 = fir.box_dims %[[aarg]], %c0{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  ! CHECK: fir.do_loop {{.*}} iter_args(%{{.*}} = %[[a]]) -> (!fir.array<?xf32>
  ! CHECK: fir.array_fetch %[[b]], %{{.*}} : (!fir.array<?xf32>, index) -> f32
  ! CHECK: fir.array_fetch %[[c]], %{{.*}} : (!fir.array<?xf32>, index) -> f32
  ! CHECK: fir.array_update %{{.*}} : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
  ! CHECK: fir.array_merge_store %[[a]], %{{.*}} to %[[aarg]] : !fir.box<!fir.array<?xf32>>
 a = b + c
end subroutine test2

! CHECK-LABEL: func @_QPtest3
subroutine test3(a,b,c,n)
  integer :: n
  real, intent(out) :: a(n)
  real, intent(in) :: b(n), c
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1(%
  ! CHECK-DAG: %[[C:.*]] = fir.load %arg2
  ! CHECK: %[[T:.*]] = fir.do_loop
  ! CHECK: %[[Bi:.*]] = fir.array_fetch %[[B]]
  ! CHECK: %[[rv:.*]] = addf %[[Bi]], %[[C]]
  ! CHECK: %[[Ti:.*]] = fir.array_update %{{.*}}, %[[rv]], %
  ! CHECK: fir.result %[[Ti]]
  a = b + c
  ! CHECK: fir.array_merge_store %[[A]], %[[T]] to %arg0
end subroutine test3

! CHECK-LABEL: func @_QPtest4
subroutine test4(a,b,c)
! TODO: this declaration fails in CallInterface lowering
!  real, allocatable, intent(out) :: a(:)
  real :: a(100) ! FIXME: fake it for now
  real, intent(in) :: b(:), c
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1
  ! CHECK: fir.do_loop
  ! CHECK: fir.array_fetch %[[B]], %
  ! CHECK: fir.array_update
  a = b + c
  ! CHECK: fir.array_merge_store %[[A]], %{{.*}} to %arg0
end subroutine test4

! CHECK-LABEL: func @_QPtest5
subroutine test5(a,b,c)
! TODO: this declaration fails in CallInterface lowering
!  real, allocatable, intent(out) :: a(:)
!  real, pointer, intent(in) :: b(:)
  real :: a(100), b(100) ! FIXME: fake it for now
  real, intent(in) :: c
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1(%
  ! CHECK: fir.do_loop
  ! CHECK: fir.array_fetch %[[B]], %
  ! CHECK: fir.array_update
  a = b + c
  ! CHECK: fir.array_merge_store %[[A]], %{{.*}} to %arg0
end subroutine test5

! CHECK-LABEL: func @_QPtest6(
! CHECK-SAME: %[[aarg:[^:]+]]: !fir.ref<!fir.array<?xf32>>,
! CHECK-SAME: %[[barg:[^:]+]]: !fir.ref<!fir.array<?xf32>>,
! CHECK-SAME: %[[carg:[^:]+]]: !fir.ref<f32>,
! CHECK-SAME: %[[narg:[^:]+]]: !fir.ref<i32>,
! CHECK-SAME: %[[marg:[^:]+]]: !fir.ref<i32>)
subroutine test6(a,b,c,n,m)
  integer :: n, m
  real, intent(out) :: a(n)
  real, intent(in) :: b(m), c
  ! CHECK: %[[ashape:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[aslice:.*]] = fir.slice %c3{{.*}}, %{{.*}}, %c4{{.*}} : (i64, i64, i64) -> !fir.slice<1>
  ! CHECK: %[[a:.*]] = fir.array_load %[[aarg]](%[[ashape]]) [%[[aslice]]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<?xf32>
  ! CHECK: %[[bshape:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[b:.*]] = fir.array_load %[[barg]](%[[bshape]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
  ! CHECK: %[[loop:.*]] = fir.do_loop {{.*}} iter_args(%{{.*}} = %[[a]]) ->
  ! CHECK: %[[bv:.*]] = fir.array_fetch %[[b]], %{{.*}} : (!fir.array<?xf32>, index) -> f32
  ! CHECK: %[[sum:.*]] = addf %[[bv]], %{{.*}} : f32
  ! CHECK: %[[res:.*]] = fir.array_update %{{.*}}, %[[sum]], %{{.*}} : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
  ! CHECK: fir.result %[[res]] : !fir.array<?xf32>
  ! CHECK: fir.array_merge_store %[[a]], %[[loop]] to %[[aarg]] : !fir.ref<!fir.array<?xf32>>
  a(3:n:4) = b + c
end subroutine test6

! CHECK-LABEL: func @_QPtest6a
! CHECK-SAME: %[[a:.*]]: !fir.ref<!fir.array<10x50xf32>>,
! CHECK-SAME: %[[b:.*]]: !fir.ref<!fir.array<10xf32>>)
subroutine test6a(a,b)
  ! copy part of 1 row to b. a's projection has rank 1.
  real :: a(10,50)
  real :: b(10)
  ! CHECK: %[[shape:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %{{.*}} = fir.array_load %[[b]](%[[shape]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK-DAG: %[[shape:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK-DAG: %[[slice:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (i64, index, index, i64, i64, i64) -> !fir.slice<2>
  ! CHECK: %{{.*}} = fir.array_load %[[a]](%[[shape]]) [%[[slice]]] : (!fir.ref<!fir.array<10x50xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.array<10x50xf32>
  ! CHECK: %{{.*}} = fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (!fir.array<10xf32>)
  ! CHECK: %[[fetch:.*]] = fir.array_fetch %{{.*}}, %{{.*}}, %[[i]] : (!fir.array<10x50xf32>, index, index) -> f32
  ! CHECK: %[[update:.*]] = fir.array_update %{{.*}}, %[[fetch]], %[[i]] : (!fir.array<10xf32>, f32, index) -> !fir.array<10xf32>
  ! CHECK: fir.result %{{.*}} : !fir.array<10xf32>
  ! CHECK: fir.array_merge_store %{{.*}}, %{{.*}} to %[[b]] : !fir.ref<!fir.array<10xf32>>
  b = a(4,41:50)
end subroutine test6a

! CHECK-LABEL: func @_QPtest6b
! CHECK-SAME: %[[a:.*]]: !fir.ref<!fir.array<10x50xf32>>,
! CHECK-SAME: %[[b:.*]]: !fir.ref<!fir.array<10xf32>>)
subroutine test6b(a,b)
  ! copy b to columns 41 to 50 of row 4 of a
  real :: a(10,50)
  real :: b(10)
  ! CHECK-DAG: %[[shape:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK-DAG: %[[slice:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (i64, index, index, i64, i64, i64) -> !fir.slice<2>
  ! CHECK: %{{.*}} = fir.array_load %[[a]](%[[shape]]) [%[[slice]]] : (!fir.ref<!fir.array<10x50xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.array<10x50xf32>
  ! CHECK: %[[shape:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %{{.*}} = fir.array_load %[[b]](%[[shape]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %{{.*}} = fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (!fir.array<10x50xf32>) {
  ! CHECK: %[[fetch:.*]] = fir.array_fetch %{{.*}}, %[[i]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK: %[[update:.*]] = fir.array_update %{{.*}}, %[[fetch]], %{{.*}}, %[[i]] : (!fir.array<10x50xf32>, f32, index, index) -> !fir.array<10x50xf32>
  ! CHECK: fir.result %{{.*}} : !fir.array<10x50xf32>
  ! CHECK: fir.array_merge_store %{{.*}}, %{{.*}} to %[[a]] : !fir.ref<!fir.array<10x50xf32>>
  a(4,41:50) = b
end subroutine test6b

! This is NOT a conflict. `a` appears on both the lhs and rhs here, but there
! are no loop-carried dependences and no copy is needed.
! CHECK-LABEL: func @_QPtest7
subroutine test7(a,b,n)
  integer :: n
  real, intent(inout) :: a(n)
  real, intent(in) :: b(n)
  ! CHECK: %[[Aout:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[Ain:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1(%
  ! CHECK: %[[T:.*]] = fir.do_loop
  ! CHECK-DAG: %[[Bi:.*]] = fir.array_fetch %[[Ain]]
  ! CHECK-DAG: %[[Ci:.*]] = fir.array_fetch %[[B]]
  ! CHECK: %[[rv:.*]] = addf %[[Bi]], %[[Ci]]
  ! CHECK: fir.array_update %{{.*}}, %[[rv]], %
  a = a + b
  ! CHECK: fir.array_merge_store %[[Aout]], %[[T]] to %arg0
end subroutine test7

! CHECK-LABEL: func @_QPtest8
subroutine test8(a,b)
  integer :: a(100), b(100)
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B1_addr:.*]] = fir.coordinate_of %arg1, %
  ! CHECK: %[[B1:.*]] = fir.load %[[B1_addr]]
  ! CHECK: %[[LOOP:.*]] = fir.do_loop
  ! CHECK: fir.array_update %{{.*}}, %[[B1]], %
  a = b(1)
  ! CHECK: fir.array_merge_store %[[A]], %[[LOOP]] to %arg0
end subroutine test8

! This FORALL construct does present a potential loop-carried dependence if
! implemented naively (and incorrectly). The final value of a(3) must be the
! value of a(2) before alistair begins execution added to b(2).
! CHECK-LABEL: func @_QPtest9
subroutine test9(a,b,n)
  integer :: n
  real, intent(inout) :: a(n)
  real, intent(in) :: b(n)
  ! CHECK: fir.do_loop
  alistair: FORALL (i=1:n-1)
     a(i+1) = a(i) + b(i)
  END FORALL alistair
end subroutine test9

! CHECK-LABEL: func @_QPtest10
subroutine test10(a,b,c,d)
  interface
     ! Function takea an array and yields an array
     function foo(a) result(res)
       real :: a(:)  ! FIXME: must be before res or semantics fails
                     ! as `size(a,1)` fails to resolve to the argument
       real, dimension(size(a,1)) :: res
     end function foo
  end interface
  interface
     ! Function takes an array and yields a scalar
     real function bar(a)
       real :: a(:)
     end function bar
  end interface
  real :: a(:), b(:), c(:), d(:)
!  a = b + foo(c + foo(d + bar(a)))
end subroutine test10

! CHECK-LABEL: func @_QPtest11
subroutine test11(a,b,c,d)
  real, external :: bar
  real :: a(100), b(100), c(100), d(100)
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1
  ! CHECK-DAG: %[[C:.*]] = fir.array_load %arg2
  ! CHECK-DAG: %[[D:.*]] = fir.array_load %arg3
  ! CHECK-DAG: %[[tmp:.*]] = fir.allocmem
  ! CHECK-DAG: %[[T:.*]] = fir.array_load %[[tmp]]
  
  !    temporary <- c + d
  ! CHECK: %[[bar_in:.*]] = fir.do_loop
  !  CHECK-DAG: %[[c_i:.*]] = fir.array_fetch %[[C]]
  !  CHECK-DAG: %[[d_i:.*]] = fir.array_fetch %[[D]]
  !  CHECK: %[[sum:.*]] = addf %[[c_i]], %[[d_i]]
  !  CHECK: fir.array_update %{{.*}}, %[[sum]], %
  ! CHECK: fir.array_merge_store %[[T]], %[[bar_in]] to %[[tmp]]
  ! CHECK: %[[cast:.*]] = fir.convert %[[tmp]]
  ! CHECK: %[[bar_out:.*]] = fir.call @_QPbar(%[[cast]]
  
  !    a <- b + bar(?)
  ! CHECK: %[[S:.*]] = fir.do_loop
  !  CHECK: %[[b_i:.*]] = fir.array_fetch %[[B]], %
  !  CHECK: %[[sum2:.*]] = addf %[[b_i]], %[[bar_out]]
  !  CHECK: fir.array_update %{{.*}}, %[[sum2]], %
  ! CHECK: fir.array_merge_store %[[A]], %[[S]] to %arg0
  a = b + bar(c + d)
end subroutine test11

! CHECK-LABEL: func @_QPtest12
subroutine test12(a,b,c,d,n,m)
  integer :: n, m
  ! CHECK: %[[n:.*]] = fir.load %arg4
  ! CHECK: %[[m:.*]] = fir.load %arg5
  ! CHECK: %[[sha:.*]] = fir.shape %
  ! CHECK: %[[A:.*]] = fir.array_load %arg0(%[[sha]])
  ! CHECK: %[[shb:.*]] = fir.shape %
  ! CHECK: %[[B:.*]] = fir.array_load %arg1(%[[shb]])
  ! CHECK: %[[tmp:.*]] = fir.allocmem !fir.array<?xf32>, %{{.*}} {name = ".array.expr"}
  ! CHECK: %[[T:.*]] = fir.array_load %[[tmp]](%
  ! CHECK: %[[C:.*]] = fir.array_load %arg2(%
  ! CHECK: %[[D:.*]] = fir.array_load %arg3(%
  real, external :: bar
  real :: a(n), b(n), c(m), d(m)
  ! CHECK: %[[LOOP:.*]] = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[T]])
    ! CHECK-DAG: fir.array_fetch %[[C]]
    ! CHECK-DAG: fir.array_fetch %[[D]]
  ! CHECK: fir.array_merge_store %[[T]], %[[LOOP]]
  ! CHECK: %[[CALL:.*]] = fir.call @_QPbar
  ! CHECK: %[[LOOP2:.*]] = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[A]])
    ! CHECK: fir.array_fetch %[[B]]
  ! CHECK: fir.array_merge_store %[[A]], %[[LOOP2]] to %arg0
  a = b + bar(c + d)
  ! CHECK: fir.freemem %[[tmp]] : !fir.heap<!fir.array<?xf32>>
end subroutine test12

! CHECK-LABEL: func @_QPtest13
subroutine test13(a,b,c,d,n,m,i)
  real :: a(n), b(m)
  complex :: c(n), d(m)
  ! CHECK: %[[A_shape:.*]] = fir.shape %
  ! CHECK: %[[A:.*]] = fir.array_load %arg0(%[[A_shape]])
  ! CHECK: %[[B_shape:.*]] = fir.shape %
  ! CHECK: %[[B_slice:.*]] = fir.slice %
  ! CHECK: %[[B:.*]] = fir.array_load %arg1(%[[B_shape]]) [%[[B_slice]]]
  ! CHECK: %[[C_shape:.*]] = fir.shape %
  ! CHECK: %[[C_slice:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}} path %
  ! CHECK: %[[C:.*]] = fir.array_load %arg2(%[[C_shape]]) [%[[C_slice]]]
  ! CHECK: %[[D_shape:.*]] = fir.shape %
  ! CHECK: %[[D_slice:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}} path %
  ! CHECK: %[[D:.*]] = fir.array_load %arg3(%[[D_shape]]) [%[[D_slice]]]
  ! CHECK: = constant -6.2598534E+18 : f32
  ! CHECK: %[[A_result:.*]] = fir.do_loop %{{.*}} = %{{.*}} iter_args(%[[A_in:.*]] = %[[A]]) ->
  ! CHECK: fir.array_fetch %[[B]],
  ! CHECK: fir.array_fetch %[[C]],
  ! CHECK: fir.array_fetch %[[D]],
  ! CHECK: fir.array_update %[[A_in]],
  a = b(i:i+2*n-2:2) + c%im - d(i:i+2*n-2:2)%re + x'deadbeef'
  ! CHECK: fir.array_merge_store %[[A]], %[[A_result]] to %arg0
end subroutine test13

! Test elemental call to function f
! CHECK-LABEL: func @_QPtest14(
! CHECK-SAME: %[[a:.*]]: !fir.ref<!fir.array<100xf32>>,
! CHECK-SAME: %[[b:.*]]: !fir.ref<!fir.array<100xf32>>)
subroutine test14(a,b)
  ! CHECK: %[[barr:.*]] = fir.array_load %[[b]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  interface
     real elemental function f1(i)
       real, intent(in) :: i
     end function f1
  end interface
  real :: a(100), b(100)
  ! CHECK: %[[loop:.*]] = fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[bth:.*]] = %[[barr]]) -> (!fir.array<100xf32>) {
  ! CHECK: %[[tmp:.*]] = fir.array_coor %[[a]](%{{.*}}) %[[i]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: %[[fres:.*]] = fir.call @_QPf1(%[[tmp]]) : (!fir.ref<f32>) -> f32
  ! CHECK: %[[res:.*]] = fir.array_update %[[bth]], %[[fres]], %[[i]] : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
  ! CHECK: fir.result %[[res]] : !fir.array<100xf32>
  ! CHECK: fir.array_merge_store %[[barr]], %[[loop]] to %[[b]]
  b = f1(a)
end subroutine test14

! Test elemental intrinsic function (abs)
! CHECK-LABEL: func @_QPtest15(
! CHECK-SAME: %[[a:.*]]: !fir.ref<!fir.array<100xf32>>,
! CHECK-SAME: %[[b:.*]]: !fir.ref<!fir.array<100xf32>>)
subroutine test15(a,b)
  ! CHECK-DAG: %[[barr:.*]] = fir.array_load %[[b]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  ! CHECK-DAG: %[[aarr:.*]] = fir.array_load %[[a]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  real :: a(100), b(100)
  ! CHECK: %[[loop:.*]] = fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[bth:.*]] = %[[barr]]) -> (!fir.array<100xf32>) {
  ! CHECK: %[[val:.*]] = fir.array_fetch %[[aarr]], %[[i]] : (!fir.array<100xf32>, index) -> f32
  ! CHECK: %[[fres:.*]] = fir.call @llvm.fabs.f32(%[[val]]) : (f32) -> f32
  ! CHECK: %[[res:.*]] = fir.array_update %[[bth]], %[[fres]], %[[i]] : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
  ! CHECK: fir.result %[[res]] : !fir.array<100xf32>
  ! CHECK: fir.array_merge_store %[[barr]], %[[loop]] to %[[b]]
  b = abs(a)
end subroutine test15

! Test elemental call to function f2 with VALUE attribute
! CHECK-LABEL: func @_QPtest16(
! CHECK-SAME: %[[a:.*]]: !fir.ref<!fir.array<100xf32>>,
! CHECK-SAME: %[[b:.*]]: !fir.ref<!fir.array<100xf32>>)
subroutine test16(a,b)
  ! CHECK: %[[tmp:.*]] = fir.alloca f32 {adapt.valuebyref}
  ! CHECK-DAG: %[[aarr:.*]] = fir.array_load %[[a]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  ! CHECK-DAG: %[[barr:.*]] = fir.array_load %[[b]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  interface
     real elemental function f2(i)
       real, VALUE :: i
     end function f2
  end interface
  real :: a(100), b(100)
  ! CHECK: %[[loop:.*]] = fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[bth:.*]] = %[[barr]]) -> (!fir.array<100xf32>) {
  ! CHECK: %[[val:.*]] = fir.array_fetch %[[aarr]], %[[i]] : (!fir.array<100xf32>, index) -> f32
  ! CHECK: fir.store %[[val]] to %[[tmp]]
  ! CHECK: %[[fres:.*]] = fir.call @_QPf2(%[[tmp]]) : (!fir.ref<f32>) -> f32
  ! CHECK: %[[res:.*]] = fir.array_update %[[bth]], %[[fres]], %[[i]] : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
  ! CHECK: fir.result %[[res]] : !fir.array<100xf32>
  ! CHECK: fir.array_merge_store %[[barr]], %[[loop]] to %[[b]]
  b = f2(a)
end subroutine test16

! Test elemental impure call to function f3.
!
! CHECK-LABEL: func @_QPtest17(
! CHECK-SAME: %[[a:[^:]+]]: !fir.ref<!fir.array<100xf32>>,
! CHECK-SAME: %[[b:[^:]+]]: !fir.ref<!fir.array<100xf32>>,
! CHECK-SAME: %[[c:.*]]: !fir.ref<!fir.array<100xf32>>)
subroutine test17(a,b,c)
  ! CHECK-DAG: %[[aarr:.*]] = fir.array_load %[[a]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  ! CHECK-DAG: %[[barr:.*]] = fir.array_load %[[b]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  interface
     real elemental impure function f3(i,j,k)
       real, intent(inout) :: i, j, k
     end function f3
  end interface
  real :: a(100), b(100), c(100)
  ! CHECK: %[[loop:.*]] = fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[bth:.*]] = %[[barr]]) -> (!fir.array<100xf32>) {
  ! CHECK-DAG: %[[val:.*]] = fir.array_fetch %[[aarr]], %[[i]] : (!fir.array<100xf32>, index) -> f32
  ! CHECK-DAG: %[[ccoor:.*]] = fir.array_coor %[[c]](%{{.*}}) %[[i]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK-DAG: %[[bcoor:.*]] = fir.array_coor %[[b]](%{{.*}}) %[[i]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK-DAG: %[[acoor:.*]] = fir.array_coor %[[a]](%{{.*}}) %[[i]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: %[[fres:.*]] = fir.call @_QPf3(%[[ccoor]], %[[bcoor]], %[[acoor]]) : (!fir.ref<f32>, !fir.ref<f32>, !fir.ref<f32>) -> f32
  ! CHECK: %[[fadd:.*]] = addf %[[val]], %[[fres]] : f32
  ! CHECK: %[[res:.*]] = fir.array_update %[[bth]], %[[fadd]], %[[i]] : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>

  ! See 10.1.4.p2 note 1. The expression below is illegal if `f3` defines the
  ! argument `a` for this statement. Since, this cannot be proven statically by
  ! the compiler, the constraint is left to the user. The compiler may give a
  ! warning that `k` is neither VALUE nor INTENT(IN) and the actual argument,
  ! `a`, appears elsewhere in the same statement.
  b = a + f3(c, b, a)

  ! CHECK: fir.result %[[res]] : !fir.array<100xf32>
  ! CHECK: fir.array_merge_store %[[barr]], %[[loop]] to %[[b]]
end subroutine test17

! CHECK-LABEL: func @_QPtest18(
subroutine test18
  integer, target :: array(10,10)
  integer, pointer :: row_i(:)
  ! CHECK: %[[iaddr:.*]] = fir.alloca i32 {name = "_QFtest18Ei"}
  ! CHECK: %[[i:.*]] = fir.load %[[iaddr]] : !fir.ref<i32>
  ! CHECK: %[[icast:.*]] = fir.convert %[[i]] : (i32) -> i64
  ! CHECK: %[[exact:.*]] = fir.undefined index
  ! CHECK: %[[ubound:.*]] = subi %{{.*}}, %c1 : index
  ! CHECK: %[[slice:.*]] = fir.slice %[[icast]], %[[exact]], %[[exact]], %c1, %[[ubound]], %c1{{.*}} : (i64, index, index, index, index, i64) -> !fir.slice<2>
  ! CHECK: = fir.embox %{{.*}}(%{{.*}}) [%[[slice]]] : (!fir.ref<!fir.array<10x10xi32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?xi32>>
  row_i => array(i, :)
end subroutine test18

! CHECK-LABEL: func @_QPtest_column_and_row_order(
subroutine test_column_and_row_order(x)
  real :: x(2,3)
  ! CHECK-DAG: %[[c2:.*]] = fir.convert %c2{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[number_of_rows:.*]] = subi %[[c2]], %c1{{.*}} : index
  ! CHECK-DAG: %[[c3:.*]] = fir.convert %c3{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[number_of_columns:.*]] = subi %[[c3]], %c1{{.*}} : index
  ! CHECK: fir.do_loop %[[column:.*]] = %c0{{.*}} to %[[number_of_columns]]
  ! CHECK: fir.do_loop %[[row:.*]] = %c0{{.*}} to %[[number_of_rows]]
  ! CHECK: = fir.array_update %{{.*}}, %{{.*}}, %[[row]], %[[column]] : (!fir.array<2x3xf32>, f32, index, index) -> !fir.array<2x3xf32>
  x = 42
end subroutine


! CHECK: func private @_QPbar(
