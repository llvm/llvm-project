! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LINE: func @_QPtest1
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
  ! CHECK: %[[rv:.*]] = fir.addf %[[Bi]], %[[Ci]]
  ! CHECK: fir.array_update %{{.*}}, %[[rv]], %
  a = b + c
  ! CHECK: fir.array_merge_store %[[A]], %[[T]] to %arg0
end subroutine test1

! CHECK-LINE: func @_QPtest1b
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
  ! CHECK: %[[rv1:.*]] = fir.addf %[[Bi]], %[[Ci]]
  ! CHECK: %[[Di:.*]] = fir.array_fetch %[[D]]
  ! CHECK: %[[rv:.*]] = fir.addf %[[rv1]], %[[Di]]
  ! CHECK: fir.array_update %{{.*}}, %[[rv]], %
  a = b + c + d
  ! CHECK: fir.array_merge_store %[[A]], %[[T]] to %arg0
end subroutine test1b

! CHECK-LINE: func @_QPtest2
subroutine test2(a,b,c)
  real, intent(out) :: a(:)
  real, intent(in) :: b(:), c(:)
!  a = b + c
end subroutine test2

! CHECK-LINE: func @_QPtest3
subroutine test3(a,b,c,n)
  integer :: n
  real, intent(out) :: a(n)
  real, intent(in) :: b(n), c
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1(%
  ! CHECK-DAG: %[[C:.*]] = fir.load %arg2
  ! CHECK: %[[T:.*]] = fir.do_loop
  ! CHECK: %[[Bi:.*]] = fir.array_fetch %[[B]]
  ! CHECK: %[[rv:.*]] = fir.addf %[[Bi]], %[[C]]
  ! CHECK: %[[Ti:.*]] = fir.array_update %{{.*}}, %[[rv]], %
  ! CHECK: fir.result %[[Ti]]
  a = b + c
  ! CHECK: fir.array_merge_store %[[A]], %[[T]] to %arg0
end subroutine test3

! CHECK-LINE: func @_QPtest4
subroutine test4(a,b,c)
! TODO: this declaration fails in CallInterface lowering
!  real, allocatable, intent(out) :: a(:)
  real :: a(100) ! FIXME: fake it for now
  real, intent(in) :: b(:), c
  ! CHECK: %[[Ba:.*]] = fir.box_addr %arg1
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %[[Ba]](%
  ! CHECK: fir.do_loop
  ! CHECK: fir.array_fetch %[[B]], %
  ! CHECK: fir.array_update
  a = b + c
  ! CHECK: fir.array_merge_store %[[A]], %{{.*}} to %arg0
end subroutine test4

! CHECK-LINE: func @_QPtest5
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

! CHECK-LINE: func @_QPtest6
subroutine test6(a,b,c,n,m)
  integer :: n, m
  real, intent(out) :: a(n)
  real, intent(in) :: b(m), c
!  a(3:n:4) = b + c
end subroutine test6

! This is NOT a conflict. `a` appears on both the lhs and rhs here, but there
! are no loop-carried dependences and no copy is needed.
! CHECK-LINE: func @_QPtest7
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
  ! CHECK: %[[rv:.*]] = fir.addf %[[Bi]], %[[Ci]]
  ! CHECK: fir.array_update %{{.*}}, %[[rv]], %
  a = a + b
  ! CHECK: fir.array_merge_store %[[Aout]], %[[T]] to %arg0
end subroutine test7

! CHECK-LINE: func @_QPtest8
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
! CHECK-LINE: func @_QPtest9
subroutine test9(a,b,n)
  integer :: n
  real, intent(inout) :: a(n)
  real, intent(in) :: b(n)
  ! CHECK: fir.do_loop
  alistair: FORALL (i=1:n-1)
     a(i+1) = a(i) + b(i)
  END FORALL alistair
end subroutine test9

! CHECK-LINE: func @_QPtest10
subroutine test10(a,b,c,d)
  interface
     function foo(a) result(res)
       real :: a(:)  ! FIXME: must be before res or semantics fails
                     ! as `size(a,1)` fails to resolve to the argument
       real, dimension(size(a,1)) :: res
     end function foo
  end interface
  interface
     real function bar(a)
       real :: a(:)
     end function bar
  end interface
  real :: a(:), b(:), c(:), d(:)
!  a = b + foo(c + foo(d + bar(a)))
end subroutine test10

! CHECK-LINE: func @_QPtest11
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
  !  CHECK: %[[sum:.*]] = fir.addf %[[c_i]], %[[d_i]]
  !  CHECK: fir.array_update %{{.*}}, %[[sum]], %
  ! CHECK: fir.array_merge_store %[[T]], %[[bar_in]] to %[[tmp]]
  ! CHECK: %[[cast:.*]] = fir.convert %[[tmp]]
  ! CHECK: %[[bar_out:.*]] = fir.call @_QPbar(%[[cast]]
  
  !    a <- b + bar(?)
  ! CHECK: %[[S:.*]] = fir.do_loop
  !  CHECK: %[[b_i:.*]] = fir.array_fetch %[[B]], %
  !  CHECK: %[[sum2:.*]] = fir.addf %[[b_i]], %[[bar_out]]
  !  CHECK: fir.array_update %{{.*}}, %[[sum2]], %
  ! CHECK: fir.array_merge_store %[[A]], %[[S]] to %arg0
  a = b + bar(c + d)
end subroutine test11

! CHECK-LINE: func @_QPtest12
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

! CHECK: func private @_QPbar(
