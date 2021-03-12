! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest1(
! CHECK-SAME: %[[a:.*]]: !fir.ref<!fir.array<100xf32>>,
! CHECK-SAME: %[[i:[^:]+]]: !fir.ref<i32>,
! CHECK-SAME: %[[j:[^:]+]]: !fir.ref<i32>,
! CHECK-SAME: %[[k:[^:]+]]: !fir.ref<i32>)
subroutine test1(a,i,j,k)
  ! CHECK-DAG: %[[c:.*]] = constant 100 : index
  ! CHECK-DAG: %[[ii:.*]] = fir.load %[[i]] : !fir.ref<i32>
  ! CHECK-DAG: %[[iv:.*]] = fir.convert %[[ii]] : (i32) -> i64
  ! CHECK-DAG: %[[jj:.*]] = fir.load %[[j]] : !fir.ref<i32>
  ! CHECK-DAG: %[[jv:.*]] = fir.convert %[[jj]] : (i32) -> i64
  ! CHECK-DAG: %[[kk:.*]] = fir.load %[[k]] : !fir.ref<i32>
  ! CHECK-DAG: %[[kv:.*]] = fir.convert %[[kk]] : (i32) -> i64

  real a(100)
  integer i, j, k
  interface
    subroutine bob(a)
      real :: a(:)
    end subroutine bob
  end interface

  ! CHECK: %[[shape:.*]] = fir.shape %[[c]] : (index) -> !fir.shape<1>
  ! CHECK: %[[slice:.*]] = fir.slice %[[iv]], %[[jv]], %[[kv]] : (i64, i64, i64) -> !fir.slice<1>
  ! CHECK: %[[box:.*]] = fir.embox %[[a]](%[[shape]]) [%[[slice]]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: fir.call @_QPbob(%[[box]]) : (!fir.box<!fir.array<?xf32>>) -> ()

  associate (name => a(i:j:k))
    call bob(name)
  end associate
end subroutine test1

! CHECK-LABEL: func @_QPtest2(
! CHECK-SAME: %[[nadd:.*]]: !fir.ref<i32>)
subroutine test2(n)
  integer :: n
  integer, external :: foo
  ! CHECK: %[[n:.*]] = fir.load %[[nadd]] : !fir.ref<i32>
  ! CHECK: %[[n10:.*]] = addi %[[n]], %c10{{.*}} : i32
  ! CHECK: fir.store %[[n10]] to %{{.*}} : !fir.ref<i32>
  ! CHECK: %[[foo:.*]] = fir.call @_QPfoo(%{{.*}}) : (!fir.ref<i32>) -> i32
  ! CHECK: fir.store %[[foo]] to %{{.*}} : !fir.ref<i32>
  associate (i => n, j => n + 10, k => foo(20))
    print *, i, j, k, n
  end associate
end subroutine test2
