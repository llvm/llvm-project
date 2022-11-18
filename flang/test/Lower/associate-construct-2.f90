! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest1(
! CHECK-SAME:     %[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_2:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_3:.*]]: !fir.ref<i32>{{.*}}) {
! CHECK:         %[[VAL_4:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_5:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> i64
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> i64
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:         %[[VAL_11:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> i64
! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i64) -> index
! CHECK:         %[[VAL_14:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_15:.*]] = fir.slice %[[VAL_7]], %[[VAL_13]], %[[VAL_10]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_16:.*]] = fir.embox %[[VAL_0]](%[[VAL_14]]) {{\[}}%[[VAL_15]]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:         fir.call @_QPbob(%[[VAL_16]]) {{.*}}: (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:         return
! CHECK:       }

subroutine test1(a,i,j,k)

  real a(100)
  integer i, j, k
  interface
    subroutine bob(a)
      real :: a(:)
    end subroutine bob
  end interface

  associate (name => a(i:j:k))
    call bob(name)
  end associate
end subroutine test1

! CHECK-LABEL: func @_QPtest2(
! CHECK-SAME: %[[nadd:.*]]: !fir.ref<i32>{{.*}})
subroutine test2(n)
  integer :: n
  integer, external :: foo
  ! CHECK: %[[n:.*]] = fir.load %[[nadd]] : !fir.ref<i32>
  ! CHECK: %[[n10:.*]] = arith.addi %[[n]], %c10{{.*}} : i32
  ! CHECK: fir.store %[[n10]] to %{{.*}} : !fir.ref<i32>
  ! CHECK: %[[foo:.*]] = fir.call @_QPfoo(%{{.*}}) {{.*}}: (!fir.ref<i32>) -> i32
  ! CHECK: fir.store %[[foo]] to %{{.*}} : !fir.ref<i32>
  associate (i => n, j => n + 10, k => foo(20))
    print *, i, j, k, n
  end associate
end subroutine test2
