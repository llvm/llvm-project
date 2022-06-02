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
  ! CHECK: %[[rv:.*]] = arith.addf %[[Bi]], %[[Ci]]
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
  ! CHECK: %[[rv1:.*]] = arith.addf %[[Bi]], %[[Ci]]
  ! CHECK: %[[Di:.*]] = fir.array_fetch %[[D]]
  ! CHECK: %[[rv:.*]] = arith.addf %[[rv1]], %[[Di]]
  ! CHECK: fir.array_update %{{.*}}, %[[rv]], %
  a = b + c + d
  ! CHECK: fir.array_merge_store %[[A]], %[[T]] to %arg0
end subroutine test1b

! CHECK-LABEL: func @_QPtest2(
! CHECK-SAME:     %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}, %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}, %[[VAL_2:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
! CHECK:         %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_5:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_6:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_7:.*]] = fir.array_load %[[VAL_2]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_10:.*]] = arith.subi %[[VAL_4]]#1, %[[VAL_8]] : index
! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_9]] to %[[VAL_10]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_5]]) -> (!fir.array<?xf32>) {
! CHECK:           %[[VAL_14:.*]] = fir.array_fetch %[[VAL_6]], %[[VAL_12]] : (!fir.array<?xf32>, index) -> f32
! CHECK:           %[[VAL_15:.*]] = fir.array_fetch %[[VAL_7]], %[[VAL_12]] : (!fir.array<?xf32>, index) -> f32
! CHECK:           %[[VAL_16:.*]] = arith.addf %[[VAL_14]], %[[VAL_15]] : f32
! CHECK:           %[[VAL_17:.*]] = fir.array_update %[[VAL_13]], %[[VAL_16]], %[[VAL_12]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
! CHECK:           fir.result %[[VAL_17]] : !fir.array<?xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_5]], %[[VAL_18:.*]] to %[[VAL_0]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.box<!fir.array<?xf32>>
! CHECK:         return
! CHECK:       }
subroutine test2(a,b,c)
  real, intent(out) :: a(:)
  real, intent(in) :: b(:), c(:)
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
  ! CHECK: %[[rv:.*]] = arith.addf %[[Bi]], %[[C]]
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
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<?xf32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<?xf32>>{{.*}}, %[[VAL_2:.*]]: !fir.ref<f32>{{.*}}, %[[VAL_3:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_4:.*]]: !fir.ref<i32>{{.*}}) {
! CHECK:         %[[VAL_5:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> i64
! CHECK:         %[[VAL_7A:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
! CHECK:         %[[C0:.*]] = arith.constant 0 : index 
! CHECK:         %[[CMP:.*]] = arith.cmpi sgt, %[[VAL_7A]], %[[C0]] : index 
! CHECK:         %[[VAL_7:.*]] = arith.select %[[CMP]], %[[VAL_7A]], %[[C0]] : index 
! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> i64
! CHECK:         %[[VAL_10A:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:         %[[C0_2:.*]] = arith.constant 0 : index 
! CHECK:         %[[CMP_2:.*]] = arith.cmpi sgt, %[[VAL_10A]], %[[C0_2]] : index 
! CHECK:         %[[VAL_10:.*]] = arith.select %[[CMP_2]], %[[VAL_10A]], %[[C0_2]] : index 
! CHECK:         %[[VAL_11:.*]] = arith.constant 3 : i64
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:         %[[VAL_13:.*]] = arith.constant 4 : i64
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:         %[[VAL_15:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
! CHECK:         %[[VAL_18:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_19:.*]] = arith.subi %[[VAL_17]], %[[VAL_12]] : index
! CHECK:         %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_14]] : index
! CHECK:         %[[VAL_21:.*]] = arith.divsi %[[VAL_20]], %[[VAL_14]] : index
! CHECK:         %[[VAL_22:.*]] = arith.cmpi sgt, %[[VAL_21]], %[[VAL_18]] : index
! CHECK:         %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_21]], %[[VAL_18]] : index
! CHECK:         %[[VAL_24:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_25:.*]] = fir.slice %[[VAL_12]], %[[VAL_17]], %[[VAL_14]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_26:.*]] = fir.array_load %[[VAL_0]](%[[VAL_24]]) {{\[}}%[[VAL_25]]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_27:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_28:.*]] = fir.array_load %[[VAL_1]](%[[VAL_27]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_29:.*]] = fir.load %[[VAL_2]] : !fir.ref<f32>
! CHECK:         %[[VAL_30:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_31:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_32:.*]] = arith.subi %[[VAL_23]], %[[VAL_30]] : index
! CHECK:         %[[VAL_33:.*]] = fir.do_loop %[[VAL_34:.*]] = %[[VAL_31]] to %[[VAL_32]] step %[[VAL_30]] unordered iter_args(%[[VAL_35:.*]] = %[[VAL_26]]) -> (!fir.array<?xf32>) {
! CHECK:           %[[VAL_36:.*]] = fir.array_fetch %[[VAL_28]], %[[VAL_34]] : (!fir.array<?xf32>, index) -> f32
! CHECK:           %[[VAL_37:.*]] = arith.addf %[[VAL_36]], %[[VAL_29]] : f32
! CHECK:           %[[VAL_38:.*]] = fir.array_update %[[VAL_35]], %[[VAL_37]], %[[VAL_34]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
! CHECK:           fir.result %[[VAL_38]] : !fir.array<?xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_26]], %[[VAL_39:.*]] to %[[VAL_0]]{{\[}}%[[VAL_25]]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.slice<1>
! CHECK:         return
! CHECK:       }

subroutine test6(a,b,c,n,m)
  integer :: n, m
  real, intent(out) :: a(n)
  real, intent(in) :: b(m), c
  a(3:n:4) = b + c
end subroutine test6

! CHECK-LABEL: func @_QPtest6a(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10x50xf32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xf32>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_3:.*]] = arith.constant 50 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_6:.*]] = fir.array_load %[[VAL_1]](%[[VAL_5]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_8:.*]] = arith.constant 4 : i64
! CHECK:         %[[VAL_9:.*]] = fir.undefined index
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:         %[[VAL_11:.*]] = arith.subi %[[VAL_10]], %[[VAL_7]] : index
! CHECK:         %[[VAL_12:.*]] = arith.constant 41 : i64
! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i64) -> index
! CHECK:         %[[VAL_14:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i64) -> index
! CHECK:         %[[VAL_16:.*]] = arith.constant 50 : i64
! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
! CHECK:         %[[VAL_18:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_19:.*]] = fir.slice %[[VAL_8]], %[[VAL_9]], %[[VAL_9]], %[[VAL_13]], %[[VAL_17]], %[[VAL_15]] : (i64, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         %[[VAL_20:.*]] = fir.array_load %[[VAL_0]](%[[VAL_18]]) {{\[}}%[[VAL_19]]] : (!fir.ref<!fir.array<10x50xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.array<10x50xf32>
! CHECK:         %[[VAL_21:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_22:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_23:.*]] = arith.subi %[[VAL_4]], %[[VAL_21]] : index
! CHECK:         %[[VAL_24:.*]] = fir.do_loop %[[VAL_25:.*]] = %[[VAL_22]] to %[[VAL_23]] step %[[VAL_21]] unordered iter_args(%[[VAL_26:.*]] = %[[VAL_6]]) -> (!fir.array<10xf32>) {
! CHECK:           %[[VAL_27:.*]] = fir.array_fetch %[[VAL_20]], %[[VAL_11]], %[[VAL_25]] : (!fir.array<10x50xf32>, index, index) -> f32
! CHECK:           %[[VAL_28:.*]] = fir.array_update %[[VAL_26]], %[[VAL_27]], %[[VAL_25]] : (!fir.array<10xf32>, f32, index) -> !fir.array<10xf32>
! CHECK:           fir.result %[[VAL_28]] : !fir.array<10xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_6]], %[[VAL_29:.*]] to %[[VAL_1]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
! CHECK:         return
! CHECK:       }

subroutine test6a(a,b)
  ! copy part of 1 row to b. a's projection has rank 1.
  real :: a(10,50)
  real :: b(10)
  b = a(4,41:50)
end subroutine test6a

! CHECK-LABEL: func @_QPtest6b(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10x50xf32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xf32>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_3:.*]] = arith.constant 50 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_6:.*]] = arith.constant 4 : i64
! CHECK:         %[[VAL_7:.*]] = fir.undefined index
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
! CHECK:         %[[VAL_9:.*]] = arith.subi %[[VAL_8]], %[[VAL_5]] : index
! CHECK:         %[[VAL_10:.*]] = arith.constant 41 : i64
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:         %[[VAL_12:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i64) -> index
! CHECK:         %[[VAL_14:.*]] = arith.constant 50 : i64
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i64) -> index
! CHECK:         %[[VAL_16:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_17:.*]] = arith.subi %[[VAL_15]], %[[VAL_11]] : index
! CHECK:         %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_13]] : index
! CHECK:         %[[VAL_19:.*]] = arith.divsi %[[VAL_18]], %[[VAL_13]] : index
! CHECK:         %[[VAL_20:.*]] = arith.cmpi sgt, %[[VAL_19]], %[[VAL_16]] : index
! CHECK:         %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_19]], %[[VAL_16]] : index
! CHECK:         %[[VAL_22:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_23:.*]] = fir.slice %[[VAL_6]], %[[VAL_7]], %[[VAL_7]], %[[VAL_11]], %[[VAL_15]], %[[VAL_13]] : (i64, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         %[[VAL_24:.*]] = fir.array_load %[[VAL_0]](%[[VAL_22]]) {{\[}}%[[VAL_23]]] : (!fir.ref<!fir.array<10x50xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.array<10x50xf32>
! CHECK:         %[[VAL_25:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_26:.*]] = fir.array_load %[[VAL_1]](%[[VAL_25]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
! CHECK:         %[[VAL_27:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_28:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_29:.*]] = arith.subi %[[VAL_21]], %[[VAL_27]] : index
! CHECK:         %[[VAL_30:.*]] = fir.do_loop %[[VAL_31:.*]] = %[[VAL_28]] to %[[VAL_29]] step %[[VAL_27]] unordered iter_args(%[[VAL_32:.*]] = %[[VAL_24]]) -> (!fir.array<10x50xf32>) {
! CHECK:           %[[VAL_33:.*]] = fir.array_fetch %[[VAL_26]], %[[VAL_31]] : (!fir.array<10xf32>, index) -> f32
! CHECK:           %[[VAL_34:.*]] = fir.array_update %[[VAL_32]], %[[VAL_33]], %[[VAL_9]], %[[VAL_31]] : (!fir.array<10x50xf32>, f32, index, index) -> !fir.array<10x50xf32>
! CHECK:           fir.result %[[VAL_34]] : !fir.array<10x50xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_24]], %[[VAL_35:.*]] to %[[VAL_0]]{{\[}}%[[VAL_23]]] : !fir.array<10x50xf32>, !fir.array<10x50xf32>, !fir.ref<!fir.array<10x50xf32>>, !fir.slice<2>
! CHECK:         return
! CHECK:       }

subroutine test6b(a,b)
  ! copy b to columns 41 to 50 of row 4 of a
  real :: a(10,50)
  real :: b(10)
  a(4,41:50) = b
end subroutine test6b

! CHECK-LABEL: func @_QPtest7(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<?xf32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<?xf32>>{{.*}}, %[[VAL_2:.*]]: !fir.ref<i32>{{.*}}) {
! CHECK:         %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i32) -> i64
! CHECK:         %[[VAL_5A:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:         %[[C0:.*]] = arith.constant 0 : index 
! CHECK:         %[[CMP:.*]] = arith.cmpi sgt, %[[VAL_5A]], %[[C0]] : index 
! CHECK:         %[[VAL_5:.*]] = arith.select %[[CMP]], %[[VAL_5A]], %[[C0]] : index 
! CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> i64
! CHECK:         %[[VAL_8A:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:         %[[C0_2:.*]] = arith.constant 0 : index 
! CHECK:         %[[CMP_2:.*]] = arith.cmpi sgt, %[[VAL_8A]], %[[C0_2]] : index 
! CHECK:         %[[VAL_8:.*]] = arith.select %[[CMP_2]], %[[VAL_8A]], %[[C0_2]] : index 
! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_0]](%[[VAL_9]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]](%[[VAL_11]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_13:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_14:.*]] = fir.array_load %[[VAL_1]](%[[VAL_13]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
! CHECK:         %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_16:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_17:.*]] = arith.subi %[[VAL_5]], %[[VAL_15]] : index
! CHECK:         %[[VAL_18:.*]] = fir.do_loop %[[VAL_19:.*]] = %[[VAL_16]] to %[[VAL_17]] step %[[VAL_15]] unordered iter_args(%[[VAL_20:.*]] = %[[VAL_10]]) -> (!fir.array<?xf32>) {
! CHECK:           %[[VAL_21:.*]] = fir.array_fetch %[[VAL_12]], %[[VAL_19]] : (!fir.array<?xf32>, index) -> f32
! CHECK:           %[[VAL_22:.*]] = fir.array_fetch %[[VAL_14]], %[[VAL_19]] : (!fir.array<?xf32>, index) -> f32
! CHECK:           %[[VAL_23:.*]] = arith.addf %[[VAL_21]], %[[VAL_22]] : f32
! CHECK:           %[[VAL_24:.*]] = fir.array_update %[[VAL_20]], %[[VAL_23]], %[[VAL_19]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
! CHECK:           fir.result %[[VAL_24]] : !fir.array<?xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_10]], %[[VAL_25:.*]] to %[[VAL_0]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>
! CHECK:         return
! CHECK:       }

! This is NOT a conflict. `a` appears on both the lhs and rhs here, but there
! are no loop-carried dependences and no copy is needed.
subroutine test7(a,b,n)
  integer :: n
  real, intent(inout) :: a(n)
  real, intent(in) :: b(n)
  a = a + b
end subroutine test7

! CHECK-LABEL: func @_QPtest8(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<100xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<100xi32>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_4:.*]] = fir.array_load %[[VAL_0]](%[[VAL_3]]) : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.array<100xi32>
! CHECK:         %[[VAL_5:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_7:.*]] = arith.subi %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:         %[[VAL_8:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_7]] : (!fir.ref<!fir.array<100xi32>>, i64) -> !fir.ref<i32>
! CHECK:         %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
! CHECK:         %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_11:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_12:.*]] = arith.subi %[[VAL_2]], %[[VAL_10]] : index
! CHECK:         %[[VAL_13:.*]] = fir.do_loop %[[VAL_14:.*]] = %[[VAL_11]] to %[[VAL_12]] step %[[VAL_10]] unordered iter_args(%[[VAL_15:.*]] = %[[VAL_4]]) -> (!fir.array<100xi32>) {
! CHECK:           %[[VAL_16:.*]] = fir.array_update %[[VAL_15]], %[[VAL_9]], %[[VAL_14]] : (!fir.array<100xi32>, i32, index) -> !fir.array<100xi32>
! CHECK:           fir.result %[[VAL_16]] : !fir.array<100xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_4]], %[[VAL_17:.*]] to %[[VAL_0]] : !fir.array<100xi32>, !fir.array<100xi32>, !fir.ref<!fir.array<100xi32>>
! CHECK:         return
! CHECK:       }

subroutine test8(a,b)
  integer :: a(100), b(100)
  a = b(1)
end subroutine test8

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

! CHECK-LABEL: func @_QPtest11(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<100xf32>>{{.*}}, %[[VAL_2:.*]]: !fir.ref<!fir.array<100xf32>>{{.*}}, %[[VAL_3:.*]]: !fir.ref<!fir.array<100xf32>>{{.*}}) {
! CHECK:         %[[VAL_4:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_6:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_7:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_8:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_0]](%[[VAL_8]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_1]](%[[VAL_10]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
! CHECK:         %[[VAL_12:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_14:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_15:.*]] = fir.array_load %[[VAL_2]](%[[VAL_14]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
! CHECK:         %[[VAL_16:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_17:.*]] = fir.array_load %[[VAL_3]](%[[VAL_16]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
! CHECK:         %[[VAL_18:.*]] = fir.allocmem !fir.array<100xf32>
! CHECK:         %[[VAL_19:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_20:.*]] = fir.array_load %[[VAL_18]](%[[VAL_19]]) : (!fir.heap<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
! CHECK:         %[[VAL_21:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_22:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_23:.*]] = arith.subi %[[VAL_12]], %[[VAL_21]] : index
! CHECK:         %[[VAL_24:.*]] = fir.do_loop %[[VAL_25:.*]] = %[[VAL_22]] to %[[VAL_23]] step %[[VAL_21]] unordered iter_args(%[[VAL_26:.*]] = %[[VAL_20]]) -> (!fir.array<100xf32>) {
! CHECK:           %[[VAL_27:.*]] = fir.array_fetch %[[VAL_15]], %[[VAL_25]] : (!fir.array<100xf32>, index) -> f32
! CHECK:           %[[VAL_28:.*]] = fir.array_fetch %[[VAL_17]], %[[VAL_25]] : (!fir.array<100xf32>, index) -> f32
! CHECK:           %[[VAL_29:.*]] = arith.addf %[[VAL_27]], %[[VAL_28]] : f32
! CHECK:           %[[VAL_30:.*]] = fir.array_update %[[VAL_26]], %[[VAL_29]], %[[VAL_25]] : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
! CHECK:           fir.result %[[VAL_30]] : !fir.array<100xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_20]], %[[VAL_31:.*]] to %[[VAL_18]] : !fir.array<100xf32>, !fir.array<100xf32>, !fir.heap<!fir.array<100xf32>>
! CHECK:         %[[VAL_32:.*]] = fir.convert %[[VAL_18]] : (!fir.heap<!fir.array<100xf32>>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:         %[[VAL_33:.*]] = fir.call @_QPbar(%[[VAL_32]]) : (!fir.ref<!fir.array<100xf32>>) -> f32
! CHECK:         %[[VAL_34:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_35:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_36:.*]] = arith.subi %[[VAL_4]], %[[VAL_34]] : index
! CHECK:         %[[VAL_37:.*]] = fir.do_loop %[[VAL_38:.*]] = %[[VAL_35]] to %[[VAL_36]] step %[[VAL_34]] unordered iter_args(%[[VAL_39:.*]] = %[[VAL_9]]) -> (!fir.array<100xf32>) {
! CHECK:           %[[VAL_40:.*]] = fir.array_fetch %[[VAL_11]], %[[VAL_38]] : (!fir.array<100xf32>, index) -> f32
! CHECK:           %[[VAL_41:.*]] = arith.addf %[[VAL_40]], %[[VAL_33]] : f32
! CHECK:           %[[VAL_42:.*]] = fir.array_update %[[VAL_39]], %[[VAL_41]], %[[VAL_38]] : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
! CHECK:           fir.result %[[VAL_42]] : !fir.array<100xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_9]], %[[VAL_43:.*]] to %[[VAL_0]] : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>
! CHECK:         fir.freemem %[[VAL_18]]
! CHECK:         return
! CHECK:       }

subroutine test11(a,b,c,d)
  real, external :: bar
  real :: a(100), b(100), c(100), d(100)
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
  ! CHECK: %[[C:.*]] = fir.array_load %arg2(%
  ! CHECK: %[[D:.*]] = fir.array_load %arg3(%
  ! CHECK: %[[tmp:.*]] = fir.allocmem !fir.array<?xf32>, %{{.*}} {{{.*}}uniq_name = ".array.expr"}
  ! CHECK: %[[T:.*]] = fir.array_load %[[tmp]](%
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
  ! CHECK: fir.freemem %[[tmp]]
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
  ! CHECK: = arith.constant -6.2598534E+18 : f32
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
! CHECK-SAME: %[[a:.*]]: !fir.ref<!fir.array<100xf32>>{{.*}}, %[[b:.*]]: !fir.ref<!fir.array<100xf32>>{{.*}})
subroutine test14(a,b)
  ! CHECK: %[[barr:.*]] = fir.array_load %[[b]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  interface
     real elemental function f1(i)
       real, intent(in) :: i
     end function f1
  end interface
  real :: a(100), b(100)
  ! CHECK: %[[loop:.*]] = fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[bth:.*]] = %[[barr]]) -> (!fir.array<100xf32>) {
  ! CHECK: %[[ishift:.*]] = arith.addi %[[i]], %c1{{.*}} : index
  ! CHECK: %[[tmp:.*]] = fir.array_coor %[[a]](%{{.*}}) %[[ishift]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: %[[fres:.*]] = fir.call @_QPf1(%[[tmp]]) : (!fir.ref<f32>) -> f32
  ! CHECK: %[[res:.*]] = fir.array_update %[[bth]], %[[fres]], %[[i]] : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
  ! CHECK: fir.result %[[res]] : !fir.array<100xf32>
  ! CHECK: fir.array_merge_store %[[barr]], %[[loop]] to %[[b]]
  b = f1(a)
end subroutine test14

! Test elemental intrinsic function (abs)
! CHECK-LABEL: func @_QPtest15(
! CHECK-SAME: %[[a:.*]]: !fir.ref<!fir.array<100xf32>>{{.*}}, %[[b:.*]]: !fir.ref<!fir.array<100xf32>>{{.*}})
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
! CHECK-SAME: %[[a:.*]]: !fir.ref<!fir.array<100xf32>>{{.*}}, %[[b:.*]]: !fir.ref<!fir.array<100xf32>>{{.*}})
subroutine test16(a,b)
  ! CHECK: %[[tmp:.*]] = fir.alloca f32 {adapt.valuebyref
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
! CHECK-SAME: %[[a:[^:]+]]: !fir.ref<!fir.array<100xf32>>{{.*}}, %[[b:[^:]+]]: !fir.ref<!fir.array<100xf32>>{{.*}}, %[[c:.*]]: !fir.ref<!fir.array<100xf32>>{{.*}})
subroutine test17(a,b,c)
  ! CHECK-DAG: %[[aarr:.*]] = fir.array_load %[[a]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  ! CHECK-DAG: %[[barr:.*]] = fir.array_load %[[b]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shapeshift<1>) -> !fir.array<100xf32>
  interface
     real elemental impure function f3(i,j,k)
       real, intent(inout) :: i, j, k
     end function f3
  end interface
  real :: a(100), b(2:101), c(3:102)
  ! CHECK: %[[loop:.*]] = fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[bth:.*]] = %[[barr]]) -> (!fir.array<100xf32>) {
  ! CHECK-DAG: %[[val:.*]] = fir.array_fetch %[[aarr]], %[[i]] : (!fir.array<100xf32>, index) -> f32
  ! CHECK-DAG: %[[ic:.*]] = arith.addi %[[i]], %c3{{.*}} : index
  ! CHECK-DAG: %[[ccoor:.*]] = fir.array_coor %[[c]](%{{.*}}) %[[ic]] : (!fir.ref<!fir.array<100xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
  ! CHECK-DAG: %[[ib:.*]] = arith.addi %[[i]], %c2{{.*}} : index
  ! CHECK-DAG: %[[bcoor:.*]] = fir.array_coor %[[b]](%{{.*}}) %[[ib]] : (!fir.ref<!fir.array<100xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
  ! CHECK-DAG: %[[ia:.*]] = arith.addi %[[i]], %c1{{.*}} : index
  ! CHECK-DAG: %[[acoor:.*]] = fir.array_coor %[[a]](%{{.*}}) %[[ia]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: %[[fres:.*]] = fir.call @_QPf3(%[[ccoor]], %[[bcoor]], %[[acoor]]) : (!fir.ref<f32>, !fir.ref<f32>, !fir.ref<f32>) -> f32
  ! CHECK: %[[fadd:.*]] = arith.addf %[[val]], %[[fres]] : f32
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

! CHECK-LABEL: func @_QPtest18() {
! CHECK:         %[[VAL_0:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.array<10x10xi32> {bindc_name = "array", fir.target, uniq_name = "_QFtest18Earray"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest18Ei"}
! CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "row_i", uniq_name = "_QFtest18Erow_i"}
! CHECK:         %[[VAL_5:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! CHECK:         %[[VAL_6:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_8:.*]] = fir.embox %[[VAL_5]](%[[VAL_7]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:         fir.store %[[VAL_8]] to %[[VAL_4]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> i64
! CHECK:         %[[VAL_12:.*]] = fir.undefined index
! CHECK:         %[[VAL_13:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:         %[[VAL_15:.*]] = arith.addi %[[VAL_9]], %[[VAL_1]] : index
! CHECK:         %[[VAL_16:.*]] = arith.subi %[[VAL_15]], %[[VAL_9]] : index
! CHECK:         %[[VAL_17:.*]] = fir.shape %[[VAL_0]], %[[VAL_1]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_18:.*]] = fir.slice %[[VAL_11]], %[[VAL_12]], %[[VAL_12]], %[[VAL_9]], %[[VAL_16]], %[[VAL_14]] : (i64, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         %[[VAL_19:.*]] = fir.embox %[[VAL_2]](%[[VAL_17]]) {{\[}}%[[VAL_18]]] : (!fir.ref<!fir.array<10x10xi32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?xi32>>
! CHECK:         %[[VAL_20:.*]] = fir.rebox %[[VAL_19]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:         fir.store %[[VAL_20]] to %[[VAL_4]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         return
! CHECK:       }

subroutine test18
  integer, target :: array(10,10)
  integer, pointer :: row_i(:)
  row_i => array(i, :)
end subroutine test18

! CHECK-LABEL: func @_QPtest_column_and_row_order(
! CHECK-SAME:              %[[VAL_0:.*]]: !fir.ref<!fir.array<2x3xf32>>{{.*}}) {
! CHECK:         %[[VAL_1:.*]] = arith.constant 2 : index
! CHECK:         %[[VAL_2:.*]] = arith.constant 3 : index
! CHECK:         %[[VAL_3:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_4:.*]] = fir.array_load %[[VAL_0]](%[[VAL_3]]) : (!fir.ref<!fir.array<2x3xf32>>, !fir.shape<2>) -> !fir.array<2x3xf32>
! CHECK:         %[[VAL_5:.*]] = arith.constant 42 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> f32
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_9:.*]] = arith.subi %[[VAL_1]], %[[VAL_7]] : index
! CHECK:         %[[VAL_10:.*]] = arith.subi %[[VAL_2]], %[[VAL_7]] : index
! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_8]] to %[[VAL_10]] step %[[VAL_7]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_4]]) -> (!fir.array<2x3xf32>) {
! CHECK:           %[[VAL_14:.*]] = fir.do_loop %[[VAL_15:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_7]] unordered iter_args(%[[VAL_16:.*]] = %[[VAL_13]]) -> (!fir.array<2x3xf32>) {
! CHECK:             %[[VAL_17:.*]] = fir.array_update %[[VAL_16]], %[[VAL_6]], %[[VAL_15]], %[[VAL_12]] : (!fir.array<2x3xf32>, f32, index, index) -> !fir.array<2x3xf32>
! CHECK:             fir.result %[[VAL_17]] : !fir.array<2x3xf32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_18:.*]] : !fir.array<2x3xf32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_4]], %[[VAL_19:.*]] to %[[VAL_0]] : !fir.array<2x3xf32>, !fir.array<2x3xf32>, !fir.ref<!fir.array<2x3xf32>>
! CHECK:         return
! CHECK:       }

subroutine test_column_and_row_order(x)
  real :: x(2,3)
  x = 42
end subroutine

! CHECK-LABEL: func @_QPtest_assigning_to_assumed_shape_slices(
! CHECK-SAME:                %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) {
! CHECK:         %[[VAL_1:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_2:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (i64) -> index
! CHECK:         %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_6:.*]] = arith.addi %[[VAL_1]], %[[VAL_5]]#1 : index
! CHECK:         %[[VAL_7:.*]] = arith.subi %[[VAL_6]], %[[VAL_1]] : index
! CHECK:         %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_9:.*]] = arith.subi %[[VAL_7]], %[[VAL_1]] : index
! CHECK:         %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_3]] : index
! CHECK:         %[[VAL_11:.*]] = arith.divsi %[[VAL_10]], %[[VAL_3]] : index
! CHECK:         %[[VAL_12:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[VAL_8]] : index
! CHECK:         %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[VAL_11]], %[[VAL_8]] : index
! CHECK:         %[[VAL_14:.*]] = fir.slice %[[VAL_1]], %[[VAL_7]], %[[VAL_3]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_15:.*]] = fir.array_load %[[VAL_0]] {{\[}}%[[VAL_14]]] : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_16:.*]] = arith.constant 42 : i32
! CHECK:         %[[VAL_17:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_18:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_19:.*]] = arith.subi %[[VAL_13]], %[[VAL_17]] : index
! CHECK:         %[[VAL_20:.*]] = fir.do_loop %[[VAL_21:.*]] = %[[VAL_18]] to %[[VAL_19]] step %[[VAL_17]] unordered iter_args(%[[VAL_22:.*]] = %[[VAL_15]]) -> (!fir.array<?xi32>) {
! CHECK:           %[[VAL_23:.*]] = fir.array_update %[[VAL_22]], %[[VAL_16]], %[[VAL_21]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:           fir.result %[[VAL_23]] : !fir.array<?xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_15]], %[[VAL_24:.*]] to %[[VAL_0]]{{\[}}%[[VAL_14]]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?xi32>>, !fir.slice<1>
! CHECK:         return
! CHECK:       }

subroutine test_assigning_to_assumed_shape_slices(x)
  integer :: x(:)
  x(::2) = 42
end subroutine

! CHECK-LABEL: func @_QPtest19a(
! CHECK-SAME:      %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}, %[[VAL_1:.*]]: !fir.boxchar<1>{{.*}}) {
! CHECK:         %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,10>>>
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,10>>>
! CHECK:         %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_8:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_3]](%[[VAL_8]]) : (!fir.ref<!fir.array<10x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.array<10x!fir.char<1,10>>
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_6]](%[[VAL_10]]) : (!fir.ref<!fir.array<10x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.array<10x!fir.char<1,10>>
! CHECK:         %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_14:.*]] = arith.subi %[[VAL_4]], %[[VAL_12]] : index
! CHECK:         %[[VAL_15:.*]] = fir.do_loop %[[VAL_16:.*]] = %[[VAL_13]] to %[[VAL_14]] step %[[VAL_12]] unordered iter_args(%[[VAL_17:.*]] = %[[VAL_9]]) -> (!fir.array<10x!fir.char<1,10>>) {
! CHECK:           %[[VAL_18:.*]] = fir.array_access %[[VAL_11]], %[[VAL_16]] : (!fir.array<10x!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
! CHECK:           %[[VAL_19:.*]] = fir.array_access %[[VAL_17]], %[[VAL_16]] : (!fir.array<10x!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
! CHECK:           %[[VAL_20:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_21:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_20]] : (index) -> i64
! CHECK:           %[[VAL_23:.*]] = arith.muli %[[VAL_21]], %[[VAL_22]] : i64
! CHECK:           %[[VAL_24:.*]] = arith.constant false
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:           fir.call @llvm.memmove.p0.p0.i64(%[[VAL_25]], %[[VAL_26]], %[[VAL_23]], %[[VAL_24]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:           %[[VAL_27:.*]] = fir.array_amend %[[VAL_17]], %[[VAL_19]] : (!fir.array<10x!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>) -> !fir.array<10x!fir.char<1,10>>
! CHECK:           fir.result %[[VAL_27]] : !fir.array<10x!fir.char<1,10>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_9]], %[[VAL_28:.*]] to %[[VAL_3]] : !fir.array<10x!fir.char<1,10>>, !fir.array<10x!fir.char<1,10>>, !fir.ref<!fir.array<10x!fir.char<1,10>>>
! CHECK:         return
! CHECK:       }

subroutine test19a(a,b)
  character(LEN=10) a(10)
  character(LEN=10) b(10)
  a = b
end subroutine test19a

! CHECK-LABEL: func @_QPtest19b(
! CHECK-SAME:                   %[[VAL_0:.*]]: !fir.boxchar<2>{{.*}}, %[[VAL_1:.*]]: !fir.boxchar<2>{{.*}}) {
! CHECK:         %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.array<20x!fir.char<2,8>>>
! CHECK:         %[[VAL_4:.*]] = arith.constant 20 : index
! CHECK:         %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
! CHECK:         %[[VAL_6:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.array<20x!fir.char<2,10>>>
! CHECK:         %[[VAL_8:.*]] = arith.constant 20 : index
! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_3]](%[[VAL_9]]) : (!fir.ref<!fir.array<20x!fir.char<2,8>>>, !fir.shape<1>) -> !fir.array<20x!fir.char<2,8>>
! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_7]](%[[VAL_11]]) : (!fir.ref<!fir.array<20x!fir.char<2,10>>>, !fir.shape<1>) -> !fir.array<20x!fir.char<2,10>>
! CHECK:         %[[VAL_13:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_14:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_15:.*]] = arith.subi %[[VAL_4]], %[[VAL_13]] : index
! CHECK:         %[[VAL_16:.*]] = fir.do_loop %[[VAL_17:.*]] = %[[VAL_14]] to %[[VAL_15]] step %[[VAL_13]] unordered iter_args(%[[VAL_18:.*]] = %[[VAL_10]]) -> (!fir.array<20x!fir.char<2,8>>) {
! CHECK:           %[[VAL_19:.*]] = fir.array_access %[[VAL_12]], %[[VAL_17]] : (!fir.array<20x!fir.char<2,10>>, index) -> !fir.ref<!fir.char<2,10>>
! CHECK:           %[[VAL_20:.*]] = fir.array_access %[[VAL_18]], %[[VAL_17]] : (!fir.array<20x!fir.char<2,8>>, index) -> !fir.ref<!fir.char<2,8>>
! CHECK:           %[[VAL_21:.*]] = arith.constant 8 : index
! CHECK:           %[[VAL_22:.*]] = arith.cmpi slt, %[[VAL_21]], %[[VAL_6]] : index
! CHECK:           %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_21]], %[[VAL_6]] : index
! CHECK:           %[[VAL_24:.*]] = arith.constant 2 : i64
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_23]] : (index) -> i64
! CHECK:           %[[VAL_26:.*]] = arith.muli %[[VAL_24]], %[[VAL_25]] : i64
! CHECK:           %[[VAL_27:.*]] = arith.constant false
! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.char<2,8>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<2,10>>) -> !fir.ref<i8>
! CHECK:           fir.call @llvm.memmove.p0.p0.i64(%[[VAL_28]], %[[VAL_29]], %[[VAL_26]], %[[VAL_27]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:           %[[VAL_30:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_31:.*]] = arith.subi %[[VAL_21]], %[[VAL_30]] : index
! CHECK:           %[[VAL_32:.*]] = arith.constant 32 : i16
! CHECK:           %[[VAL_33:.*]] = fir.undefined !fir.char<2>
! CHECK:           %[[VAL_34:.*]] = fir.insert_value %[[VAL_33]], %[[VAL_32]], [0 : index] : (!fir.char<2>, i16) -> !fir.char<2>
! CHECK:           %[[VAL_35:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_36:.*]] = %[[VAL_23]] to %[[VAL_31]] step %[[VAL_35]] {
! CHECK:             %[[VAL_37:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.char<2,8>>) -> !fir.ref<!fir.array<8x!fir.char<2>>>
! CHECK:             %[[VAL_38:.*]] = fir.coordinate_of %[[VAL_37]], %[[VAL_36]] : (!fir.ref<!fir.array<8x!fir.char<2>>>, index) -> !fir.ref<!fir.char<2>>
! CHECK:             fir.store %[[VAL_34]] to %[[VAL_38]] : !fir.ref<!fir.char<2>>
! CHECK:           }
! CHECK:           %[[VAL_39:.*]] = fir.array_amend %[[VAL_18]], %[[VAL_20]] : (!fir.array<20x!fir.char<2,8>>, !fir.ref<!fir.char<2,8>>) -> !fir.array<20x!fir.char<2,8>>
! CHECK:           fir.result %[[VAL_39]] : !fir.array<20x!fir.char<2,8>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_10]], %[[VAL_40:.*]] to %[[VAL_3]] : !fir.array<20x!fir.char<2,8>>, !fir.array<20x!fir.char<2,8>>, !fir.ref<!fir.array<20x!fir.char<2,8>>>
! CHECK:         return
! CHECK:       }

subroutine test19b(a,b)
  character(KIND=2, LEN=8) a(20)
  character(KIND=2, LEN=10) b(20)
  a = b
end subroutine test19b

! CHECK-LABEL: func @_QPtest19c(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<4>{{.*}}, %[[VAL_1:.*]]: !fir.boxchar<4>{{.*}}, %[[VAL_2:.*]]: !fir.ref<i32>{{.*}}) {
! CHECK:         %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<30x!fir.char<4,10>>>
! CHECK:         %[[VAL_6:.*]] = arith.constant 30 : index
! CHECK:         %[[VAL_7:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_9:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_9]] : i32
! CHECK:         %[[VAL_11:.*]] = arith.select %[[VAL_10]], %[[VAL_8]], %[[VAL_9]] : i32
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<30x!fir.char<4,?>>>
! CHECK:         %[[VAL_13:.*]] = arith.constant 30 : index
! CHECK:         %[[VAL_14:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_15:.*]] = fir.array_load %[[VAL_12]](%[[VAL_14]]) typeparams %[[VAL_11]] : (!fir.ref<!fir.array<30x!fir.char<4,?>>>, !fir.shape<1>, i32) -> !fir.array<30x!fir.char<4,?>>
! CHECK:         %[[VAL_16:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_17:.*]] = fir.array_load %[[VAL_5]](%[[VAL_16]]) : (!fir.ref<!fir.array<30x!fir.char<4,10>>>, !fir.shape<1>) -> !fir.array<30x!fir.char<4,10>>
! CHECK:         %[[VAL_18:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_19:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_20:.*]] = arith.subi %[[VAL_13]], %[[VAL_18]] : index
! CHECK:         %[[VAL_21:.*]] = fir.do_loop %[[VAL_22:.*]] = %[[VAL_19]] to %[[VAL_20]] step %[[VAL_18]] unordered iter_args(%[[VAL_23:.*]] = %[[VAL_15]]) -> (!fir.array<30x!fir.char<4,?>>) {
! CHECK:           %[[VAL_24:.*]] = fir.array_access %[[VAL_17]], %[[VAL_22]] : (!fir.array<30x!fir.char<4,10>>, index) -> !fir.ref<!fir.char<4,10>>
! CHECK:           %[[VAL_25:.*]] = fir.array_access %[[VAL_23]], %[[VAL_22]] typeparams %[[VAL_11]] : (!fir.array<30x!fir.char<4,?>>, index, i32) -> !fir.ref<!fir.char<4,?>>
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
! CHECK:           %[[VAL_27:.*]] = arith.cmpi slt, %[[VAL_26]], %[[VAL_4]] : index
! CHECK:           %[[VAL_28:.*]] = arith.select %[[VAL_27]], %[[VAL_26]], %[[VAL_4]] : index
! CHECK:           %[[VAL_29:.*]] = arith.constant 4 : i64
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_28]] : (index) -> i64
! CHECK:           %[[VAL_31:.*]] = arith.muli %[[VAL_29]], %[[VAL_30]] : i64
! CHECK:           %[[VAL_32:.*]] = arith.constant false
! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<4,10>>) -> !fir.ref<i8>
! CHECK:           fir.call @llvm.memmove.p0.p0.i64(%[[VAL_33]], %[[VAL_34]], %[[VAL_31]], %[[VAL_32]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:           %[[VAL_35:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_36:.*]] = arith.subi %[[VAL_11]], %[[VAL_35]] : i32
! CHECK:           %[[VAL_37:.*]] = arith.constant 32 : i32
! CHECK:           %[[VAL_38:.*]] = fir.undefined !fir.char<4>
! CHECK:           %[[VAL_39:.*]] = fir.insert_value %[[VAL_38]], %[[VAL_37]], [0 : index] : (!fir.char<4>, i32) -> !fir.char<4>
! CHECK:           %[[VAL_40:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_41:.*]] = fir.convert %[[VAL_36]] : (i32) -> index
! CHECK:           fir.do_loop %[[VAL_42:.*]] = %[[VAL_28]] to %[[VAL_41]] step %[[VAL_40]] {
! CHECK:             %[[VAL_43:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<?x!fir.char<4>>>
! CHECK:             %[[VAL_44:.*]] = fir.coordinate_of %[[VAL_43]], %[[VAL_42]] : (!fir.ref<!fir.array<?x!fir.char<4>>>, index) -> !fir.ref<!fir.char<4>>
! CHECK:             fir.store %[[VAL_39]] to %[[VAL_44]] : !fir.ref<!fir.char<4>>
! CHECK:           }
! CHECK:           %[[VAL_45:.*]] = fir.array_amend %[[VAL_23]], %[[VAL_25]] : (!fir.array<30x!fir.char<4,?>>, !fir.ref<!fir.char<4,?>>) -> !fir.array<30x!fir.char<4,?>>
! CHECK:           fir.result %[[VAL_45]] : !fir.array<30x!fir.char<4,?>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_15]], %[[VAL_46:.*]] to %[[VAL_12]] typeparams %[[VAL_11]] : !fir.array<30x!fir.char<4,?>>, !fir.array<30x!fir.char<4,?>>, !fir.ref<!fir.array<30x!fir.char<4,?>>>, i32
! CHECK:         return
! CHECK:       }

subroutine test19c(a,b,i)
  character(KIND=4, LEN=i) a(30)
  character(KIND=4, LEN=10) b(30)
  a = b
end subroutine test19c

! CHECK-LABEL: func @_QPtest19d(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}, %[[VAL_1:.*]]: !fir.boxchar<1>{{.*}}, %[[VAL_2:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_3:.*]]: !fir.ref<i32>{{.*}}) {
! CHECK:         %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_5:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_6:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_7:.*]] = arith.cmpi sgt, %[[VAL_5]], %[[VAL_6]] : i32
! CHECK:         %[[VAL_8:.*]] = arith.select %[[VAL_7]], %[[VAL_5]], %[[VAL_6]] : i32
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<40x!fir.char<1,?>>>
! CHECK:         %[[VAL_10:.*]] = arith.constant 40 : index
! CHECK:         %[[VAL_11:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:         %[[VAL_13:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_13]] : i32
! CHECK:         %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_12]], %[[VAL_13]] : i32
! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_11]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<40x!fir.char<1,?>>>
! CHECK:         %[[VAL_17:.*]] = arith.constant 40 : index
! CHECK:         %[[VAL_18:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_19:.*]] = fir.array_load %[[VAL_9]](%[[VAL_18]]) typeparams %[[VAL_8]] : (!fir.ref<!fir.array<40x!fir.char<1,?>>>, !fir.shape<1>, i32) -> !fir.array<40x!fir.char<1,?>>
! CHECK:         %[[VAL_20:.*]] = fir.shape %[[VAL_17]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_21:.*]] = fir.array_load %[[VAL_16]](%[[VAL_20]]) typeparams %[[VAL_15]] : (!fir.ref<!fir.array<40x!fir.char<1,?>>>, !fir.shape<1>, i32) -> !fir.array<40x!fir.char<1,?>>
! CHECK:         %[[VAL_22:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_23:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_24:.*]] = arith.subi %[[VAL_10]], %[[VAL_22]] : index
! CHECK:         %[[VAL_25:.*]] = fir.do_loop %[[VAL_26:.*]] = %[[VAL_23]] to %[[VAL_24]] step %[[VAL_22]] unordered iter_args(%[[VAL_27:.*]] = %[[VAL_19]]) -> (!fir.array<40x!fir.char<1,?>>) {
! CHECK:           %[[VAL_28:.*]] = fir.array_access %[[VAL_21]], %[[VAL_26]] typeparams %[[VAL_15]] : (!fir.array<40x!fir.char<1,?>>, index, i32) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_29:.*]] = fir.array_access %[[VAL_27]], %[[VAL_26]] typeparams %[[VAL_8]] : (!fir.array<40x!fir.char<1,?>>, index, i32) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_15]] : (i32) -> index
! CHECK:           %[[VAL_32:.*]] = arith.cmpi slt, %[[VAL_30]], %[[VAL_31]] : index
! CHECK:           %[[VAL_33:.*]] = arith.select %[[VAL_32]], %[[VAL_30]], %[[VAL_31]] : index
! CHECK:           %[[VAL_34:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_33]] : (index) -> i64
! CHECK:           %[[VAL_36:.*]] = arith.muli %[[VAL_34]], %[[VAL_35]] : i64
! CHECK:           %[[VAL_37:.*]] = arith.constant false
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_29]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_39:.*]] = fir.convert %[[VAL_28]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:           fir.call @llvm.memmove.p0.p0.i64(%[[VAL_38]], %[[VAL_39]], %[[VAL_36]], %[[VAL_37]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:           %[[VAL_40:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_41:.*]] = arith.subi %[[VAL_8]], %[[VAL_40]] : i32
! CHECK:           %[[VAL_42:.*]] = arith.constant 32 : i8
! CHECK:           %[[VAL_43:.*]] = fir.undefined !fir.char<1>
! CHECK:           %[[VAL_44:.*]] = fir.insert_value %[[VAL_43]], %[[VAL_42]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:           %[[VAL_45:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_46:.*]] = fir.convert %[[VAL_41]] : (i32) -> index
! CHECK:           fir.do_loop %[[VAL_47:.*]] = %[[VAL_33]] to %[[VAL_46]] step %[[VAL_45]] {
! CHECK:             %[[VAL_48:.*]] = fir.convert %[[VAL_29]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:             %[[VAL_49:.*]] = fir.coordinate_of %[[VAL_48]], %[[VAL_47]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:             fir.store %[[VAL_44]] to %[[VAL_49]] : !fir.ref<!fir.char<1>>
! CHECK:           }
! CHECK:           %[[VAL_50:.*]] = fir.array_amend %[[VAL_27]], %[[VAL_29]] : (!fir.array<40x!fir.char<1,?>>, !fir.ref<!fir.char<1,?>>) -> !fir.array<40x!fir.char<1,?>>
! CHECK:           fir.result %[[VAL_50]] : !fir.array<40x!fir.char<1,?>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_19]], %[[VAL_51:.*]] to %[[VAL_9]] typeparams %[[VAL_8]] : !fir.array<40x!fir.char<1,?>>, !fir.array<40x!fir.char<1,?>>, !fir.ref<!fir.array<40x!fir.char<1,?>>>, i32
! CHECK:         return
! CHECK:       }

subroutine test19d(a,b,i,j)
  character(i) a(40)
  character(j) b(40)
  a = b
end subroutine test19d

! CHECK-LABEL: func @_QPtest19e(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}, %[[VAL_1:.*]]: !fir.boxchar<1>{{.*}}) {
! CHECK:         %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<50x!fir.char<1,?>>>
! CHECK:         %[[VAL_4:.*]] = arith.constant 50 : index
! CHECK:         %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<50x!fir.char<1,?>>>
! CHECK:         %[[VAL_7:.*]] = arith.constant 50 : index
! CHECK:         %[[VAL_8:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_3]](%[[VAL_8]]) typeparams %[[VAL_2]]#1 : (!fir.ref<!fir.array<50x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.array<50x!fir.char<1,?>>
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_6]](%[[VAL_10]]) typeparams %[[VAL_5]]#1 : (!fir.ref<!fir.array<50x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.array<50x!fir.char<1,?>>
! CHECK:         %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_14:.*]] = arith.subi %[[VAL_4]], %[[VAL_12]] : index
! CHECK:         %[[VAL_15:.*]] = fir.do_loop %[[VAL_16:.*]] = %[[VAL_13]] to %[[VAL_14]] step %[[VAL_12]] unordered iter_args(%[[VAL_17:.*]] = %[[VAL_9]]) -> (!fir.array<50x!fir.char<1,?>>) {
! CHECK:           %[[VAL_18:.*]] = fir.array_access %[[VAL_11]], %[[VAL_16]] typeparams %[[VAL_5]]#1 : (!fir.array<50x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_19:.*]] = fir.array_access %[[VAL_17]], %[[VAL_16]] typeparams %[[VAL_2]]#1 : (!fir.array<50x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_20:.*]] = arith.cmpi slt, %[[VAL_2]]#1, %[[VAL_5]]#1 : index
! CHECK:           %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_2]]#1, %[[VAL_5]]#1 : index
! CHECK:           %[[VAL_22:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_21]] : (index) -> i64
! CHECK:           %[[VAL_24:.*]] = arith.muli %[[VAL_22]], %[[VAL_23]] : i64
! CHECK:           %[[VAL_25:.*]] = arith.constant false
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:           fir.call @llvm.memmove.p0.p0.i64(%[[VAL_26]], %[[VAL_27]], %[[VAL_24]], %[[VAL_25]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:           %[[VAL_28:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_29:.*]] = arith.subi %[[VAL_2]]#1, %[[VAL_28]] : index
! CHECK:           %[[VAL_30:.*]] = arith.constant 32 : i8
! CHECK:           %[[VAL_31:.*]] = fir.undefined !fir.char<1>
! CHECK:           %[[VAL_32:.*]] = fir.insert_value %[[VAL_31]], %[[VAL_30]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:           %[[VAL_33:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_34:.*]] = %[[VAL_21]] to %[[VAL_29]] step %[[VAL_33]] {
! CHECK:             %[[VAL_35:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:             %[[VAL_36:.*]] = fir.coordinate_of %[[VAL_35]], %[[VAL_34]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:             fir.store %[[VAL_32]] to %[[VAL_36]] : !fir.ref<!fir.char<1>>
! CHECK:           }
! CHECK:           %[[VAL_37:.*]] = fir.array_amend %[[VAL_17]], %[[VAL_19]] : (!fir.array<50x!fir.char<1,?>>, !fir.ref<!fir.char<1,?>>) -> !fir.array<50x!fir.char<1,?>>
! CHECK:           fir.result %[[VAL_37]] : !fir.array<50x!fir.char<1,?>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_9]], %[[VAL_38:.*]] to %[[VAL_3]] typeparams %[[VAL_2]]#1 : !fir.array<50x!fir.char<1,?>>, !fir.array<50x!fir.char<1,?>>, !fir.ref<!fir.array<50x!fir.char<1,?>>>, index
! CHECK:         return
! CHECK:       }

subroutine test19e(a,b)
  character(*) a(50)
  character(*) b(50)
  a = b
end subroutine test19e

! CHECK-LABEL: func @_QPtest19f(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}, %[[VAL_1:.*]]: !fir.boxchar<1>{{.*}}) {
! CHECK:         %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<60x!fir.char<1,?>>>
! CHECK:         %[[VAL_4:.*]] = arith.constant 60 : index
! CHECK:         %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<60x!fir.char<1,?>>>
! CHECK:         %[[VAL_7:.*]] = arith.constant 60 : index
! CHECK:         %[[VAL_8:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_3]](%[[VAL_8]]) typeparams %[[VAL_2]]#1 : (!fir.ref<!fir.array<60x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.array<60x!fir.char<1,?>>
! CHECK:         %[[VAL_10:.*]] = fir.address_of(@_QQcl.70726566697820) : !fir.ref<!fir.char<1,7>>
! CHECK:         %[[VAL_11:.*]] = arith.constant 7 : index
! CHECK:         %[[VAL_12:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_13:.*]] = fir.array_load %[[VAL_6]](%[[VAL_12]]) typeparams %[[VAL_5]]#1 : (!fir.ref<!fir.array<60x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.array<60x!fir.char<1,?>>
! CHECK:         %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_15:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_16:.*]] = arith.subi %[[VAL_4]], %[[VAL_14]] : index
! CHECK:         %[[VAL_17:.*]] = fir.do_loop %[[VAL_18:.*]] = %[[VAL_15]] to %[[VAL_16]] step %[[VAL_14]] unordered iter_args(%[[VAL_19:.*]] = %[[VAL_9]]) -> (!fir.array<60x!fir.char<1,?>>) {
! CHECK:           %[[VAL_20:.*]] = fir.array_access %[[VAL_13]], %[[VAL_18]] typeparams %[[VAL_5]]#1 : (!fir.array<60x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_21:.*]] = arith.addi %[[VAL_11]], %[[VAL_5]]#1 : index
! CHECK:           %[[VAL_22:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_21]] : index) {bindc_name = ".chrtmp"}
! CHECK:           %[[VAL_23:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_11]] : (index) -> i64
! CHECK:           %[[VAL_25:.*]] = arith.muli %[[VAL_23]], %[[VAL_24]] : i64
! CHECK:           %[[VAL_26:.*]] = arith.constant false
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.char<1,7>>) -> !fir.ref<i8>
! CHECK:           fir.call @llvm.memmove.p0.p0.i64(%[[VAL_27]], %[[VAL_28]], %[[VAL_25]], %[[VAL_26]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:           %[[VAL_29:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_30:.*]] = arith.subi %[[VAL_21]], %[[VAL_29]] : index
! CHECK:           fir.do_loop %[[VAL_31:.*]] = %[[VAL_11]] to %[[VAL_30]] step %[[VAL_29]] {
! CHECK:             %[[VAL_32:.*]] = arith.subi %[[VAL_31]], %[[VAL_11]] : index
! CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:             %[[VAL_34:.*]] = fir.coordinate_of %[[VAL_33]], %[[VAL_32]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:             %[[VAL_35:.*]] = fir.load %[[VAL_34]] : !fir.ref<!fir.char<1>>
! CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:             %[[VAL_37:.*]] = fir.coordinate_of %[[VAL_36]], %[[VAL_31]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:             fir.store %[[VAL_35]] to %[[VAL_37]] : !fir.ref<!fir.char<1>>
! CHECK:           }
! CHECK:           %[[VAL_38:.*]] = fir.array_access %[[VAL_19]], %[[VAL_18]] typeparams %[[VAL_2]]#1 : (!fir.array<60x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_39:.*]] = arith.cmpi slt, %[[VAL_2]]#1, %[[VAL_21]] : index
! CHECK:           %[[VAL_40:.*]] = arith.select %[[VAL_39]], %[[VAL_2]]#1, %[[VAL_21]] : index
! CHECK:           %[[VAL_41:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_42:.*]] = fir.convert %[[VAL_40]] : (index) -> i64
! CHECK:           %[[VAL_43:.*]] = arith.muli %[[VAL_41]], %[[VAL_42]] : i64
! CHECK:           %[[VAL_44:.*]] = arith.constant false
! CHECK:           %[[VAL_45:.*]] = fir.convert %[[VAL_38]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_46:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:           fir.call @llvm.memmove.p0.p0.i64(%[[VAL_45]], %[[VAL_46]], %[[VAL_43]], %[[VAL_44]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:           %[[VAL_47:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_48:.*]] = arith.subi %[[VAL_2]]#1, %[[VAL_47]] : index
! CHECK:           %[[VAL_49:.*]] = arith.constant 32 : i8
! CHECK:           %[[VAL_50:.*]] = fir.undefined !fir.char<1>
! CHECK:           %[[VAL_51:.*]] = fir.insert_value %[[VAL_50]], %[[VAL_49]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:           %[[VAL_52:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_53:.*]] = %[[VAL_40]] to %[[VAL_48]] step %[[VAL_52]] {
! CHECK:             %[[VAL_54:.*]] = fir.convert %[[VAL_38]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:             %[[VAL_55:.*]] = fir.coordinate_of %[[VAL_54]], %[[VAL_53]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:             fir.store %[[VAL_51]] to %[[VAL_55]] : !fir.ref<!fir.char<1>>
! CHECK:           }
! CHECK:           %[[VAL_56:.*]] = fir.array_amend %[[VAL_19]], %[[VAL_38]] : (!fir.array<60x!fir.char<1,?>>, !fir.ref<!fir.char<1,?>>) -> !fir.array<60x!fir.char<1,?>>
! CHECK:           fir.result %[[VAL_56]] : !fir.array<60x!fir.char<1,?>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_9]], %[[VAL_57:.*]] to %[[VAL_3]] typeparams %[[VAL_2]]#1 : !fir.array<60x!fir.char<1,?>>, !fir.array<60x!fir.char<1,?>>, !fir.ref<!fir.array<60x!fir.char<1,?>>>, index
! CHECK:         return
! CHECK:       }

subroutine test19f(a,b)
  character(*) a(60)
  character(*) b(60)
  a = "prefix " // b
end subroutine test19f

! CHECK-LABEL: func @_QPtest19g(
! CHECK-SAME:            %[[VAL_0:.*]]: !fir.boxchar<4>{{.*}}, %[[VAL_1:.*]]: !fir.boxchar<2>{{.*}}, %[[VAL_2:.*]]: !fir.ref<i32>{{.*}}) {
! CHECK:         %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
! CHECK:         %[[VAL_4:.*]] = arith.constant 13 : index
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.array<140x!fir.char<2,13>>>
! CHECK:         %[[VAL_6:.*]] = arith.constant 140 : index
! CHECK:         %[[VAL_7:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_9:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_9]] : i32
! CHECK:         %[[VAL_11:.*]] = arith.select %[[VAL_10]], %[[VAL_8]], %[[VAL_9]] : i32
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<70x!fir.char<4,?>>>
! CHECK:         %[[VAL_13:.*]] = arith.constant 70 : index
! CHECK:         %[[VAL_14:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_15:.*]] = fir.array_load %[[VAL_12]](%[[VAL_14]]) typeparams %[[VAL_11]] : (!fir.ref<!fir.array<70x!fir.char<4,?>>>, !fir.shape<1>, i32) -> !fir.array<70x!fir.char<4,?>>
! CHECK:         %[[VAL_16:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
! CHECK:         %[[VAL_18:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i64) -> index
! CHECK:         %[[VAL_20:.*]] = arith.constant 140 : i64
! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
! CHECK:         %[[VAL_22:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_23:.*]] = fir.slice %[[VAL_17]], %[[VAL_21]], %[[VAL_19]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_24:.*]] = fir.array_load %[[VAL_5]](%[[VAL_22]]) {{\[}}%[[VAL_23]]] : (!fir.ref<!fir.array<140x!fir.char<2,13>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<140x!fir.char<2,13>>
! CHECK:         %[[VAL_25:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i32) -> i64
! CHECK:         %[[VAL_27:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_28:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_29:.*]] = arith.subi %[[VAL_13]], %[[VAL_27]] : index
! CHECK:         %[[VAL_30:.*]] = fir.do_loop %[[VAL_31:.*]] = %[[VAL_28]] to %[[VAL_29]] step %[[VAL_27]] unordered iter_args(%[[VAL_32:.*]] = %[[VAL_15]]) -> (!fir.array<70x!fir.char<4,?>>) {
! CHECK:           %[[VAL_33:.*]] = fir.array_access %[[VAL_24]], %[[VAL_31]] : (!fir.array<140x!fir.char<2,13>>, index) -> !fir.ref<!fir.char<2,13>>
! CHECK:           %[[VAL_34:.*]] = fir.alloca !fir.char<4,?>(%[[VAL_4]] : index)
! CHECK:           %[[VAL_35:.*]] = arith.cmpi slt, %[[VAL_4]], %[[VAL_4]] : index
! CHECK:           %[[VAL_36:.*]] = arith.select %[[VAL_35]], %[[VAL_4]], %[[VAL_4]] : index
! CHECK:           fir.char_convert %[[VAL_33]] for %[[VAL_36]] to %[[VAL_34]] : !fir.ref<!fir.char<2,13>>, index, !fir.ref<!fir.char<4,?>>
! CHECK:           %[[VAL_37:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_38:.*]] = arith.subi %[[VAL_4]], %[[VAL_37]] : index
! CHECK:           %[[VAL_39:.*]] = arith.constant 32 : i32
! CHECK:           %[[VAL_40:.*]] = fir.undefined !fir.char<4>
! CHECK:           %[[VAL_41:.*]] = fir.insert_value %[[VAL_40]], %[[VAL_39]], [0 : index] : (!fir.char<4>, i32) -> !fir.char<4>
! CHECK:           %[[VAL_42:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_43:.*]] = %[[VAL_36]] to %[[VAL_38]] step %[[VAL_42]] {
! CHECK:             %[[VAL_44:.*]] = fir.convert %[[VAL_34]] : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<?x!fir.char<4>>>
! CHECK:             %[[VAL_45:.*]] = fir.coordinate_of %[[VAL_44]], %[[VAL_43]] : (!fir.ref<!fir.array<?x!fir.char<4>>>, index) -> !fir.ref<!fir.char<4>>
! CHECK:             fir.store %[[VAL_41]] to %[[VAL_45]] : !fir.ref<!fir.char<4>>
! CHECK:           }
! CHECK:           %[[VAL_46:.*]] = fir.array_access %[[VAL_32]], %[[VAL_31]] typeparams %[[VAL_11]] : (!fir.array<70x!fir.char<4,?>>, index, i32) -> !fir.ref<!fir.char<4,?>>
! CHECK:           %[[VAL_47:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
! CHECK:           %[[VAL_48:.*]] = fir.convert %[[VAL_26]] : (i64) -> index
! CHECK:           %[[VAL_49:.*]] = arith.cmpi slt, %[[VAL_47]], %[[VAL_48]] : index
! CHECK:           %[[VAL_50:.*]] = arith.select %[[VAL_49]], %[[VAL_47]], %[[VAL_48]] : index
! CHECK:           %[[VAL_51:.*]] = arith.constant 4 : i64
! CHECK:           %[[VAL_52:.*]] = fir.convert %[[VAL_50]] : (index) -> i64
! CHECK:           %[[VAL_53:.*]] = arith.muli %[[VAL_51]], %[[VAL_52]] : i64
! CHECK:           %[[VAL_54:.*]] = arith.constant false
! CHECK:           %[[VAL_55:.*]] = fir.convert %[[VAL_46]] : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_56:.*]] = fir.convert %[[VAL_34]] : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<i8>
! CHECK:           fir.call @llvm.memmove.p0.p0.i64(%[[VAL_55]], %[[VAL_56]], %[[VAL_53]], %[[VAL_54]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:           %[[VAL_57:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_58:.*]] = arith.subi %[[VAL_11]], %[[VAL_57]] : i32
! CHECK:           %[[VAL_59:.*]] = arith.constant 32 : i32
! CHECK:           %[[VAL_60:.*]] = fir.undefined !fir.char<4>
! CHECK:           %[[VAL_61:.*]] = fir.insert_value %[[VAL_60]], %[[VAL_59]], [0 : index] : (!fir.char<4>, i32) -> !fir.char<4>
! CHECK:           %[[VAL_62:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_63:.*]] = fir.convert %[[VAL_58]] : (i32) -> index
! CHECK:           fir.do_loop %[[VAL_64:.*]] = %[[VAL_50]] to %[[VAL_63]] step %[[VAL_62]] {
! CHECK:             %[[VAL_65:.*]] = fir.convert %[[VAL_46]] : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<?x!fir.char<4>>>
! CHECK:             %[[VAL_66:.*]] = fir.coordinate_of %[[VAL_65]], %[[VAL_64]] : (!fir.ref<!fir.array<?x!fir.char<4>>>, index) -> !fir.ref<!fir.char<4>>
! CHECK:             fir.store %[[VAL_61]] to %[[VAL_66]] : !fir.ref<!fir.char<4>>
! CHECK:           }
! CHECK:           %[[VAL_67:.*]] = fir.array_amend %[[VAL_32]], %[[VAL_46]] : (!fir.array<70x!fir.char<4,?>>, !fir.ref<!fir.char<4,?>>) -> !fir.array<70x!fir.char<4,?>>
! CHECK:           fir.result %[[VAL_67]] : !fir.array<70x!fir.char<4,?>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_15]], %[[VAL_68:.*]] to %[[VAL_12]] typeparams %[[VAL_11]] : !fir.array<70x!fir.char<4,?>>, !fir.array<70x!fir.char<4,?>>, !fir.ref<!fir.array<70x!fir.char<4,?>>>, i32
! CHECK:         return
! CHECK:       }

subroutine test19g(a,b,i)
  character(kind=4,len=i) a(70)
  character(kind=2,len=13) b(140)
  a = b(1:140:2)
end subroutine test19g

! CHECK-LABEL: func @_QPtest19h(
! CHECK-SAME:       %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}, %[[VAL_1:.*]]: !fir.boxchar<1>{{.*}}, %[[VAL_2:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_3:.*]]: !fir.ref<i32>{{.*}}) {
! CHECK:         %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_5:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_6:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_7:.*]] = arith.cmpi sgt, %[[VAL_5]], %[[VAL_6]] : i32
! CHECK:         %[[VAL_8:.*]] = arith.select %[[VAL_7]], %[[VAL_5]], %[[VAL_6]] : i32
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<70x!fir.char<1,?>>>
! CHECK:         %[[VAL_10:.*]] = arith.constant 70 : index
! CHECK:         %[[VAL_11:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_13:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> i64
! CHECK:         %[[VAL_15A:.*]] = fir.convert %[[VAL_14]] : (i64) -> index
! CHECK:         %[[C0:.*]] = arith.constant 0 : index 
! CHECK:         %[[CMP:.*]] = arith.cmpi sgt, %[[VAL_15A]], %[[C0]] : index 
! CHECK:         %[[VAL_15:.*]] = arith.select %[[CMP]], %[[VAL_15A]], %[[C0]] : index 
! CHECK:         %[[VAL_16:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_17:.*]] = fir.array_load %[[VAL_9]](%[[VAL_16]]) typeparams %[[VAL_8]] : (!fir.ref<!fir.array<70x!fir.char<1,?>>>, !fir.shape<1>, i32) -> !fir.array<70x!fir.char<1,?>>
! CHECK:         %[[VAL_18:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i64) -> index
! CHECK:         %[[VAL_20:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
! CHECK:         %[[VAL_22:.*]] = arith.constant 140 : i64
! CHECK:         %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
! CHECK:         %[[VAL_24:.*]] = fir.shape %[[VAL_15]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_25:.*]] = fir.slice %[[VAL_19]], %[[VAL_23]], %[[VAL_21]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_26:.*]] = fir.array_load %[[VAL_12]](%[[VAL_24]]) {{\[}}%[[VAL_25]]] typeparams %[[VAL_11]]#1 : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.array<?x!fir.char<1,?>>
! CHECK:         %[[VAL_27:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_28:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_29:.*]] = arith.subi %[[VAL_10]], %[[VAL_27]] : index
! CHECK:         %[[VAL_30:.*]] = fir.do_loop %[[VAL_31:.*]] = %[[VAL_28]] to %[[VAL_29]] step %[[VAL_27]] unordered iter_args(%[[VAL_32:.*]] = %[[VAL_17]]) -> (!fir.array<70x!fir.char<1,?>>) {
! CHECK:           %[[VAL_33:.*]] = fir.array_access %[[VAL_26]], %[[VAL_31]] typeparams %[[VAL_11]]#1 : (!fir.array<?x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_34:.*]] = fir.array_access %[[VAL_32]], %[[VAL_31]] typeparams %[[VAL_8]] : (!fir.array<70x!fir.char<1,?>>, index, i32) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
! CHECK:           %[[VAL_36:.*]] = arith.cmpi slt, %[[VAL_35]], %[[VAL_11]]#1 : index
! CHECK:           %[[VAL_37:.*]] = arith.select %[[VAL_36]], %[[VAL_35]], %[[VAL_11]]#1 : index
! CHECK:           %[[VAL_38:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_39:.*]] = fir.convert %[[VAL_37]] : (index) -> i64
! CHECK:           %[[VAL_40:.*]] = arith.muli %[[VAL_38]], %[[VAL_39]] : i64
! CHECK:           %[[VAL_41:.*]] = arith.constant false
! CHECK:           %[[VAL_42:.*]] = fir.convert %[[VAL_34]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_43:.*]] = fir.convert %[[VAL_33]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:           fir.call @llvm.memmove.p0.p0.i64(%[[VAL_42]], %[[VAL_43]], %[[VAL_40]], %[[VAL_41]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:           %[[VAL_44:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_45:.*]] = arith.subi %[[VAL_8]], %[[VAL_44]] : i32
! CHECK:           %[[VAL_46:.*]] = arith.constant 32 : i8
! CHECK:           %[[VAL_47:.*]] = fir.undefined !fir.char<1>
! CHECK:           %[[VAL_48:.*]] = fir.insert_value %[[VAL_47]], %[[VAL_46]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:           %[[VAL_49:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_50:.*]] = fir.convert %[[VAL_45]] : (i32) -> index
! CHECK:           fir.do_loop %[[VAL_51:.*]] = %[[VAL_37]] to %[[VAL_50]] step %[[VAL_49]] {
! CHECK:             %[[VAL_52:.*]] = fir.convert %[[VAL_34]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:             %[[VAL_53:.*]] = fir.coordinate_of %[[VAL_52]], %[[VAL_51]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:             fir.store %[[VAL_48]] to %[[VAL_53]] : !fir.ref<!fir.char<1>>
! CHECK:           }
! CHECK:           %[[VAL_54:.*]] = fir.array_amend %[[VAL_32]], %[[VAL_34]] : (!fir.array<70x!fir.char<1,?>>, !fir.ref<!fir.char<1,?>>) -> !fir.array<70x!fir.char<1,?>>
! CHECK:           fir.result %[[VAL_54]] : !fir.array<70x!fir.char<1,?>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_17]], %[[VAL_55:.*]] to %[[VAL_9]] typeparams %[[VAL_8]] : !fir.array<70x!fir.char<1,?>>, !fir.array<70x!fir.char<1,?>>, !fir.ref<!fir.array<70x!fir.char<1,?>>>, i32
! CHECK:         return
! CHECK:       }

subroutine test19h(a,b,i,j)
  character(i) a(70)
  character(*) b(j)
  a = b(1:140:2)
end subroutine test19h

! CHECK-LABEL: func @_QPtest_elemental_character_intrinsic(
! CHECK-SAME:      %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}, %[[VAL_1:.*]]: !fir.boxchar<1>{{.*}}) {
! CHECK:         %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,?>>>
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,?>>>
! CHECK:         %[[VAL_7:.*]] = arith.constant 2 : index
! CHECK:         %[[VAL_8:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_9:.*]] = arith.constant -1 : i32
! CHECK:         %[[VAL_10:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_12:.*]] = arith.constant {{.*}} : i32
! CHECK:         %[[VAL_13:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_9]], %[[VAL_11]], %[[VAL_12]]) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_15:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_16:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_17:.*]] = fir.shape_shift %[[VAL_7]], %[[VAL_8]] : (index, index) -> !fir.shapeshift<1>
! CHECK:         %[[VAL_18:.*]] = fir.allocmem !fir.array<10xi32>
! CHECK:         %[[VAL_19:.*]] = fir.shape %[[VAL_15]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_20:.*]] = fir.array_load %[[VAL_18]](%[[VAL_19]]) : (!fir.heap<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_21:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_22:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_23:.*]] = arith.subi %[[VAL_15]], %[[VAL_21]] : index
! CHECK:         %[[VAL_24:.*]] = fir.do_loop %[[VAL_25:.*]] = %[[VAL_22]] to %[[VAL_23]] step %[[VAL_21]] unordered iter_args(%[[VAL_26:.*]] = %[[VAL_20]]) -> (!fir.array<10xi32>) {
! CHECK:           %[[VAL_27:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_28:.*]] = arith.addi %[[VAL_25]], %[[VAL_27]] : index
! CHECK:           %[[VAL_29:.*]] = fir.array_coor %[[VAL_3]](%[[VAL_16]]) %[[VAL_28]] typeparams %[[VAL_2]]#1 : (!fir.ref<!fir.array<10x!fir.char<1,?>>>, !fir.shape<1>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_30:.*]] = arith.addi %[[VAL_25]], %[[VAL_7]] : index
! CHECK:           %[[VAL_31:.*]] = fir.array_coor %[[VAL_6]](%[[VAL_17]]) %[[VAL_30]] typeparams %[[VAL_5]]#1 : (!fir.ref<!fir.array<10x!fir.char<1,?>>>, !fir.shapeshift<1>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_32:.*]] = arith.constant false
! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_29]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_2]]#1 : (index) -> i64
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_31]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_5]]#1 : (index) -> i64
! CHECK:           %[[VAL_37:.*]] = fir.call @_FortranAScan1(%[[VAL_33]], %[[VAL_34]], %[[VAL_35]], %[[VAL_36]], %[[VAL_32]]) : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i64) -> i32
! CHECK:           %[[VAL_39:.*]] = fir.array_update %[[VAL_26]], %[[VAL_38]], %[[VAL_25]] : (!fir.array<10xi32>, i32, index) -> !fir.array<10xi32>
! CHECK:           fir.result %[[VAL_39]] : !fir.array<10xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_20]], %[[VAL_40:.*]] to %[[VAL_18]] : !fir.array<10xi32>, !fir.array<10xi32>, !fir.heap<!fir.array<10xi32>>
! CHECK:         %[[VAL_41:.*]] = fir.shape %[[VAL_15]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_42:.*]] = fir.embox %[[VAL_18]](%[[VAL_41]]) : (!fir.heap<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
! CHECK:         %[[VAL_43:.*]] = fir.convert %[[VAL_42]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
! CHECK:         %[[VAL_44:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_13]], %[[VAL_43]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:         fir.freemem %[[VAL_18]]
! CHECK:         %[[VAL_45:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_13]]) : (!fir.ref<i8>) -> i32
! CHECK:         return
! CHECK:       }

subroutine test_elemental_character_intrinsic(c1, c2)
  character(*) :: c1(10), c2(2:11)
  print *, scan(c1, c2)
end subroutine

! CHECK: func private @_QPbar(
