! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPreshape_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:[^:]+]]: !fir.box<!fir.array<?x?x?xi32>>{{.*}}, %[[arg2:[^:]+]]: !fir.box<!fir.array<?x?x?xi32>>{{.*}}, %[[arg3:.*]]: !fir.ref<!fir.array<2xi32>>{{.*}}, %[[arg4:.*]]: !fir.ref<!fir.array<2xi32>>{{.*}}) {
subroutine reshape_test(x, source, pd, sh, ord)
    integer :: x(:,:)
    integer :: source(:,:,:)
    integer :: pd(:,:,:)
    integer :: sh(2)
    integer :: ord(2)
  ! CHECK-DAG:  %[[c2:.*]] = arith.constant 2 : index
  ! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>>
  ! CHECK-DAG:  %[[a1:.*]] = fir.shape %[[c2]] : (index) -> !fir.shape<1>
  ! CHECK-DAG:  %[[a2:.*]] = fir.embox %[[arg3]](%{{.*}}) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
  ! CHECK-DAG:  %[[a3:.*]] = fir.embox %[[arg4]](%{{.*}}) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
  ! CHECK-DAG:  %[[a8:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG:  %[[a9:.*]] = fir.convert %[[arg1]] : (!fir.box<!fir.array<?x?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG:  %[[a10:.*]] = fir.convert %[[a2]] : (!fir.box<!fir.array<2xi32>>) -> !fir.box<none>
  ! CHECK-DAG:  %[[a11:.*]] = fir.convert %[[arg2]] : (!fir.box<!fir.array<?x?x?xi32>>) -> !fir.box<none>
  ! CHECK-DAG:  %[[a12:.*]] = fir.convert %[[a3]] : (!fir.box<!fir.array<2xi32>>) -> !fir.box<none>
    x = reshape(source, sh, pd, ord)
  ! CHECK:  %{{.*}} = fir.call @_FortranAReshape(%[[a8]], %[[a9]], %[[a10]], %[[a11]], %[[a12]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK-DAG:  %[[a15:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  ! CHECK-DAG:  %[[a18:.*]] = fir.box_addr %[[a15]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !fir.heap<!fir.array<?x?xi32>>
  ! CHECK-DAG:  fir.freemem %[[a18]]
  end subroutine
  
  ! CHECK-LABEL: func @_QPtest_reshape_optional(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  subroutine test_reshape_optional(pad, order, source, shape)
    real, pointer :: pad(:, :) 
    integer, pointer :: order(:) 
    real :: source(:, :, :)
    integer :: shape(4) 
    print *, reshape(source=source, shape=shape, pad=pad, order=order)  
  ! CHECK:  %[[VAL_13:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  ! CHECK:  %[[VAL_14:.*]] = fir.box_addr %[[VAL_13]] : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>) -> !fir.ptr<!fir.array<?x?xf32>>
  ! CHECK:  %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (!fir.ptr<!fir.array<?x?xf32>>) -> i64
  ! CHECK:  %[[VAL_16:.*]] = arith.constant 0 : i64
  ! CHECK:  %[[VAL_17:.*]] = arith.cmpi ne, %[[VAL_15]], %[[VAL_16]] : i64
  ! CHECK:  %[[VAL_18:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  ! CHECK:  %[[VAL_19:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
  ! CHECK:  %[[VAL_20:.*]] = arith.select %[[VAL_17]], %[[VAL_18]], %[[VAL_19]] : !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
  ! CHECK:  %[[VAL_21:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  ! CHECK:  %[[VAL_22:.*]] = fir.box_addr %[[VAL_21]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
  ! CHECK:  %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
  ! CHECK:  %[[VAL_24:.*]] = arith.constant 0 : i64
  ! CHECK:  %[[VAL_25:.*]] = arith.cmpi ne, %[[VAL_23]], %[[VAL_24]] : i64
  ! CHECK:  %[[VAL_26:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  ! CHECK:  %[[VAL_27:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?xi32>>>
  ! CHECK:  %[[VAL_28:.*]] = arith.select %[[VAL_25]], %[[VAL_26]], %[[VAL_27]] : !fir.box<!fir.ptr<!fir.array<?xi32>>>
  ! CHECK:  %[[VAL_38:.*]] = fir.convert %[[VAL_20]] : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>) -> !fir.box<none>
  ! CHECK:  %[[VAL_39:.*]] = fir.convert %[[VAL_28]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.box<none>
  ! CHECK:  %[[VAL_41:.*]] = fir.call @_FortranAReshape({{.*}}, {{.*}}, %{{.*}}, %[[VAL_38]], %[[VAL_39]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  end subroutine
