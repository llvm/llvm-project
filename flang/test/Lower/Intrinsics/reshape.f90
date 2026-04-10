! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPreshape_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[arg1:[^:]+]]: !fir.box<!fir.array<?x?x?xi32>>{{.*}}, %[[arg2:[^:]+]]: !fir.box<!fir.array<?x?x?xi32>>{{.*}}, %[[arg3:.*]]: !fir.ref<!fir.array<2xi32>>{{.*}}, %[[arg4:.*]]: !fir.ref<!fir.array<2xi32>>{{.*}}) {
subroutine reshape_test(x, source, pd, sh, ord)
    integer :: x(:,:)
    integer :: source(:,:,:)
    integer :: pd(:,:,:)
    integer :: sh(2)
    integer :: ord(2)
  ! CHECK-DAG: %[[ordDecl:.*]]:2 = hlfir.declare %[[arg4]]
  ! CHECK-DAG: %[[pdDecl:.*]]:2 = hlfir.declare %[[arg2]]
  ! CHECK-DAG: %[[shDecl:.*]]:2 = hlfir.declare %[[arg3]]
  ! CHECK-DAG: %[[srcDecl:.*]]:2 = hlfir.declare %[[arg1]]
  ! CHECK-DAG: %[[xDecl:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[res:.*]] = hlfir.reshape %[[srcDecl]]#0 %[[shDecl]]#0 pad %[[pdDecl]]#0 order %[[ordDecl]]#0
  ! CHECK: hlfir.assign %[[res]] to %[[xDecl]]#0
  ! CHECK: hlfir.destroy %[[res]]
    x = reshape(source, sh, pd, ord)
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
  ! CHECK-DAG:  %[[padDecl:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}pad
  ! CHECK-DAG:  %[[orderDecl:.*]]:2 = hlfir.declare %[[VAL_1]] {{.*}}order
  ! CHECK:  %[[padLoad1:.*]] = fir.load %[[padDecl]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  ! CHECK:  %[[padAddr:.*]] = fir.box_addr %[[padLoad1]] : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>) -> !fir.ptr<!fir.array<?x?xf32>>
  ! CHECK:  %[[padI64:.*]] = fir.convert %[[padAddr]] : (!fir.ptr<!fir.array<?x?xf32>>) -> i64
  ! CHECK:  %[[c0:.*]] = arith.constant 0 : i64
  ! CHECK:  %[[padNonNull:.*]] = arith.cmpi ne, %[[padI64]], %[[c0]] : i64
  ! CHECK:  %[[orderLoad1:.*]] = fir.load %[[orderDecl]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  ! CHECK:  %[[orderAddr:.*]] = fir.box_addr %[[orderLoad1]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
  ! CHECK:  %[[orderI64:.*]] = fir.convert %[[orderAddr]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
  ! CHECK:  %[[c0_2:.*]] = arith.constant 0 : i64
  ! CHECK:  %[[orderNonNull:.*]] = arith.cmpi ne, %[[orderI64]], %[[c0_2]] : i64
  ! CHECK:  %[[padLoad2:.*]] = fir.load %[[padDecl]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  ! CHECK:  %[[padAbsent:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
  ! CHECK:  %[[padOpt:.*]] = arith.select %[[padNonNull]], %[[padLoad2]], %[[padAbsent]] : !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
  ! CHECK:  %[[orderLoad2:.*]] = fir.load %[[orderDecl]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  ! CHECK:  %[[orderAbsent:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?xi32>>>
  ! CHECK:  %[[orderOpt:.*]] = arith.select %[[orderNonNull]], %[[orderLoad2]], %[[orderAbsent]] : !fir.box<!fir.ptr<!fir.array<?xi32>>>
  ! CHECK:  hlfir.reshape {{.*}} pad %[[padOpt]] order %[[orderOpt]]
  end subroutine

! CHECK-LABEL: func.func @_QPtest_reshape_shape_slice() {
subroutine test_reshape_shape_slice()
  integer, parameter :: i = 1
  real :: tmp(4) = [1,2,3,4]
  integer ::  dims(4) = [2,2,2,2]
  ! CHECK:  %[[dimsDecl:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFtest_reshape_shape_sliceEdims"}
  ! CHECK:  %[[tmpDecl:.*]]:2 = hlfir.declare {{.*}}{uniq_name = "_QFtest_reshape_shape_sliceEtmp"}
  ! CHECK:  %[[sliceRef:.*]] = hlfir.designate %[[dimsDecl]]#0 ({{.*}}:{{.*}}:{{.*}})
  ! CHECK:  hlfir.reshape %[[tmpDecl]]#0 %[[sliceRef]]
  call some_proc(reshape(tmp, dims(i:2)))
end
