! Test SIZEOF lowering for polymorphic entities.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

integer(8) function test1(x)
  class(*) :: x
  test1 = sizeof(x)
end function
! CHECK-LABEL:   func.func @_QPtest1(
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest1Ex"} : (!fir.class<none>, !fir.dscope) -> (!fir.class<none>, !fir.class<none>)
! CHECK:           %[[VAL_4:.*]] = fir.box_elesize %[[VAL_3]]#1 : (!fir.class<none>) -> i64
! CHECK:           hlfir.assign %[[VAL_4]] to %{{.*}} : i64, !fir.ref<i64>

integer(8) function test2(x)
  class(*) :: x(:, :)
  test2 = sizeof(x)
end function
! CHECK-LABEL:   func.func @_QPtest2(
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest2Ex"} : (!fir.class<!fir.array<?x?xnone>>, !fir.dscope) -> (!fir.class<!fir.array<?x?xnone>>, !fir.class<!fir.array<?x?xnone>>)
! CHECK:           %[[VAL_4:.*]] = fir.box_elesize %[[VAL_3]]#1 : (!fir.class<!fir.array<?x?xnone>>) -> i64
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.class<!fir.array<?x?xnone>>) -> !fir.box<none>
! CHECK:           %[[VAL_9:.*]] = fir.call @_FortranASize(%[[VAL_7]], %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> i64
! CHECK:           %[[VAL_10:.*]] = arith.muli %[[VAL_4]], %[[VAL_9]] : i64
! CHECK:           hlfir.assign %[[VAL_10]] to %{{.*}} : i64, !fir.ref<i64>
