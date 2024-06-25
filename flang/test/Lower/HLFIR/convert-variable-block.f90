! Test that hlfir.declare is not created again for dummy arguments
! used in specifications of BLOCK variables.
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

subroutine test(n)
  integer(8) :: n
  call before_block()
  block
    real :: x(n)
    call foo(x)
  end block
end subroutine
! CHECK-LABEL:   func.func @_QPtest(
! CHECK-SAME:                       %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtestEn"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           fir.call @_QPbefore_block() {{.*}}: () -> ()
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_6:.*]] = arith.cmpi sgt, %[[VAL_4]], %[[VAL_5]] : index
! CHECK:           %[[VAL_7:.*]] = arith.select %[[VAL_6]], %[[VAL_4]], %[[VAL_5]] : index
! CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.array<?xf32>, %[[VAL_7]] {bindc_name = "x", uniq_name = "_QFtestB1Ex"}
! CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_8]](%[[VAL_9]]) {uniq_name = "_QFtestB1Ex"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:           fir.call @_QPfoo(%[[VAL_10]]#1) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()
