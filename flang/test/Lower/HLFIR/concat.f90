! Test lowering of character concatenation to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

subroutine concat(c1, c2, c3)
  character(*) :: c1, c2, c3
  c1 = c2 // c3
end subroutine
! CHECK-LABEL: func.func @_QPconcat
! CHECK:  hlfir.declare {{.*}}c1
! CHECK:  %[[VAL_5:.*]]:2 = fir.unboxchar %{{.*}} : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare {{.*}}c2
! CHECK:  %[[VAL_7:.*]]:2 = fir.unboxchar %{{.*}} : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_8:.*]]:2 = hlfir.declare {{.*}}c3
! CHECK:  %[[VAL_9:.*]] = arith.addi %[[VAL_5]]#1, %[[VAL_7]]#1 : index
! CHECK:  %[[VAL_10:.*]] = hlfir.concat %[[VAL_6]]#0, %[[VAL_8]]#0 len %[[VAL_9]] : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>

subroutine concat_2(c1, c2, c3)
  character(*) :: c1(100)
  character :: c2(100)*10, c3(100)*20
  c1(1) = c2(1) // c3(1)
end subroutine
! CHECK-LABEL: func.func @_QPconcat_2
! CHECK:  %[[VAL_9:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_13:.*]]:2 = hlfir.declare %{{.*}}c2
! CHECK:  %[[VAL_15:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_19:.*]]:2 = hlfir.declare {{.*}}c3
! CHECK:  %[[VAL_21:.*]] = hlfir.designate %[[VAL_13]]#0 (%{{.*}})  typeparams %[[VAL_9]] : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
! CHECK:  %[[VAL_23:.*]] = hlfir.designate %[[VAL_19]]#0 (%{{.*}})  typeparams %[[VAL_15]] : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, index, index) -> !fir.ref<!fir.char<1,20>>
! CHECK:  %[[VAL_24:.*]] = arith.addi %[[VAL_9]], %[[VAL_15]] : index
! CHECK:  %[[VAL_25:.*]] = hlfir.concat %[[VAL_21]], %[[VAL_23]] len %[[VAL_24]] : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>, index) -> !hlfir.expr<!fir.char<1,30>>

subroutine concat3(c1, c2, c3, c4)
  character(*) :: c1, c2, c3, c4
  c1 = c2 // c3 // c4
end subroutine
! CHECK-LABEL: func.func @_QPconcat3
! CHECK:  hlfir.declare {{.*}}c1
! CHECK:  %[[VAL_5:.*]]:2 = fir.unboxchar %{{.*}}
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare {{.*}}c2
! CHECK:  %[[VAL_7:.*]]:2 = fir.unboxchar %{{.*}}
! CHECK:  %[[VAL_8:.*]]:2 = hlfir.declare {{.*}}c3
! CHECK:  %[[VAL_9:.*]]:2 = fir.unboxchar %{{.*}}
! CHECK:  %[[VAL_10:.*]]:2 = hlfir.declare {{.*}}c4
! CHECK:  %[[VAL_11:.*]] = arith.addi %[[VAL_5]]#1, %[[VAL_7]]#1 : index
! CHECK:  %[[VAL_12:.*]] = hlfir.concat %[[VAL_6]]#0, %[[VAL_8]]#0 len %[[VAL_11]] : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:  %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_9]]#1 : index
! CHECK:  %[[VAL_14:.*]] = hlfir.concat %[[VAL_12]], %[[VAL_10]]#0 len %[[VAL_13]] : (!hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
