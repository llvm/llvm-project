! Test lowering of the max and min intrinsics on character types to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

subroutine max1(c1, c2, c3)
  character(*) :: c1, c2, c3
  c1 = max(c2,c3)
end subroutine

! CHECK-LABEL: func.func @_QPmax1
! CHECK:  %[[VAL_0:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %{{.*}} : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_1:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_0]]#0 typeparams %[[VAL_0]]#1 {{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_2:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %{{.*}} : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_3:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 {{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_4:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %{{.*}} : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_5:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_4]]#1 {{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_6:[a-zA-Z0-9_]*]] = hlfir.char_extremum max, %[[VAL_3]]#0, %[[VAL_5]]#0 : (!fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:  hlfir.assign %[[VAL_6]] to %[[VAL_1]]#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
! CHECK:  hlfir.destroy %[[VAL_6]] : !hlfir.expr<!fir.char<1,?>>

subroutine min1(c1, c2, c3)
  character(*) :: c1, c2, c3
  c1 = min(c2,c3)
end subroutine
! CHECK-LABEL: func.func @_QPmin1
! CHECK:  %[[VAL_0:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %{{.*}} : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_1:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_0]]#0 typeparams %[[VAL_0]]#1 {{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_2:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %{{.*}} : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_3:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 {{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_4:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %{{.*}} : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_5:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_4]]#1 {{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_6:[a-zA-Z0-9_]*]] = hlfir.char_extremum min, %[[VAL_3]]#0, %[[VAL_5]]#0 : (!fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:  hlfir.assign %[[VAL_6]] to %[[VAL_1]]#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
! CHECK:  hlfir.destroy %[[VAL_6]] : !hlfir.expr<!fir.char<1,?>>
! CHECK:  return

subroutine max2(c1, c2, c3)
  character(*) :: c1(100)
  character :: c2(100)*10, c3(100)*20
  c1(1) = max(c2(1),c3(1))
end subroutine
! CHECK-LABEL:  func.func @_QPmax2
! CHECK:  %[[VAL_0:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %{{.*}} : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_1:[a-zA-Z0-9_]*]] = fir.convert %[[VAL_0]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,?>>>
! CHECK:  %[[VAL_C100:[a-zA-Z0-9_]*]] = arith.constant 100 : index
! CHECK:  %[[VAL_2:[a-zA-Z0-9_]*]]  = fir.shape %[[VAL_C100]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_3:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_2]]) typeparams %[[VAL_0]]#1 {{.*}} : (!fir.ref<!fir.array<100x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<100x!fir.char<1,?>>>, !fir.ref<!fir.array<100x!fir.char<1,?>>>)
! CHECK:  %[[VAL_4:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %{{.*}} : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_5:[a-zA-Z0-9_]*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,10>>>
! CHECK:  %[[VAL_C10:[a-zA-Z0-9_]*]] = arith.constant 10 : index
! CHECK:  %[[VAL_C100_0:[a-zA-Z0-9_]*]] = arith.constant 100 : index
! CHECK:  %[[VAL_6:[a-zA-Z0-9_]*]] = fir.shape %[[VAL_C100_0]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_7:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_6]]) typeparams %[[VAL_C10]] {{.*}} : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.ref<!fir.array<100x!fir.char<1,10>>>)
! CHECK:  %[[VAL_8:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %{{.*}} : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_9:[a-zA-Z0-9_]*]] = fir.convert %[[VAL_8]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,20>>>
! CHECK:  %[[VAL_C20:[a-zA-Z0-9_]*]] = arith.constant 20 : index
! CHECK:  %[[VAL_C100_1:[a-zA-Z0-9_]*]] = arith.constant 100 : index
! CHECK:  %[[VAL_10:[a-zA-Z0-9_]*]] = fir.shape %[[VAL_C100_1]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_11:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_9]](%[[VAL_10]]) typeparams %[[VAL_C20]] {{.*}} : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.ref<!fir.array<100x!fir.char<1,20>>>)
! CHECK:  %[[VAL_C1:[a-zA-Z0-9_]*]] = arith.constant 1 : index
! CHECK:  %[[VAL_12:[a-zA-Z0-9_]*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_C1]])  typeparams %[[VAL_C10]] : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
! CHECK:  %[[VAL_C1_2:[a-zA-Z0-9_]*]] = arith.constant 1 : index
! CHECK:  %[[VAL_13:[a-zA-Z0-9_]*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_C1_2]])  typeparams %[[VAL_C20]] : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, index, index) -> !fir.ref<!fir.char<1,20>>
! CHECK:  %[[VAL_14:[a-zA-Z0-9_]*]] = hlfir.char_extremum max, %[[VAL_12]], %[[VAL_13]] : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>) -> !hlfir.expr<!fir.char<1,20>>
! CHECK:  %c1_3 = arith.constant 1 : index
! CHECK:  %[[VAL_15:[a-zA-Z0-9_]*]] = hlfir.designate %[[VAL_3]]#0 (%c1_3)  typeparams %[[VAL_0]]#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:  hlfir.assign %[[VAL_14]] to %[[VAL_15]] : !hlfir.expr<!fir.char<1,20>>, !fir.boxchar<1>
! CHECK:  hlfir.destroy %[[VAL_14]] : !hlfir.expr<!fir.char<1,20>>

subroutine min2(c1, c2, c3)
  character(*) :: c1(100)
  character :: c2(100)*10, c3(100)*20
  c1(1) = min(c2(1),c3(1))
end subroutine
! CHECK-LABEL: func.func @_QPmin2
! CHECK:  %[[VAL_0:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_1:[a-zA-Z0-9_]*]] = fir.convert %[[VAL_0]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,?>>>
! CHECK:  %[[VAL_C100:[a-zA-Z0-9_]*]] = arith.constant 100 : index
! CHECK:  %[[VAL_2:[a-zA-Z0-9_]*]] = fir.shape %[[VAL_C100]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_3:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_2]]) typeparams %[[VAL_0]]#1 {{.*}} : (!fir.ref<!fir.array<100x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<100x!fir.char<1,?>>>, !fir.ref<!fir.array<100x!fir.char<1,?>>>)
! CHECK:  %[[VAL_4:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_5:[a-zA-Z0-9_]*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,10>>>
! CHECK:  %[[VAL_C10:[a-zA-Z0-9_]*]] = arith.constant 10 : index
! CHECK:  %[[VAL_C100_0:[a-zA-Z0-9_]*]] = arith.constant 100 : index
! CHECK:  %[[VAL_6:[a-zA-Z0-9_]*]] = fir.shape %[[VAL_C100_0]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_7:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_6]]) typeparams %[[VAL_C10]] {{.*}} : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,10>>>, !fir.ref<!fir.array<100x!fir.char<1,10>>>)
! CHECK:  %[[VAL_8:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_C9:[a-zA-Z0-9_]*]] = fir.convert %[[VAL_8]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<100x!fir.char<1,20>>>
! CHECK:  %[[VAL_C20:[a-zA-Z0-9_]*]] = arith.constant 20 : index
! CHECK:  %[[VAL_C100_1:[a-zA-Z0-9_]*]] = arith.constant 100 : index
! CHECK:  %[[VAL_10:[a-zA-Z0-9_]*]] = fir.shape %[[VAL_C100_1]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_11:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_C9]](%[[VAL_10]]) typeparams %[[VAL_C20]] {{.*}} : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.ref<!fir.array<100x!fir.char<1,20>>>)
! CHECK:  %[[VAL_C1:[a-zA-Z0-9_]*]] = arith.constant 1 : index
! CHECK:  %[[VAL_12:[a-zA-Z0-9_]*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_C1]])  typeparams %[[VAL_C10]] : (!fir.ref<!fir.array<100x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
! CHECK:  %[[VAL_C1_2:[a-zA-Z0-9_]*]] = arith.constant 1 : index
! CHECK:  %[[VAL_13:[a-zA-Z0-9_]*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_C1_2]])  typeparams %[[VAL_C20]] : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, index, index) -> !fir.ref<!fir.char<1,20>>
! CHECK:  %[[VAL_14:[a-zA-Z0-9_]*]] = hlfir.char_extremum min, %[[VAL_12]], %[[VAL_13]] : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>) -> !hlfir.expr<!fir.char<1,20>>
! CHECK:  %[[VAL_C1_3:[a-zA-Z0-9_]*]] = arith.constant 1 : index
! CHECK:  %[[VAL_15:[a-zA-Z0-9_]*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_C1_3]])  typeparams %[[VAL_0]]#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:  hlfir.assign %[[VAL_14]] to %[[VAL_15]] : !hlfir.expr<!fir.char<1,20>>, !fir.boxchar<1>
! CHECK:  hlfir.destroy %[[VAL_14]] : !hlfir.expr<!fir.char<1,20>>

subroutine max3(c1, c2, c3, c4)
  character(*) :: c1, c2, c3, c4
  c1 = max(c2, c3, c4)
end subroutine
! CHECK-LABEL: func.func @_QPmax3
! CHECK:  %[[VAL_0:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_1:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_0]]#0 typeparams %[[VAL_0]]#1 {{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_2:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_3:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 {{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_4:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_5:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_4]]#1 {{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_6:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %arg3 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_7:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 {{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_8:[a-zA-Z0-9_]*]] = hlfir.char_extremum max, %[[VAL_3]]#0, %[[VAL_5]]#0, %[[VAL_7]]#0 : (!fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:  hlfir.assign %[[VAL_8]] to %[[VAL_1]]#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
! CHECK:  hlfir.destroy %[[VAL_8]] : !hlfir.expr<!fir.char<1,?>>

subroutine min3(c1, c2, c3, c4)
  character(*) :: c1, c2, c3, c4
  c1 = min(c2, c3, c4)
end subroutine
! CHECK-LABEL: func.func @_QPmin3
! CHECK:  %[[VAL_0:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_1:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_0]]#0 typeparams %[[VAL_0]]#1 {{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_2:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_3:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 {{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_4:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_5:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_4]]#1 {{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_6:[a-zA-Z0-9_]*]]:2 = fir.unboxchar %arg3 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_7:[a-zA-Z0-9_]*]]:2 = hlfir.declare %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 {{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_8:[a-zA-Z0-9_]*]] = hlfir.char_extremum min, %[[VAL_3]]#0, %[[VAL_5]]#0, %[[VAL_7]]#0 : (!fir.boxchar<1>, !fir.boxchar<1>, !fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:  hlfir.assign %[[VAL_8]] to %[[VAL_1]]#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
! CHECK:  hlfir.destroy %[[VAL_8]] : !hlfir.expr<!fir.char<1,?>>
