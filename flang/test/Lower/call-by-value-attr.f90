! Test for PassBy::BaseAddressValueAttribute
! RUN: bbc -emit-fir %s -o - | FileCheck %s
program call_by_value_attr
  interface
     subroutine subri(val)
       integer, value :: val
     end subroutine subri
     subroutine subra(val)
       integer, value, dimension(10) :: val
     end subroutine subra
  end interface

!CHECK-LABEL: func @_QQmain()
  integer :: v
  integer, dimension(10) :: a
  integer, dimension(15) :: b
  v = 17
  call subri(v)
  !CHECK: %[[COPY:.*]] = fir.alloca i32
  !CHECK: %[[ARRAY_A:.*]] = fir.address_of(@_QFEa)
  !CHECK: %[[CONST_10_1:.*]] = arith.constant 10 : index
  !CHECK: %[[ARRAY_B:.*]] = fir.address_of(@_QFEb)
  !CHECK: %[[CONST_15_1:.*]] = arith.constant 15 : index
  !CHECK: %[[VALUE:.*]] = fir.alloca i32 {bindc_name = "v", {{.*}}}
  !CHECK: %[[CONST:.*]] = arith.constant 17
  !CHECK: fir.store %[[CONST]] to %[[VALUE]]
  !CHECK: %[[LOAD:.*]] = fir.load %[[VALUE]]
  !CHECK: fir.store %[[LOAD]] to %[[COPY]]
  !CHECK: fir.call @_QPsubri(%[[COPY]]) : {{.*}}
  a = (/ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 /)
  !CHECK: %[[SHAPE_1:.*]] = fir.shape %[[CONST_10_1]]
  !CHECK: %[[ARRAY_LOAD_1:.*]] = fir.array_load %[[ARRAY_A]](%[[SHAPE_1]]) : {{.*}}
  !CHECK: %[[ARRAY_INIT_A:.*]] = fir.address_of({{.*}})
  !CHECK: %[[CONST_10_2:.*]] = arith.constant 10 : index
  !CHECK: %[[SHAPE_2:.*]] = fir.shape %[[CONST_10_2]]
  !CHECK: %[[ARRAY_LOAD_2:.*]] = fir.array_load %[[ARRAY_INIT_A]](%[[SHAPE_2]]) : {{.*}}
  !CHECK: %[[DO_1:.*]] = fir.do_loop {{.*}} {
  !CHECK: }
  !CHECK: fir.array_merge_store %[[ARRAY_LOAD_1]], %[[DO_1]] to %[[ARRAY_A]]
  !CHECK: %[[ARRAY_COPY:.*]] = fir.allocmem !fir.array<10xi32>, %[[CONST_10_1]] {uniq_name = ".copy"}
  !CHECK: %[[SHAPE_3:.*]] = fir.shape %[[CONST_10_1]]
  !CHECK: %[[ARRAY_LOAD_3:.*]] = fir.array_load %[[ARRAY_COPY]](%[[SHAPE_3]]) : {{.*}}
  !CHECK: %[[SHAPE_4:.*]] = fir.shape %[[CONST_10_1]]
  !CHECK: %[[ARRAY_LOAD_4:.*]] = fir.array_load %[[ARRAY_A]](%[[SHAPE_4]]) : {{.*}}
  !CHECK: %[[DO_2:.*]] = fir.do_loop {{.*}} {
  !CHECK: }
  !CHECK: fir.array_merge_store %[[ARRAY_LOAD_3]], %[[DO_2]] to %[[ARRAY_COPY]]
  !CHECK: %[[CONVERT:.*]] = fir.convert %[[ARRAY_COPY]] : (!fir.heap<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  !CHECK: fir.call @_QPsubra(%[[CONVERT]])
  call subra(a)
  b = (/ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 /)
  !CHECK: %[[SHAPE_5:.*]] = fir.shape %[[CONST_15_1]]
  !CHECK: %[[ARRAY_LOAD_5:.*]] = fir.array_load %[[ARRAY_B]](%[[SHAPE_5]]) : {{.*}}
  !CHECK: %[[ARRAY_INIT_B:.*]] = fir.address_of({{.*}})
  !CHECK: %[[CONST_15_2:.*]] = arith.constant 15 : index
  !CHECK: %[[SHAPE_6:.*]] = fir.shape %[[CONST_15_2]] : (index) -> !fir.shape<1>
  !CHECK: %[[ARRAY_LOAD_6:.*]] = fir.array_load %[[ARRAY_INIT_B]](%[[SHAPE_6]]) : (!fir.ref<!fir.array<15xi32>>, !fir.shape<1>) -> !fir.array<15xi32>
  !CHECK: %[[DO_3:.*]] = fir.do_loop {{.*}} {
  !CHECK: }
  !CHECK: fir.array_merge_store %[[ARRAY_LOAD_5]], %[[DO_3]] to %[[ARRAY_B]]
  !CHECK: %[[CONST_5:.*]] = arith.constant 5 : i64
  !CHECK: %[[CONV_5:.*]] = fir.convert %[[CONST_5]] : (i64) -> index
  !CHECK: %[[CONST_1:.*]] = arith.constant 1 : i64
  !CHECK: %[[CONV_1:.*]] = fir.convert %[[CONST_1]] : (i64) -> index
  !CHECK: %[[CONST_15_3:.*]] = arith.constant 15 : i64
  !CHECK: %[[CONV_15:.*]] = fir.convert %[[CONST_15_3]] : (i64) -> index
  !CHECK: %[[SHAPE_7:.*]]  = fir.shape %[[CONST_15_1]] : (index) -> !fir.shape<1>
  !CHECK: %[[SLICE:.*]] = fir.slice %[[CONV_5]], %[[CONV_15]], %[[CONV_1]] : (index, index, index) -> !fir.slice<1>
  !CHECK: %[[BOX:.*]] = fir.embox %[[ARRAY_B]](%[[SHAPE_7]]) [%[[SLICE]]] : (!fir.ref<!fir.array<15xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
  !CHECK: %[[CONST_0:.*]] = arith.constant 0 : index
  !CHECK: %[[DIMS:.*]]:3 = fir.box_dims %[[BOX]], %[[CONST_0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  !CHECK: %[[ARRAY_COPY_2:.*]] = fir.allocmem !fir.array<?xi32>, %[[DIMS]]#1 {uniq_name = ".copy"}
  !CHECK: %[[SHAPE_8:.*]] = fir.shape %[[DIMS]]#1 : (index) -> !fir.shape<1>
  !CHECK: %[[ARRAY_LOAD_7:.*]] = fir.array_load %[[ARRAY_COPY_2]](%[[SHAPE_8]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.array<?xi32>
  !CHECK: %[[ARRAY_LOAD_8:.*]] = fir.array_load %[[BOX]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
  !CHECK: %[[DO_4:.*]] = fir.do_loop {{.*}} {
  !CHECK: }
  !CHECK fir.array_merge_store %[[ARRAY_LOAD_7]], %[[DO_4]] to %[[ARRAY_COPY_2]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.heap<!fir.array<?xi32>>
  !CHECK: %[[CONVERT_B:.*]] = fir.convert %[[ARRAY_COPY_2]] : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<10xi32>>
  !CHECK: fir.call @_QPsubra(%[[CONVERT_B]])
  call subra(b(5:15))
end program call_by_value_attr
