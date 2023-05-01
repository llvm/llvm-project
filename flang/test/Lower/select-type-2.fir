// RUN: fir-opt --fir-polymorphic-op %s | FileCheck %s

// Test that cf.br for the default select case passes the blocks argument
// as expected.

// Original code:
//subroutine test()
//  interface
//     function fstar()
//       class(*),pointer :: fstar
//     end function fstar
//     subroutine foo(x)
//       class(*) :: x
//     end subroutine foo
//  end interface
//
//  select type(y=>fstar())
//  type is (integer)
//    y = 777
//  class default
//     call foo(y)
//     stop 'fail type'
//  end select
//
//  select type(y=>fstar())
//  type is (integer)
//    y = 23
//  class default
//     call foo(y)
//     stop 'fail type'
//  end select
//end subroutine test

// CHECK-LABEL:   func.func @_QPtest() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 23 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant false
// CHECK:           %[[VAL_2:.*]] = arith.constant 9 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 777 : i32
// CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.class<!fir.ptr<none>> {bindc_name = ".result"}
// CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.class<!fir.ptr<none>> {bindc_name = ".result"}
// CHECK:           %[[VAL_6:.*]] = fir.call @_QPfstar() fastmath<contract> : () -> !fir.class<!fir.ptr<none>>
// CHECK:           fir.save_result %[[VAL_6]] to %[[VAL_5]] : !fir.class<!fir.ptr<none>>, !fir.ref<!fir.class<!fir.ptr<none>>>
// CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.class<!fir.ptr<none>>>
// CHECK:           %[[VAL_8:.*]] = arith.constant 9 : i8
// CHECK:           %[[VAL_9:.*]] = fir.box_typecode %[[VAL_7]] : (!fir.class<!fir.ptr<none>>) -> i8
// CHECK:           %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_8]] : i8
// CHECK:           cf.cond_br %[[VAL_10]], ^bb2, ^bb1
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb3(%[[VAL_7]] : !fir.class<!fir.ptr<none>>)
// CHECK:         ^bb2:
// CHECK:           %[[VAL_11:.*]] = fir.box_addr %[[VAL_7]] : (!fir.class<!fir.ptr<none>>) -> !fir.ptr<i32>
// CHECK:           fir.store %[[VAL_3]] to %[[VAL_11]] : !fir.ptr<i32>
// CHECK:           %[[VAL_12:.*]] = fir.call @_QPfstar() fastmath<contract> : () -> !fir.class<!fir.ptr<none>>
// CHECK:           fir.save_result %[[VAL_12]] to %[[VAL_4]] : !fir.class<!fir.ptr<none>>, !fir.ref<!fir.class<!fir.ptr<none>>>
// CHECK:           %[[VAL_13:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.class<!fir.ptr<none>>>
// CHECK:           %[[VAL_14:.*]] = arith.constant 9 : i8
// CHECK:           %[[VAL_15:.*]] = fir.box_typecode %[[VAL_13]] : (!fir.class<!fir.ptr<none>>) -> i8
// CHECK:           %[[VAL_16:.*]] = arith.cmpi eq, %[[VAL_15]], %[[VAL_14]] : i8
// CHECK:           cf.cond_br %[[VAL_16]], ^bb5, ^bb4
// CHECK:         ^bb3(%[[VAL_17:.*]]: !fir.class<!fir.ptr<none>>):
// CHECK:           %[[VAL_18:.*]] = fir.rebox %[[VAL_17]] : (!fir.class<!fir.ptr<none>>) -> !fir.class<none>
// CHECK:           fir.call @_QPfoo(%[[VAL_18]]) fastmath<contract> : (!fir.class<none>) -> ()
// CHECK:           %[[VAL_19:.*]] = fir.address_of(@_QQcl.6661696C2074797065) : !fir.ref<!fir.char<1,9>>
// CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<1,9>>) -> !fir.ref<i8>
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_2]] : (index) -> i64
// CHECK:           %[[VAL_22:.*]] = fir.call @_FortranAStopStatementText(%[[VAL_20]], %[[VAL_21]], %[[VAL_1]], %[[VAL_1]]) fastmath<contract> : (!fir.ref<i8>, i64, i1, i1) -> none
// CHECK:           fir.unreachable
// CHECK:         ^bb4:
// CHECK:           cf.br ^bb3(%[[VAL_13]] : !fir.class<!fir.ptr<none>>)
// CHECK:         ^bb5:
// CHECK:           %[[VAL_23:.*]] = fir.box_addr %[[VAL_13]] : (!fir.class<!fir.ptr<none>>) -> !fir.ptr<i32>
// CHECK:           fir.store %[[VAL_0]] to %[[VAL_23]] : !fir.ptr<i32>
// CHECK:           return
// CHECK:         }
func.func @_QPtest() {
  %c23_i32 = arith.constant 23 : i32
  %false = arith.constant false
  %c9 = arith.constant 9 : index
  %c777_i32 = arith.constant 777 : i32
  %0 = fir.alloca !fir.class<!fir.ptr<none>> {bindc_name = ".result"}
  %1 = fir.alloca !fir.class<!fir.ptr<none>> {bindc_name = ".result"}
  %2 = fir.call @_QPfstar() fastmath<contract> : () -> !fir.class<!fir.ptr<none>>
  fir.save_result %2 to %1 : !fir.class<!fir.ptr<none>>, !fir.ref<!fir.class<!fir.ptr<none>>>
  %3 = fir.load %1 : !fir.ref<!fir.class<!fir.ptr<none>>>
  fir.select_type %3 : !fir.class<!fir.ptr<none>> [#fir.type_is<i32>, ^bb1, unit, ^bb2(%3 : !fir.class<!fir.ptr<none>>)]
^bb1:  // pred: ^bb0
  %4 = fir.box_addr %3 : (!fir.class<!fir.ptr<none>>) -> !fir.ptr<i32>
  fir.store %c777_i32 to %4 : !fir.ptr<i32>
  %5 = fir.call @_QPfstar() fastmath<contract> : () -> !fir.class<!fir.ptr<none>>
  fir.save_result %5 to %0 : !fir.class<!fir.ptr<none>>, !fir.ref<!fir.class<!fir.ptr<none>>>
  %6 = fir.load %0 : !fir.ref<!fir.class<!fir.ptr<none>>>
  fir.select_type %6 : !fir.class<!fir.ptr<none>> [#fir.type_is<i32>, ^bb3, unit, ^bb2(%6 : !fir.class<!fir.ptr<none>>)]
^bb2(%7: !fir.class<!fir.ptr<none>>):  // 2 preds: ^bb0, ^bb1
  %8 = fir.rebox %7 : (!fir.class<!fir.ptr<none>>) -> !fir.class<none>
  fir.call @_QPfoo(%8) fastmath<contract> : (!fir.class<none>) -> ()
  %9 = fir.address_of(@_QQcl.6661696C2074797065) : !fir.ref<!fir.char<1,9>>
  %10 = fir.convert %9 : (!fir.ref<!fir.char<1,9>>) -> !fir.ref<i8>
  %11 = fir.convert %c9 : (index) -> i64
  %12 = fir.call @_FortranAStopStatementText(%10, %11, %false, %false) fastmath<contract> : (!fir.ref<i8>, i64, i1, i1) -> none
  fir.unreachable
^bb3:  // pred: ^bb1
  %13 = fir.box_addr %6 : (!fir.class<!fir.ptr<none>>) -> !fir.ptr<i32>
  fir.store %c23_i32 to %13 : !fir.ptr<i32>
  return
}
