// RUN: fir-opt --character-conversion %s | FileCheck %s

func @char_convert() {
  %1 = fir.undefined i32
  %2 = fir.undefined !fir.ref<!fir.char<1>>
  %3 = fir.undefined !fir.ref<!fir.array<?x!fir.char<2,?>>>
  fir.char_convert %2 for %1 to %3 : !fir.ref<!fir.char<1>>, i32, !fir.ref<!fir.array<?x!fir.char<2,?>>>
  return
}

// CHECK-LABEL:   func @char_convert() {
// CHECK:           %[[VAL_0:.*]] = fir.undefined i32
// CHECK:           %[[VAL_1:.*]] = fir.undefined !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_2:.*]] = fir.undefined !fir.ref<!fir.array<?x!fir.char<2,?>>>
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_0]] : (i32) -> index
// CHECK:           %[[VAL_6:.*]] = arith.subi %[[VAL_5]], %[[VAL_4]] : index
// CHECK:           fir.do_loop %[[VAL_7:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_4]] {
// CHECK:             %[[VAL_8:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.array<?xi8>>
// CHECK:             %[[VAL_9:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<?x!fir.char<2,?>>>) -> !fir.ref<!fir.array<?xi16>>
// CHECK:             %[[VAL_10:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_7]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
// CHECK:             %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_9]], %[[VAL_7]] : (!fir.ref<!fir.array<?xi16>>, index) -> !fir.ref<i16>
// CHECK:             %[[VAL_12:.*]] = fir.load %[[VAL_10]] : !fir.ref<i8>
// CHECK:             %[[VAL_13:.*]] = arith.extui %[[VAL_12]] : i8 to i16
// CHECK:             fir.store %[[VAL_13]] to %[[VAL_11]] : !fir.ref<i16>
// CHECK:           }
// CHECK:           return
// CHECK:         }
