// Test to assure we don't hit assertion when passing characters to maxval
// RUN: fir-opt %s -opt-bufferization | FileCheck %s


// This simplified `fir` is derived from this test program:
//  program FlangOptimizerBug
//     character(len=*), parameter :: values(*) = &
//              [ "To be   ","or not  " &
//              , "to      ","be.     " &
//              , "that    ","is      " &
//              , "the     ","question"]
//      integer :: me, ni, i
//      character(len=len(values)) :: my_val, expected
// 
//      me = 1
//      ni = 8
// 
//      my_val = values(mod(me-1, size(values))+1)
//      expected = maxval([(values(mod(i-1,size(values))+1), i = 1, ni)])
// 
//      print *, my_val, expected
//  end program

func.func @_QQmain() {
  %c8 = arith.constant 8 : index
  %1 = fir.alloca !fir.char<1, 8>
  %24 = fir.shape %c8 : (index) -> !fir.shape<1>
  %25 = hlfir.elemental %24 typeparams %c8 unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1, 8>> {
  ^bb0(%arg0: index):
    %dummy = fir.string_lit "A"(8) : !fir.char<1, 8>
    hlfir.yield_element %dummy : !fir.char<1, 8>
  }
  %26 = hlfir.maxval %25 {fastmath = #arith.fastmath<contract>} : (!hlfir.expr<?x!fir.char<1, 8>>) -> !hlfir.expr<!fir.char<1, 8>>
  hlfir.assign %26 to %1 : !hlfir.expr<!fir.char<1, 8>>, !fir.ref<!fir.char<1, 8>>   // Assign to %1 directly
  hlfir.destroy %26 : !hlfir.expr<!fir.char<1, 8>>
  hlfir.destroy %25 : !hlfir.expr<?x!fir.char<1, 8>>
  return
}

// CHECK-LABEL:  func.func @_QQmain() {
// CHECK-NEXT:    %c8 = arith.constant 8 : index
// CHECK-NEXT:    %[[V0:.*]] = fir.alloca !fir.char<1,8>
// CHECK-NEXT:    %[[V1:.*]] = fir.shape %c8 : (index) -> !fir.shape<1>
// CHECK-NEXT:    %[[V2:.*]] = hlfir.elemental %1 typeparams %c8 unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,8>> {
// CHECK-NEXT:    ^bb0(%arg0: index):
// CHECK-NEXT:      %[[V4:.*]] = fir.string_lit "A"(8) : !fir.char<1,8>
// CHECK-NEXT:      hlfir.yield_element %[[V4]] : !fir.char<1,8>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[V3:.*]] = hlfir.maxval %[[V2]] {fastmath = #arith.fastmath<contract>} : (!hlfir.expr<?x!fir.char<1,8>>) -> !hlfir.expr<!fir.char<1,8>>
// CHECK-NEXT:    hlfir.assign %[[V3]] to %[[V0]] : !hlfir.expr<!fir.char<1,8>>, !fir.ref<!fir.char<1,8>>
// CHECK-NEXT:    hlfir.destroy %[[V3]] : !hlfir.expr<!fir.char<1,8>>
// CHECK-NEXT:    hlfir.destroy %[[V2]] : !hlfir.expr<?x!fir.char<1,8>>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
