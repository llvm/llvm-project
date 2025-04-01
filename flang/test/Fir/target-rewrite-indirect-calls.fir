// Test that ABI attributes are set in indirect calls to BIND(C) functions.
// RUN: fir-opt --target-rewrite="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

func.func @test(%arg0: () -> (), %arg1: !fir.ref<!fir.type<t{a:!fir.array<5xf64>}>>, %arg2: f64) {
  %0 = fir.load %arg1 : !fir.ref<!fir.type<t{a:!fir.array<5xf64>}>>
  %1 = fir.convert %arg0 : (() -> ()) -> ((!fir.type<t{a:!fir.array<5xf64>}>, f64) -> ())
  fir.call %1(%0, %arg2) proc_attrs<bind_c> : (!fir.type<t{a:!fir.array<5xf64>}>, f64) -> ()
  return
}
// CHECK-LABEL:   func.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: () -> (),
// CHECK-SAME:                    %[[VAL_1:.*]]: !fir.ref<!fir.type<t{a:!fir.array<5xf64>}>>,
// CHECK-SAME:                    %[[VAL_2:.*]]: f64) {
// CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.type<t{a:!fir.array<5xf64>}>>
// CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_0]] : (() -> ()) -> ((!fir.ref<!fir.type<t{a:!fir.array<5xf64>}>>, f64) -> ())
// CHECK:           %[[VAL_5:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.type<t{a:!fir.array<5xf64>}>
// CHECK:           fir.store %[[VAL_3]] to %[[VAL_6]] : !fir.ref<!fir.type<t{a:!fir.array<5xf64>}>>
// CHECK:           fir.call %[[VAL_4]](%[[VAL_6]], %[[VAL_2]]) : (!fir.ref<!fir.type<t{a:!fir.array<5xf64>}>> {llvm.align = 8 : i32, llvm.byval = !fir.type<t{a:!fir.array<5xf64>}>}, f64) -> ()
// CHECK:           llvm.intr.stackrestore %[[VAL_5]] : !llvm.ptr
// CHECK:           return
// CHECK:         }
