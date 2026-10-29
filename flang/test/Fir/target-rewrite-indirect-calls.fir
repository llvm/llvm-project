// Test that ABI attributes are set in indirect calls to BIND(C) functions.
// RUN: fir-opt --target-rewrite="target=x86_64-unknown-linux-gnu" %s | FileCheck %s --check-prefix=CHECK-X86
// RUN: fir-opt --target-rewrite="target=s390x-unknown-linux-gnu" %s | FileCheck %s --check-prefix=CHECK-S390

func.func @test(%arg0: () -> (), %arg1: !fir.ref<!fir.type<t{a:!fir.array<5xf64>}>>, %arg2: f64) {
  %0 = fir.load %arg1 : !fir.ref<!fir.type<t{a:!fir.array<5xf64>}>>
  %1 = fir.convert %arg0 : (() -> ()) -> ((!fir.type<t{a:!fir.array<5xf64>}>, f64) -> ())
  fir.call %1(%0, %arg2) proc_attrs<bind_c> : (!fir.type<t{a:!fir.array<5xf64>}>, f64) -> ()
  return
}

// CHECK-X86-LABEL:   func.func @test(
// CHECK-X86-SAME:                    %[[VAL_0:.*]]: () -> (),
// CHECK-X86-SAME:                    %[[VAL_1:.*]]: !fir.ref<!fir.type<t{a:!fir.array<5xf64>}>>,
// CHECK-X86-SAME:                    %[[VAL_2:.*]]: f64) {
// CHECK-X86:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.type<t{a:!fir.array<5xf64>}>>
// CHECK-X86:           %[[VAL_4:.*]] = fir.convert %[[VAL_0]] : (() -> ()) -> ((!fir.ref<!fir.type<t{a:!fir.array<5xf64>}>>, f64) -> ())
// CHECK-X86:           %[[VAL_5:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK-X86:           %[[VAL_6:.*]] = fir.alloca !fir.type<t{a:!fir.array<5xf64>}>
// CHECK-X86:           fir.store %[[VAL_3]] to %[[VAL_6]] : !fir.ref<!fir.type<t{a:!fir.array<5xf64>}>>
// CHECK-X86:           fir.call %[[VAL_4]](%[[VAL_6]], %[[VAL_2]]) : (!fir.ref<!fir.type<t{a:!fir.array<5xf64>}>> {llvm.align = 8 : i32, llvm.byval = !fir.type<t{a:!fir.array<5xf64>}>}, f64) -> ()
// CHECK-X86:           llvm.intr.stackrestore %[[VAL_5]] : !llvm.ptr
// CHECK-X86:           return
// CHECK-X86:         }

// CHECK-S390-LABEL:   func.func @test(
// CHECK-S390-SAME:                    %[[VAL_0:.*]]: () -> (),
// CHECK-S390-SAME:                    %[[VAL_1:.*]]: !fir.ref<!fir.type<t{a:!fir.array<5xf64>}>>,
// CHECK-S390-SAME:                    %[[VAL_2:.*]]: f64) {
// CHECK-S390:           %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.type<t{a:!fir.array<5xf64>}>>
// CHECK-S390:           %[[VAL_4:.*]] = fir.convert %[[VAL_0]] : (() -> ()) -> ((!fir.ref<!fir.type<t{a:!fir.array<5xf64>}>>, f64) -> ())
// CHECK-S390:           %[[VAL_5:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK-S390:           %[[VAL_6:.*]] = fir.alloca !fir.type<t{a:!fir.array<5xf64>}>
// CHECK-S390:           fir.store %[[VAL_3]] to %[[VAL_6]] : !fir.ref<!fir.type<t{a:!fir.array<5xf64>}>>
// CHECK-S390:           fir.call %[[VAL_4]](%[[VAL_6]], %[[VAL_2]]) : (!fir.ref<!fir.type<t{a:!fir.array<5xf64>}>> {llvm.align = 8 : i32}, f64) -> ()
// CHECK-S390-NOT:       llvm.byval
// CHECK-S390:           llvm.intr.stackrestore %[[VAL_5]] : !llvm.ptr
// CHECK-S390:           return
// CHECK-S390:         }
