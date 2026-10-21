// RUN: fir-opt --simplify-intrinsics --mlir-print-debuginfo %s | FileCheck %s

// Verify that the source location of a maxloc call is propagated into the
// fir.do_loop generated inside the simplified function body, so that
// downstream passes that require a valid FileLineColLoc do not crash.

func.func @_QPtest(%arg0: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"}) {
  %c10 = arith.constant 10 : index
  %c4_i32 = arith.constant 4 : i32
  %c5_i32 = arith.constant 5 : i32
  %false = arith.constant false
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  %1 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %c0 = arith.constant 0 : index
  %2 = fir.shape %c0 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %4 = fir.shape %c10 : (index) -> !fir.shape<1>
  %5 = fir.embox %arg0(%4) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
  %6 = fir.absent !fir.box<i1>
  %7 = fir.address_of(@_QQcl) : !fir.ref<!fir.char<1,8>>
  %8 = fir.convert %0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  %9 = fir.convert %5 : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
  %10 = fir.convert %7 : (!fir.ref<!fir.char<1,8>>) -> !fir.ref<i8>
  %11 = fir.convert %6 : (!fir.box<i1>) -> !fir.box<none>
  fir.call @_FortranAMaxlocInteger4(%8, %9, %c4_i32, %10, %c5_i32, %11, %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> () loc("test.f90":5:3)
  return
}

// CHECK-LABEL: func.func private @_FortranAMaxlocInteger4x1_i32_contract_simplified(
// CHECK:         fir.do_loop
// CHECK:         loc([[LOC:#[^ ]*]])
// CHECK:         [[LOC]] = loc("test.f90":5:3)

func.func private @_FortranAMaxlocInteger4(!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> ()
fir.global linkonce @_QQcl constant : !fir.char<1,8> {
  %0 = fir.string_lit "test.f90\00"(8) : !fir.char<1,8>
  fir.has_value %0 : !fir.char<1,8>
}
