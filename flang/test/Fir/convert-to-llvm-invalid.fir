// Test FIR to LLVM IR conversion invalid cases and diagnostics.

// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" --verify-diagnostics %s

// Test `fir.zero` conversion failure with aggregate type.
// Not implemented yet.

func @zero_aggregate() {
  // expected-error@+1{{failed to legalize operation 'fir.zero_bits'}}
  %a = fir.zero_bits !fir.array<10xf32>
  return
}

// -----

// Test that `fir.dispatch` fails to be legalized. Not implemented yet.

func @dispatch(%arg0: !fir.box<!fir.type<derived3{f:f32}>>) {
  // expected-error@+1{{failed to legalize operation 'fir.dispatch'}}
  %0 = fir.dispatch "method"(%arg0) : (!fir.box<!fir.type<derived3{f:f32}>>) -> i32
  return
}

// -----

// Test that `fir.dispatch_table`/`fir.dt_entry` fails to be legalized.
// Not implemented yet.

// expected-error@+1{{failed to legalize operation 'fir.dispatch_table'}}
fir.dispatch_table @dispatch_tbl {
  fir.dt_entry "method", @method_impl
}

// -----

// Test `fir.select_case` conversion failure with character type.
// Not implemented yet.

func @select_case_charachter(%arg0: !fir.char<2, 10>, %arg1: !fir.char<2, 10>, %arg2: !fir.char<2, 10>) {
  // expected-error@+1{{failed to legalize operation 'fir.select_case'}}
  fir.select_case %arg0 : !fir.char<2, 10> [#fir.point, %arg1, ^bb1,
                                            #fir.point, %arg2, ^bb2,
                                            unit, ^bb3]
^bb1:
  %c1_i32 = arith.constant 1 : i32
  br ^bb3
^bb2:
  %c2_i32 = arith.constant 2 : i32
  br ^bb3
^bb3:
  return
}
