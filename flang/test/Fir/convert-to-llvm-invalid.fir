// Test FIR to LLVM IR conversion invalid cases and diagnostics.

// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" --verify-diagnostics %s

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
