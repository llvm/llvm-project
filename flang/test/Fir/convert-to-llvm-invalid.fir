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

// -----

// Test `fir.select_type` conversion failure. Not implemented yet.

func @bar_select_type(%arg : !fir.box<!fir.type<derived3{f:f32}>>) -> i32 {
  %0 = arith.constant 1 : i32
  %2 = arith.constant 3 : i32

  // expected-error@+1{{failed to legalize operation 'fir.select_type'}}
  fir.select_type %arg : !fir.box<!fir.type<derived3{f:f32}>> [ 
    #fir.instance<!fir.int<4>>,^bb1(%0:i32), 
    #fir.instance<!fir.int<8>>,^bb2(%2:i32), 
    unit,^bb5 ]
^bb1(%a : i32) :
  return %a : i32
^bb2(%b : i32) :
  return %b : i32
^bb5 :
  %zero = arith.constant 0 : i32
  return %zero : i32
}

// -----

// Test `fir.global_len` conversion failure. Not implemented yet.

fir.global @global_derived : !fir.type<minez(f:i32)> {
  // expected-error@+1{{failed to legalize operation 'fir.global_len'}}
  fir.global_len f, 1 : i32
  %0 = fir.undefined !fir.type<minez>
  fir.has_value %0 : !fir.type<minez>
}

// -----

// Test `fir.gentypedesc` conversion failure. Not implemented yet.

func @gentypedesc() {
  // expected-error@+1{{failed to legalize operation 'fir.gentypedesc'}}
  %0 = fir.gentypedesc !fir.type<derived3>
  return
}
