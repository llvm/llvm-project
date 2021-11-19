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

// -----

// Verify that `fir.dt_entry` requires a parent op

// expected-error@+1{{'fir.dt_entry' op expects parent op 'fir.dispatch_table'}}
fir.dt_entry "method", @method_impl

// -----

// Test `fir.shape` conversion failure because the op has uses.

func @shape_not_dead(%arg0: !fir.ref<!fir.array<?x?xf32>>, %i: index, %j: index) {
  %c0 = arith.constant 1 : index
  // expected-error@+1{{failed to legalize operation 'fir.shape'}}
  %0 = fir.shape %c0, %c0 : (index, index) -> !fir.shape<2>
  %1 = fir.array_coor %arg0(%0) %i, %j : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
  return
}

// -----

// Test `fir.slice` conversion failure because the op has uses.

func @slice_not_dead(%arg0: !fir.ref<!fir.array<?x?xf32>>, %i: index, %j: index) {
  %c0 = arith.constant 1 : index
  // expected-error@+1{{failed to legalize operation 'fir.slice'}}
  %0 = fir.slice %c0, %c0, %c0, %c0, %c0, %c0 : (index, index, index, index, index, index) -> !fir.slice<2>
  %1 = fir.array_coor %arg0[%0] %i, %j : (!fir.ref<!fir.array<?x?xf32>>, !fir.slice<2>, index, index) -> !fir.ref<f32>
  return
}

// -----

// Test `fir.shift` conversion failure because the op has uses.

func @shift_not_dead(%arg0: !fir.box<!fir.array<?xf32>>, %i: index) {
  %c0 = arith.constant 1 : index
  // expected-error@+1{{failed to legalize operation 'fir.shift'}}
  %0 = fir.shift %c0 : (index) -> !fir.shift<1>
  %1 = fir.array_coor %arg0(%0) %i : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, index) -> !fir.ref<f32>
  return
}

// -----

// Test `fir.shape_shift` conversion failure because the op has uses.

func @shape_shift_not_dead(%arg0: !fir.ref<!fir.array<?x?xf32>>, %i: index, %j: index) {
  %c0 = arith.constant 1 : index
  // expected-error@+1{{failed to legalize operation 'fir.shape_shift'}}
  %0 = fir.shape_shift %c0, %c0, %c0, %c0 : (index, index, index, index) -> !fir.shapeshift<2>
  %1 = fir.array_coor %arg0(%0) %i, %j : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>, index, index) -> !fir.ref<f32>
  return
}

// -----

// Test that the type `fir.boxproc` fails to be legalized.
// Not implemented yet.

// expected-error@+1{{failed to legalize operation 'builtin.func'}}
func private @foo0(%arg0: !fir.boxproc<() -> ()>)

// -----

// Test that `fir.emboxproc` fails to be legalized.
// Not implemented yet.

func private @method_impl(i32)

func @emboxproc_test() {
  %host_vars = fir.alloca tuple<i32,f64>
// expected-error@+1{{failed to legalize operation 'fir.emboxproc'}}
  %bproc = fir.emboxproc @method_impl, %host_vars : ((i32) -> (), !fir.ref<tuple<i32,f64>>) -> !fir.boxproc<(i32) -> ()>
  return
}

// Test that `fir.unboxproc` and `fir.boxproc_host` also fails to be legalized.
// At the moment these cannot be tested since the `fir.boxproc` type does not have a conversion.

// -----

// Test `fir.end` conversion failure. Not implemented yet.

// expected-error@+1{{failed to legalize operation 'fir.end'}}
"fir.end"() : () -> ()
