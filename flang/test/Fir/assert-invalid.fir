// RUN: fir-opt --split-input-file --verify-diagnostics %s

func.func @format(%ok: i1, %i: i32) {
  // expected-error@+1 {{message supports only %lld and %% format directives}}
  fir.assert %ok, argument, "%d" values(%i : i32)
  return
}

// -----

func.func @trailing_percent(%ok: i1) {
  // expected-error@+1 {{message supports only %lld and %% format directives}}
  fir.assert %ok, argument, "bad %"
  return
}

// -----

func.func @too_few_values(%ok: i1) {
  // expected-error@+1 {{number of %lld directives must match diagnostic values}}
  fir.assert %ok, argument, "%lld"
  return
}

// -----

func.func @too_many_values(%ok: i1, %i: i32) {
  // expected-error@+1 {{number of %lld directives must match diagnostic values}}
  fir.assert %ok, argument, "%%lld" values(%i : i32)
  return
}

// -----

func.func @wide_integer(%ok: i1, %i: i128) {
  // expected-error@+1 {{diagnostic integers must be at most 64 bits wide}}
  fir.assert %ok, argument, "%lld" values(%i : i128)
  return
}

// -----

func.func @null_character(%ok: i1) {
  // expected-error@+1 {{message must not contain an embedded null character}}
  fir.assert %ok, argument, "bad\00argument"
  return
}

// -----

func.func @wrong_result_type(%ok: i1, %i: i32) {
  // expected-error@+1 {{failed to verify that result types match guarded operand types}}
  %g = "fir.assert"(%ok, %i) <{kind = #fir.runtime_check_kind<argument>, message = "bad", operandSegmentSizes = array<i32: 1, 0, 1>}> : (i1, i32) -> i64
  return
}

// -----

func.func @wrong_result_count(%ok: i1, %i: i32) {
  // expected-error@+1 {{failed to verify that result types match guarded operand types}}
  "fir.assert"(%ok, %i) <{kind = #fir.runtime_check_kind<argument>, message = "bad", operandSegmentSizes = array<i32: 1, 0, 1>}> : (i1, i32) -> ()
  return
}

// -----

func.func @real_value(%ok: i1, %x: f32) {
  // expected-error@+1 {{must be variadic of signless integer or index}}
  fir.assert %ok, argument, "%lld" values(%x : f32)
  return
}
