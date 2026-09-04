// RUN: inter-opt %s --split-input-file --inter-insert-sync="chip=bmg" -verify-diagnostics

func.func @unavailable_wait() attributes {xemachine.grf_count = 128 : i32} {
  // expected-error@+1 {{wait names SBID 16 but this GRF mode exposes 16}}
  %wait = xemachine.sync nop
      {swsbToken = 16 : i32, swsbTokenMode = 3 : i32}
      : !xemachine.mem.token
  return
}

// -----

func.func @unavailable_wait_mask()
    attributes {xemachine.grf_count = 128 : i32} {
  // expected-error@+1 {{selective wait mask names an unavailable SBID}}
  %wait = xemachine.sync allwr {sbidMask = 65536 : i32}
      : !xemachine.mem.token
  return
}

// -----

func.func @token_mode_without_token() {
  // expected-error@+1 {{token mode requires a nonnegative token}}
  %wait = xemachine.sync nop {swsbTokenMode = 3 : i32}
      : !xemachine.mem.token
  return
}
