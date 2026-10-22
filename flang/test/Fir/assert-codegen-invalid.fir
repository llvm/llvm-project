// RUN: fir-opt --split-input-file --fir-to-llvm-ir --verify-diagnostics %s

llvm.func @_FortranAReportFatalUserErrorValues(!llvm.ptr, !llvm.ptr, i32, i32, i64, i64, i64)
func.func @llvm_signature(%ok: i1) {
  // expected-error@+2 {{incompatible runtime function declaration}}
  // expected-error@+1 {{failed to legalize operation}}
  fir.assert %ok, status, "x"
  return
}

// -----

fir.global @_FortranAReportFatalUserErrorValues : i32
func.func @name_conflict(%ok: i1) {
  // expected-error@+2 {{runtime function name conflicts with a symbol}}
  // expected-error@+1 {{failed to legalize operation}}
  fir.assert %ok, status, "x"
  return
}

// -----

func.func @func_arity(%ok: i1) {
  // expected-error@+2 {{incompatible runtime function declaration}}
  // expected-error@+1 {{failed to legalize operation}}
  fir.assert %ok, status, "x"
  return
}
func.func private @_FortranAReportFatalUserErrorValues(!fir.ref<i8>, !fir.ref<i8>, i32, i64, i64, i64)

// -----

func.func @func_argument_type(%ok: i1) {
  // expected-error@+2 {{incompatible runtime function declaration}}
  // expected-error@+1 {{failed to legalize operation}}
  fir.assert %ok, status, "x"
  return
}
func.func private @_FortranAReportFatalUserErrorValues(!fir.ref<i8>, !fir.ref<i8>, i32, i32, i64, i64, i64)

// -----

func.func @func_result(%ok: i1) {
  // expected-error@+2 {{incompatible runtime function declaration}}
  // expected-error@+1 {{failed to legalize operation}}
  fir.assert %ok, status, "x"
  return
}
func.func private @_FortranAReportFatalUserErrorValues(!fir.ref<i8>, !fir.ref<i8>, i32, i64, i64, i64, i64) -> i32

// -----

llvm.func @_FortranAReportFatalUserErrorValueArray(!llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr)
func.func @array_llvm_signature(%ok: i1, %n: i64) {
  // expected-error@+2 {{incompatible runtime function declaration}}
  // expected-error@+1 {{failed to legalize operation}}
  fir.assert %ok, bounds, "%lld %lld %lld %lld %lld" values(%n, %n, %n, %n, %n : i64, i64, i64, i64, i64)
  return
}

// -----

fir.global @_FortranAReportFatalUserErrorValueArray : i32
func.func @array_name_conflict(%ok: i1, %n: i64) {
  // expected-error@+2 {{runtime function name conflicts with a symbol}}
  // expected-error@+1 {{failed to legalize operation}}
  fir.assert %ok, bounds, "%lld %lld %lld %lld %lld" values(%n, %n, %n, %n, %n : i64, i64, i64, i64, i64)
  return
}

// -----

func.func @array_func_argument_type(%ok: i1, %n: i64) {
  // expected-error@+2 {{incompatible runtime function declaration}}
  // expected-error@+1 {{failed to legalize operation}}
  fir.assert %ok, bounds, "%lld %lld %lld %lld %lld" values(%n, %n, %n, %n, %n : i64, i64, i64, i64, i64)
  return
}
func.func private @_FortranAReportFatalUserErrorValueArray(!fir.ref<i8>, !fir.ref<i8>, i32, i64, i64)
