// Use --mlir-disable-threading so that the printing is serialized.
// RUN: fir-opt %s -pass-pipeline='builtin.module(test-fir-openacc-support)' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// A local variable goes by the name the source spells and by the name it is
// uniqued under.

func.func @local_alloca() {
  %0 = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFlocal_allocaEx", test.var_name}
  return
}

// CHECK: Visiting: %{{.*}} = fir.alloca i32
// CHECK-NEXT: Demangled name: "x"
// CHECK-NEXT: Mangled name: "_QFlocal_allocaEx"

// -----

// A declared variable states only its uniqued name, which is deconstructed
// when the name the source spells is asked for.

func.func @declared_variable() {
  %0 = fir.alloca f32
  %1 = fir.declare %0 {uniq_name = "_QMmodFdeclared_variableEy", test.var_name} : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

// CHECK: Visiting: %{{.*}} = fir.declare
// CHECK-NEXT: Demangled name: "y"
// CHECK-NEXT: Mangled name: "_QMmodFdeclared_variableEy"

// -----

// A global is reached through the symbol it is addressed by. The declaration
// of it holds the uniqued name, so both names are recovered through it.

fir.global @_QMmodEglob : i32 {
  %0 = fir.zero_bits i32
  fir.has_value %0 : i32
}

func.func @global_through_declare() {
  %0 = fir.address_of(@_QMmodEglob) : !fir.ref<i32>
  %1 = fir.declare %0 {uniq_name = "_QMmodEglob", test.var_name} : (!fir.ref<i32>) -> !fir.ref<i32>
  return
}

// CHECK: Visiting: %{{.*}} = fir.declare
// CHECK-NEXT: Demangled name: "glob"
// CHECK-NEXT: Mangled name: "_QMmodEglob"

// -----

// Even without a declaration to walk to, the symbol a global is addressed
// through is uniqued, so the name the source spells is recovered from it.

fir.global @_QMmodEglob : i32 {
  %0 = fir.zero_bits i32
  fir.has_value %0 : i32
}

func.func @global_address_of() {
  %0 = fir.address_of(@_QMmodEglob) {test.var_name} : !fir.ref<i32>
  return
}

// CHECK: Visiting: %{{.*}} = fir.address_of
// CHECK-NEXT: Demangled name: "glob"
// CHECK-NEXT: Mangled name: "_QMmodEglob"

// -----

// A data clause records the name the source spells, so the name the object is
// emitted under is recovered from the variable the clause states.

func.func @data_clause_name() {
  %0 = fir.alloca i32 {bindc_name = "arr", uniq_name = "_QFdata_clause_nameEarr"}
  %1 = acc.copyin varPtr(%0 : !fir.ref<i32>) name("arr") -> !fir.ref<i32> {test.var_name}
  acc.data dataOperands(%1 : !fir.ref<i32>) {
    acc.terminator
  }
  return
}

// CHECK: Visiting: %{{.*}} = acc.copyin
// CHECK-NEXT: Demangled name: "arr"
// CHECK-NEXT: Mangled name: "_QFdata_clause_nameEarr"

// -----

// A global mapped by a data clause is resolved against the symbols of the
// binary by the symbol it is addressed through.

fir.global @_QMmodEglob : i32 {
  %0 = fir.zero_bits i32
  fir.has_value %0 : i32
}

func.func @data_clause_global() {
  %0 = fir.address_of(@_QMmodEglob) : !fir.ref<i32>
  %1 = acc.copyin varPtr(%0 : !fir.ref<i32>) name("glob") -> !fir.ref<i32> {test.var_name}
  acc.data dataOperands(%1 : !fir.ref<i32>) {
    acc.terminator
  }
  return
}

// CHECK: Visiting: %{{.*}} = acc.copyin
// CHECK-NEXT: Demangled name: "glob"
// CHECK-NEXT: Mangled name: "_QMmodEglob"

// -----

// A data clause on a variable with no name to recover keeps the name the
// clause records.

func.func @data_clause_unnamed_variable() {
  %0 = fir.alloca i32
  %1 = acc.copyin varPtr(%0 : !fir.ref<i32>) name("unnamed") -> !fir.ref<i32> {test.var_name}
  acc.data dataOperands(%1 : !fir.ref<i32>) {
    acc.terminator
  }
  return
}

// CHECK: Visiting: %{{.*}} = acc.copyin
// CHECK-NEXT: Demangled name: "unnamed"
// CHECK-NEXT: Mangled name: "unnamed"

// -----

// A component is named through a path whose root is the variable that holds
// it. Only that root is uniqued - a component is not emitted under a name of
// its own - so only it is spelled differently by the two renderings.

func.func @derived_component() {
  %0 = fir.alloca !fir.type<_QMmodTpair{a:i32,b:i32}> {bindc_name = "p", uniq_name = "_QFderived_componentEp"}
  %1 = fir.coordinate_of %0, b {test.var_name} : (!fir.ref<!fir.type<_QMmodTpair{a:i32,b:i32}>>) -> !fir.ref<i32>
  return
}

// CHECK: Visiting: %{{.*}} = fir.coordinate_of
// CHECK-NEXT: Demangled name: "p%b"
// CHECK-NEXT: Mangled name: "_QFderived_componentEp%b"
