// RUN: fir-opt --split-input-file --omp-polymorphic-rtti-declare-target %s | FileCheck %s

// Data-only polymorphic maps require type descriptor data for the declared
// type and materialized descendants, but they must not pull in dispatch-table
// procedures when no device dispatch occurs.

builtin.module attributes {omp.version = #omp.version<version = 61>} {
  fir.global @"_QMdata_onlyE.dt.base_t" : !fir.type<_QMdata_onlyTbase_t{base:i32}>
  fir.global @"_QMdata_onlyE.dt.child_t" : !fir.type<_QMdata_onlyTchild_t{base:i32,child:i32}>
  fir.type_info @_QMdata_onlyTbase_t : !fir.type<_QMdata_onlyTbase_t{base:i32}> dispatch_table {
    fir.dt_entry "unused", @data_only_unused
  }
  fir.type_info @_QMdata_onlyTchild_t extends !fir.type<_QMdata_onlyTbase_t{base:i32}> : !fir.type<_QMdata_onlyTchild_t{base:i32,child:i32}> dispatch_table {
    fir.dt_entry "unused", @data_only_child_unused
  }
  func.func private @data_only_unused()
  func.func private @data_only_child_unused()

  func.func @data_only_map(%arg0: !fir.ref<!fir.class<!fir.heap<!fir.type<_QMdata_onlyTbase_t{base:i32}>>>>) {
    %0 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMdata_onlyTbase_t{base:i32}>>>>, !fir.class<!fir.heap<!fir.type<_QMdata_onlyTbase_t{base:i32}>>>) map_clauses(tofrom) capture(ByRef) name("obj") -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMdata_onlyTbase_t{base:i32}>>>>
    omp.target kernel_type(generic) map_entries(%0 -> %arg1 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMdata_onlyTbase_t{base:i32}>>>>) {
      %c42 = arith.constant 42 : i32
      %1 = fir.load %arg1 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMdata_onlyTbase_t{base:i32}>>>>
      %2 = fir.box_addr %1 : (!fir.class<!fir.heap<!fir.type<_QMdata_onlyTbase_t{base:i32}>>>) -> !fir.heap<!fir.type<_QMdata_onlyTbase_t{base:i32}>>
      %3 = fir.coordinate_of %2, base : (!fir.heap<!fir.type<_QMdata_onlyTbase_t{base:i32}>>) -> !fir.ref<i32>
      fir.store %c42 to %3 : !fir.ref<i32>
      omp.terminator
    }
    return
  }
}

// CHECK-LABEL: module attributes
// CHECK:         fir.global @_QMdata_onlyE.dt.base_t {{{.*}}omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>{{.*}}}
// CHECK:         fir.global @_QMdata_onlyE.dt.child_t {{{.*}}omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>{{.*}}}
// CHECK:         func.func private @data_only_unused()
// CHECK:         func.func private @data_only_child_unused()
// CHECK-LABEL:   func.func @data_only_map

// -----

// A device fir.dispatch should mark only the selected binding name from each
// reachable dispatch table. Other bindings in the same tables remain host-only.

builtin.module attributes {omp.version = #omp.version<version = 61>} {
  fir.global @"_QMdispatch_usedE.dt.base_t" : !fir.type<_QMdispatch_usedTbase_t{base:i32}>
  fir.global @"_QMdispatch_usedE.dt.child_t" : !fir.type<_QMdispatch_usedTchild_t{base:i32,child:i32}>
  fir.type_info @_QMdispatch_usedTbase_t : !fir.type<_QMdispatch_usedTbase_t{base:i32}> dispatch_table {
    fir.dt_entry "used", @dispatch_base_used
    fir.dt_entry "unused", @dispatch_base_unused
  }
  fir.type_info @_QMdispatch_usedTchild_t extends !fir.type<_QMdispatch_usedTbase_t{base:i32}> : !fir.type<_QMdispatch_usedTchild_t{base:i32,child:i32}> dispatch_table {
    fir.dt_entry "used", @dispatch_child_used
    fir.dt_entry "unused", @dispatch_child_unused
  }
  func.func private @dispatch_base_used()
  func.func private @dispatch_base_unused()
  func.func private @dispatch_child_used()
  func.func private @dispatch_child_unused()

  func.func @dispatch_target(%arg0: !fir.ref<!fir.class<!fir.heap<!fir.type<_QMdispatch_usedTbase_t{base:i32}>>>>) {
    %0 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMdispatch_usedTbase_t{base:i32}>>>>, !fir.class<!fir.heap<!fir.type<_QMdispatch_usedTbase_t{base:i32}>>>) map_clauses(tofrom) capture(ByRef) name("obj") -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMdispatch_usedTbase_t{base:i32}>>>>
    omp.target kernel_type(generic) map_entries(%0 -> %arg1 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMdispatch_usedTbase_t{base:i32}>>>>) {
      %1 = fir.load %arg1 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMdispatch_usedTbase_t{base:i32}>>>>
      fir.dispatch "used"(%1 : !fir.class<!fir.heap<!fir.type<_QMdispatch_usedTbase_t{base:i32}>>>)
      omp.terminator
    }
    return
  }
}

// CHECK-LABEL: module attributes
// CHECK:         fir.global @_QMdispatch_usedE.dt.base_t {{{.*}}omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>{{.*}}}
// CHECK:         fir.global @_QMdispatch_usedE.dt.child_t {{{.*}}omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>{{.*}}}
// CHECK:         func.func private @dispatch_base_used() attributes {{{.*}}omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>{{.*}}}
// CHECK:         func.func private @dispatch_base_unused()
// CHECK:         func.func private @dispatch_child_used() attributes {{{.*}}omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>{{.*}}}
// CHECK:         func.func private @dispatch_child_unused()
// CHECK-LABEL:   func.func @dispatch_target

// -----

// Mapping class(child_a_t) should process child_a_t, its descendants, and its
// ancestors, but not unrelated siblings of child_a_t.

builtin.module attributes {omp.version = #omp.version<version = 61>} {
  fir.global @"_QMno_siblingE.dt.base_t" : !fir.type<_QMno_siblingTbase_t{base:i32}>
  fir.global @"_QMno_siblingE.dt.child_a_t" : !fir.type<_QMno_siblingTchild_a_t{base:i32,a:i32}>
  fir.global @"_QMno_siblingE.dt.grandchild_a_t" : !fir.type<_QMno_siblingTgrandchild_a_t{base:i32,a:i32,ga:i32}>
  fir.global @"_QMno_siblingE.dt.child_b_t" : !fir.type<_QMno_siblingTchild_b_t{base:i32,b:i32}>
  fir.type_info @_QMno_siblingTbase_t : !fir.type<_QMno_siblingTbase_t{base:i32}>
  fir.type_info @_QMno_siblingTchild_a_t extends !fir.type<_QMno_siblingTbase_t{base:i32}> : !fir.type<_QMno_siblingTchild_a_t{base:i32,a:i32}>
  fir.type_info @_QMno_siblingTgrandchild_a_t extends !fir.type<_QMno_siblingTchild_a_t{base:i32,a:i32}> : !fir.type<_QMno_siblingTgrandchild_a_t{base:i32,a:i32,ga:i32}>
  fir.type_info @_QMno_siblingTchild_b_t extends !fir.type<_QMno_siblingTbase_t{base:i32}> : !fir.type<_QMno_siblingTchild_b_t{base:i32,b:i32}>

  func.func @no_sibling_map(%arg0: !fir.ref<!fir.class<!fir.heap<!fir.type<_QMno_siblingTchild_a_t{base:i32,a:i32}>>>>) {
    %0 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMno_siblingTchild_a_t{base:i32,a:i32}>>>>, !fir.class<!fir.heap<!fir.type<_QMno_siblingTchild_a_t{base:i32,a:i32}>>>) map_clauses(tofrom) capture(ByRef) name("obj") -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMno_siblingTchild_a_t{base:i32,a:i32}>>>>
    omp.target kernel_type(generic) map_entries(%0 -> %arg1 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMno_siblingTchild_a_t{base:i32,a:i32}>>>>) {
      omp.terminator
    }
    return
  }
}

// CHECK-LABEL: module attributes
// CHECK:         fir.global @_QMno_siblingE.dt.base_t {{{.*}}omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>{{.*}}}
// CHECK:         fir.global @_QMno_siblingE.dt.child_a_t {{{.*}}omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>{{.*}}}
// CHECK:         fir.global @_QMno_siblingE.dt.grandchild_a_t {{{.*}}omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>{{.*}}}
// CHECK:         fir.global @_QMno_siblingE.dt.child_b_t : !fir.type<_QMno_siblingTchild_b_t
// CHECK-LABEL:   func.func @no_sibling_map

// -----

// Target regions may call helper routines that perform the actual dispatch.
// This RTTI-specific pass scans such helpers so their fir.dispatch operations
// mark selected bindings. The generic OpenMP MarkDeclareTarget pass owns
// marking the helper functions themselves.

builtin.module attributes {omp.version = #omp.version<version = 61>} {
  fir.global @"_QMhelperE.dt.base_t" : !fir.type<_QMhelperTbase_t{base:i32}>
  fir.type_info @_QMhelperTbase_t : !fir.type<_QMhelperTbase_t{base:i32}> dispatch_table {
    fir.dt_entry "run", @helper_run_impl
    fir.dt_entry "unused", @helper_unused_impl
  }
  func.func private @helper_run_impl()
  func.func private @helper_unused_impl()

  func.func @helper(%arg0: !fir.class<!fir.heap<!fir.type<_QMhelperTbase_t{base:i32}>>>) {
    fir.dispatch "run"(%arg0 : !fir.class<!fir.heap<!fir.type<_QMhelperTbase_t{base:i32}>>>)
    return
  }

  func.func @helper_target(%arg0: !fir.ref<!fir.class<!fir.heap<!fir.type<_QMhelperTbase_t{base:i32}>>>>) {
    %0 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMhelperTbase_t{base:i32}>>>>, !fir.class<!fir.heap<!fir.type<_QMhelperTbase_t{base:i32}>>>) map_clauses(tofrom) capture(ByRef) name("obj") -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMhelperTbase_t{base:i32}>>>>
    omp.target kernel_type(generic) map_entries(%0 -> %arg1 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMhelperTbase_t{base:i32}>>>>) {
      %1 = fir.load %arg1 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMhelperTbase_t{base:i32}>>>>
      fir.call @helper(%1) : (!fir.class<!fir.heap<!fir.type<_QMhelperTbase_t{base:i32}>>>) -> ()
      omp.terminator
    }
    return
  }
}

// CHECK-LABEL: module attributes
// CHECK:         fir.global @_QMhelperE.dt.base_t {{{.*}}omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>{{.*}}}
// CHECK:         func.func private @helper_run_impl() attributes {{{.*}}omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>{{.*}}}
// CHECK:         func.func private @helper_unused_impl()
// CHECK:         func.func @helper(%{{.*}})
// CHECK-LABEL:   func.func @helper_target

// -----

// Existing explicit declare-target device functions are scan roots. Data-only
// RTTI use in such a function marks the descriptor global, but does not mark
// dispatch-table procedures.

builtin.module attributes {omp.version = #omp.version<version = 61>} {
  fir.global @"_QMexplicit_dataE.dt.base_t" : !fir.type<_QMexplicit_dataTbase_t{base:i32}>
  fir.type_info @_QMexplicit_dataTbase_t : !fir.type<_QMexplicit_dataTbase_t{base:i32}> dispatch_table {
    fir.dt_entry "unused", @explicit_data_unused
  }
  func.func private @explicit_data_unused()

  func.func @explicit_data() attributes {omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to>} {
    %0 = fir.type_desc !fir.type<_QMexplicit_dataTbase_t{base:i32}>
    return
  }
}

// CHECK-LABEL: module attributes
// CHECK:         fir.global @_QMexplicit_dataE.dt.base_t {{{.*}}omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>{{.*}}}
// CHECK:         func.func private @explicit_data_unused()
// CHECK:         func.func @explicit_data() attributes {omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to>}
