// RUN: fir-opt %s --pass-pipeline="builtin.module(any(acc-fir-map-info-prep))" | FileCheck %s

// The data clauses of a declare directive on a module variable are mapped from
// the constructor and destructor of that variable. Those run outside of any
// Fortran function, so the map entries have to be materialized there as well:
// an allocatable maps as an attach of a CFI-described object, which the
// address of the descriptor alone does not state.

fir.global @_QMmEdata : !fir.box<!fir.heap<!fir.array<?xi32>>>

// CHECK-LABEL: llvm.func @_QMmEdata_acc_ctor
// CHECK: %[[SIZE:.*]] = arith.constant 0 : i64
// CHECK: acc.map_info varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
// CHECK-SAME: size(%[[SIZE]] : i64) elementSize(4)
// CHECK-SAME: descKind(cfi) mapFlags(ptr_and_obj)
// CHECK-NOT: acc.create
llvm.func @_QMmEdata_acc_ctor() {
  %addr = fir.address_of(@_QMmEdata) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %create = acc.create varPtr(%addr : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
      structured(false) name("data") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  acc.declare_enter dataOperands(%create : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  llvm.return
}

// CHECK-LABEL: llvm.func @_QMmEdata_acc_dtor
// CHECK: %[[SIZE:.*]] = arith.constant 0 : i64
// CHECK: acc.map_info varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
// CHECK-SAME: size(%[[SIZE]] : i64) elementSize(4)
// CHECK-SAME: exitLoc({{.*}}) descKind(cfi) mapFlags(ptr_and_obj)
// CHECK-NOT: acc.getdeviceptr
// CHECK-NOT: acc.delete
llvm.func @_QMmEdata_acc_dtor() {
  %addr = fir.address_of(@_QMmEdata) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %devptr = acc.getdeviceptr varPtr(%addr : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
      dataClause(acc_create) structured(false) name("data") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  acc.declare_exit dataOperands(%devptr : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  acc.delete accPtr(%devptr : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
      dataClause(acc_create) structured(false) name("data")
  llvm.return
}
