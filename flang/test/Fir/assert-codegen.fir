// RUN: fir-opt --split-input-file --fir-to-llvm-ir %s | FileCheck %s

func.func @check(%ok: i1, %small: i32, %n: index, %array: !fir.ref<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>> {
  %g = fir.assert %ok, conformance, "extents %lld and %lld" values(%small, %n : i32, index) guard(%array : !fir.ref<!fir.array<?xi32>>) loc("source.f90":42:7)
  return %g : !fir.ref<!fir.array<?xi32>>
}
// CHECK-DAG: llvm.mlir.global internal constant @{{.*}}("extents %lld and %lld\00")
// CHECK-DAG: llvm.mlir.global internal constant @{{.*}}("source.f90\00")
// CHECK-DAG: llvm.func @_FortranAReportFatalUserErrorValues(!llvm.ptr, !llvm.ptr, i32, i64, i64, i64, i64) attributes {noreturn}
// CHECK-LABEL: llvm.func @check(
// CHECK-SAME: %[[OK:.*]]: i1, %[[SMALL:.*]]: i32, %[[N:.*]]: i64, %[[ARRAY:.*]]: !llvm.ptr)
// CHECK-NEXT: llvm.cond_br %[[OK]] weights([2000, 1]), ^[[CONT:bb[0-9]+]], ^[[FAIL:bb[0-9]+]]
// CHECK: ^[[CONT]]:
// CHECK-NEXT: llvm.return %[[ARRAY]] : !llvm.ptr
// CHECK: ^[[FAIL]]:
// CHECK: %[[MSG:.*]] = llvm.mlir.addressof
// CHECK: %[[FILE:.*]] = llvm.mlir.addressof
// CHECK: %[[LINE:.*]] = llvm.mlir.constant(42 : i32)
// CHECK: %[[EXT:.*]] = llvm.sext %[[SMALL]] : i32 to i64
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i64)
// CHECK: llvm.call @_FortranAReportFatalUserErrorValues(%[[MSG]], %[[FILE]], %[[LINE]], %[[EXT]], %[[N]], %[[ZERO]], %{{.*}})
// CHECK-NEXT: llvm.unreachable

// -----

// Existing declarations must also acquire noreturn.
llvm.func @_FortranAReportFatalUserErrorValues(!llvm.ptr, !llvm.ptr, i32, i64, i64, i64, i64)
func.func @existing_llvm(%ok: i1) {
  fir.assert %ok, status, "allocation failed" loc(unknown)
  return
}
// CHECK: llvm.func @_FortranAReportFatalUserErrorValues({{.*}}) attributes {noreturn}
// CHECK-LABEL: llvm.func @existing_llvm
// CHECK: llvm.cond_br
// CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: %[[LINE:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: llvm.call @_FortranAReportFatalUserErrorValues(%{{.*}}, %[[NULL]], %[[LINE]],
// CHECK-NEXT: llvm.unreachable

// -----

func.func @existing_fir(%ok: i1) {
  fir.assert %ok, status, "allocation failed" loc(fused["inlined"(callsite("inner.f90":9:2 at "outer.f90":80:1))])
  return
}
func.func private @_FortranAReportFatalUserErrorValues(!fir.ref<i8>, !fir.ref<i8>, i32, i64, i64, i64, i64)
// CHECK: llvm.mlir.global internal constant @{{.*}}("inner.f90\00")
// CHECK-LABEL: llvm.func @existing_fir
// CHECK: llvm.mlir.constant(9 : i32)
// CHECK: llvm.call @_FortranAReportFatalUserErrorValues
// CHECK: llvm.func @_FortranAReportFatalUserErrorValues({{.*}}) attributes {noreturn{{.*}}}

// -----

gpu.module @device {
  gpu.func @check_device(%ok: i1, %n: i64) {
    fir.assert %ok, bounds, "subscript %lld" values(%n : i64) loc("kernel.f90":11:3)
    gpu.return
  }
}
// CHECK-LABEL: gpu.module @device {
// CHECK-DAG: llvm.mlir.global internal constant @{{.*}}("kernel.f90\00")
// CHECK-DAG: llvm.mlir.global internal constant @{{.*}}("subscript %lld\00")
// CHECK-DAG: llvm.func @_FortranAReportFatalUserErrorValues({{.*}}) attributes {noreturn}
// CHECK: gpu.func @check_device
// CHECK: llvm.cond_br
// CHECK: llvm.call @_FortranAReportFatalUserErrorValues
// CHECK: llvm.unreachable
