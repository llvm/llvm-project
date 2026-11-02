// RUN: fir-opt --split-input-file --fir-to-llvm-ir %s | FileCheck %s
// RUN: fir-opt --split-input-file --fir-to-llvm-ir="datalayout=e-p:64:64-p5:32:32-A5" %s | FileCheck %s --check-prefix=PRIVATE
// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=i386-unknown-linux-gnu datalayout=e-p:32:32" %s | FileCheck %s --check-prefix=COUNT32
// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=amdgcn-amd-amdhsa datalayout=e-p:64:64-p1:64:64-p5:32:32-A5-G1" %s | FileCheck %s --check-prefix=GLOBAL

// Five values select the array entry. Conversion, allocation and stores must
// stay on the failure path, and the guarded result survives on success.
func.func @array_values(%ok: i1, %a: i8, %b: i16, %c: i32, %d: index, %e: i64, %base: !fir.ref<i32>) -> !fir.ref<i32> {
  %g = fir.assert %ok, bounds, "dimension %lld, position %lld, index %lld, bounds %lld:%lld (100%%)" values(%a, %b, %c, %d, %e : i8, i16, i32, index, i64) guard(%base : !fir.ref<i32>) loc("bounds.f90":17:2)
  return %g : !fir.ref<i32>
}
// CHECK: llvm.func @_FortranAReportFatalUserErrorValueArray(!llvm.ptr, !llvm.ptr, i32, i64, !llvm.ptr) attributes {noreturn}
// CHECK-LABEL: llvm.func @array_values(
// CHECK-SAME: %[[OK:.*]]: i1, %[[A:.*]]: i8, %[[B:.*]]: i16, %[[C:.*]]: i32, %[[D:.*]]: i64, %[[E:.*]]: i64, %[[BASE:.*]]: !llvm.ptr)
// CHECK-NEXT: llvm.cond_br %[[OK]] weights([2000, 1]), ^[[CONT:bb[0-9]+]], ^[[FAIL:bb[0-9]+]]
// CHECK: ^[[CONT]]:
// CHECK-NEXT: llvm.return %[[BASE]] : !llvm.ptr
// CHECK: ^[[FAIL]]:
// CHECK: %[[COUNT:.*]] = llvm.mlir.constant(5 : i64)
// CHECK-NEXT: %[[STORAGE:.*]] = llvm.alloca %[[COUNT]] x i64 : (i64) -> !llvm.ptr
// CHECK: %[[P0:.*]] = llvm.getelementptr %[[STORAGE]][%{{.*}}]
// CHECK: %[[V0:.*]] = llvm.sext %[[A]] : i8 to i64
// CHECK-NEXT: llvm.store %[[V0]], %[[P0]]
// CHECK: %[[P1:.*]] = llvm.getelementptr %[[STORAGE]][%{{.*}}]
// CHECK: %[[V1:.*]] = llvm.sext %[[B]] : i16 to i64
// CHECK-NEXT: llvm.store %[[V1]], %[[P1]]
// CHECK: %[[P2:.*]] = llvm.getelementptr %[[STORAGE]][%{{.*}}]
// CHECK: %[[V2:.*]] = llvm.sext %[[C]] : i32 to i64
// CHECK-NEXT: llvm.store %[[V2]], %[[P2]]
// CHECK: %[[P3:.*]] = llvm.getelementptr %[[STORAGE]][%{{.*}}]
// CHECK-NEXT: llvm.store %[[D]], %[[P3]]
// CHECK: %[[P4:.*]] = llvm.getelementptr %[[STORAGE]][%{{.*}}]
// CHECK-NEXT: llvm.store %[[E]], %[[P4]]
// CHECK: llvm.call @_FortranAReportFatalUserErrorValueArray(%{{.*}}, %{{.*}}, %{{.*}}, %[[COUNT]], %[[STORAGE]])
// CHECK-NEXT: llvm.unreachable
// PRIVATE-LABEL: llvm.func @array_values
// PRIVATE: %[[STORAGE:.*]] = llvm.alloca %{{.*}} x i64 : (i64) -> !llvm.ptr<5>
// PRIVATE: %[[GENERIC:.*]] = llvm.addrspacecast %[[STORAGE]] : !llvm.ptr<5> to !llvm.ptr
// PRIVATE-NEXT: llvm.call @_FortranAReportFatalUserErrorValueArray(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[GENERIC]])
// COUNT32: llvm.func @_FortranAReportFatalUserErrorValueArray(!llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr) attributes {noreturn}
// COUNT32-LABEL: llvm.func @array_values
// COUNT32: %[[COUNT:.*]] = llvm.mlir.constant(5 : i32)
// COUNT32-NEXT: llvm.alloca %[[COUNT]] x i64 : (i32) -> !llvm.ptr
// COUNT32: llvm.call @_FortranAReportFatalUserErrorValueArray
// GLOBAL-DAG: llvm.mlir.global internal constant @{{.*}}("bounds.f90\00") {{.*}}addr_space = 1
// GLOBAL-DAG: llvm.mlir.global internal constant @{{.*}}("dimension %lld, position %lld, index %lld, bounds %lld:%lld (100%%)\00") {{.*}}addr_space = 1
// GLOBAL-LABEL: llvm.func @array_values
// GLOBAL: %[[MSG:.*]] = llvm.addrspacecast %{{.*}} : !llvm.ptr<1> to !llvm.ptr
// GLOBAL: %[[FILE:.*]] = llvm.addrspacecast %{{.*}} : !llvm.ptr<1> to !llvm.ptr
// GLOBAL: %[[STORAGE:.*]] = llvm.alloca %{{.*}} x i64 : (i64) -> !llvm.ptr<5>
// GLOBAL: %[[VALUES:.*]] = llvm.addrspacecast %[[STORAGE]] : !llvm.ptr<5> to !llvm.ptr
// GLOBAL: llvm.call @_FortranAReportFatalUserErrorValueArray(%[[MSG]], %[[FILE]], %{{.*}}, %{{.*}}, %[[VALUES]])

// -----

// A complete pair of rank-six shapes needs twelve values. The device runtime
// declaration and diagnostic strings must stay inside the GPU module.
gpu.module @device {
  gpu.func @shapes(%ok: i1, %n: index) {
    fir.assert %ok, conformance, "shapes (%lld,%lld,%lld,%lld,%lld,%lld) and (%lld,%lld,%lld,%lld,%lld,%lld)" values(%n, %n, %n, %n, %n, %n, %n, %n, %n, %n, %n, %n : index, index, index, index, index, index, index, index, index, index, index, index) loc("shapes.f90":23:1)
    gpu.return
  }
}
// CHECK-LABEL: gpu.module @device {
// CHECK-DAG: llvm.mlir.global internal constant @{{.*}}("shapes.f90\00")
// CHECK-DAG: llvm.func @_FortranAReportFatalUserErrorValueArray({{.*}}) attributes {noreturn}
// CHECK: gpu.func @shapes
// CHECK: llvm.cond_br
// CHECK: %[[COUNT:.*]] = llvm.mlir.constant(12 : i64)
// CHECK: %[[ARRAY:.*]] = llvm.alloca %[[COUNT]] x i64
// CHECK-COUNT-12: llvm.store
// CHECK: llvm.call @_FortranAReportFatalUserErrorValueArray(%{{.*}}, %{{.*}}, %{{.*}}, %[[COUNT]], %[[ARRAY]])
// CHECK-NEXT: llvm.unreachable
// GLOBAL-LABEL: gpu.module @device {
// GLOBAL-DAG: llvm.mlir.global internal constant @{{.*}}("shapes.f90\00") {{.*}}addr_space = 1
// GLOBAL: gpu.func @shapes
// GLOBAL: llvm.addrspacecast %{{.*}} : !llvm.ptr<1> to !llvm.ptr
// GLOBAL: llvm.call @_FortranAReportFatalUserErrorValueArray

// -----

func.func private @_FortranAReportFatalUserErrorValueArray(!fir.ref<i8>, !fir.ref<i8>, i32, index, !fir.ref<i64>)
func.func @existing_array(%ok: i1, %n: i64) {
  fir.assert %ok, bounds, "%lld %lld %lld %lld %lld" values(%n, %n, %n, %n, %n : i64, i64, i64, i64, i64)
  return
}
// CHECK: llvm.func @_FortranAReportFatalUserErrorValueArray({{.*}}) attributes {noreturn{{.*}}}
// CHECK-LABEL: llvm.func @existing_array
// CHECK: llvm.call @_FortranAReportFatalUserErrorValueArray
