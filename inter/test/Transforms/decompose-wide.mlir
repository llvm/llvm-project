// RUN: inter-opt %s --pass-pipeline='builtin.module(func.func(inter-decompose-wide,canonicalize,cse))' | FileCheck %s

module {
  // CHECK-LABEL: func.func @scaled_id
  // CHECK: [[ID:%.*]] = xw.global_id 0 : i64
  // CHECK: [[WIDE:%.*]] = xw.wide_extend [[ID]] : i64
  // CHECK: [[SCALED:%.*]] = xw.wide_shl [[WIDE]], 2
  // CHECK: xw.ptradd %{{.*}}, [[SCALED]] : !llvm.ptr<1>, i64
  // CHECK-NOT: llvm.mul
  func.func @scaled_id(%base: !llvm.ptr<1>) -> !llvm.ptr<1> {
    %id = xw.global_id 0 : i64
    %four = llvm.mlir.constant(4 : i64) : i64
    %offset = llvm.mul %id, %four : i64
    %ptr = xw.ptradd %base, %offset : !llvm.ptr<1>, i64
    return %ptr : !llvm.ptr<1>
  }

  // CHECK-LABEL: func.func @extensions
  // CHECK-DAG: [[SIGNED:%.*]] = xw.wide_extend %{{.*}} signed : i32
  // CHECK-DAG: [[UNSIGNED:%.*]] = xw.wide_extend %{{.*}} : i32
  // CHECK-DAG: xw.ptradd %{{.*}}, [[SIGNED]] : !llvm.ptr<1>, i64
  // CHECK-DAG: xw.ptradd %{{.*}}, [[UNSIGNED]] : !llvm.ptr<1>, i64
  func.func @extensions(%base: !llvm.ptr<1>, %signed_index: i32,
                        %unsigned_index: i32) -> (!llvm.ptr<1>, !llvm.ptr<1>) {
    %signed = llvm.sext %signed_index : i32 to i64
    %signed_ptr = xw.ptradd %base, %signed : !llvm.ptr<1>, i64
    %unsigned = llvm.zext %unsigned_index : i32 to i64
    %unsigned_ptr = xw.ptradd %base, %unsigned : !llvm.ptr<1>, i64
    return %signed_ptr, %unsigned_ptr : !llvm.ptr<1>, !llvm.ptr<1>
  }

  // CHECK-LABEL: func.func @arithmetic
  // CHECK-COUNT-2: xw.wide_extend %{{.*}} signed : i32
  // CHECK: [[ADD:%.*]] = xw.wide_add
  // CHECK: [[SUB:%.*]] = xw.wide_sub [[ADD]], %{{.*}}
  // CHECK: xw.ptradd %{{.*}}, [[SUB]] : !llvm.ptr<1>, i64
  func.func @arithmetic(%base: !llvm.ptr<1>, %lhs: i32, %rhs: i32)
      -> !llvm.ptr<1> {
    %lhs64 = llvm.sext %lhs : i32 to i64
    %rhs64 = llvm.sext %rhs : i32 to i64
    %sum = llvm.add %lhs64, %rhs64 : i64
    %difference = llvm.sub %sum, %rhs64 : i64
    %ptr = xw.ptradd %base, %difference : !llvm.ptr<1>, i64
    return %ptr : !llvm.ptr<1>
  }

  // CHECK-LABEL: func.func @slm_unchanged
  // CHECK: [[OFFSET:%.*]] = llvm.mul
  // CHECK: xw.ptradd %{{.*}}, [[OFFSET]] : !llvm.ptr<3>, i64
  // CHECK-NOT: xw.wide
  func.func @slm_unchanged(%base: !llvm.ptr<3>, %index: i64) -> !llvm.ptr<3> {
    %four = llvm.mlir.constant(4 : i64) : i64
    %offset = llvm.mul %index, %four : i64
    %ptr = xw.ptradd %base, %offset : !llvm.ptr<3>, i64
    return %ptr : !llvm.ptr<3>
  }
}
