// RUN: inter-opt %s | FileCheck %s

module {
  func.func @ptradd(%base: !llvm.ptr<1>, %offset: i64) {
    // CHECK: xw.ptradd %{{.*}}, %{{.*}} : !llvm.ptr<1>, i64
    %ptr = xw.ptradd %base, %offset : !llvm.ptr<1>, i64
    // CHECK: xw.ptradd %{{.*}}, %{{.*}} {gep_flags = 3 : i32}
    %inbounds = xw.ptradd %ptr, %offset {gep_flags = 3 : i32}
        : !llvm.ptr<1>, i64
    return
  }

  func.func @wide_ptradd(%base: !llvm.ptr<1>, %input: i32) {
    // CHECK: [[WIDE:%.*]] = xw.wide_extend %{{.*}} signed : i32
    %wide = xw.wide_extend %input signed : i32
    // CHECK: xw.ptradd %{{.*}}, [[WIDE]] : !llvm.ptr<1>, i64
    %ptr = xw.ptradd %base, %wide : !llvm.ptr<1>, i64
    return
  }
}
