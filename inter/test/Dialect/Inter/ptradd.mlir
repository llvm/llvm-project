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
}
