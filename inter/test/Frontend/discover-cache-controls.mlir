// RUN: inter-opt %s --inter-discover-cache-controls | FileCheck %s
// RUN: inter-opt %s --inter-discover-cache-controls --lift-cf-to-scf \
// RUN:   --inter-verify-structured --inter-convert-llvm-to-xw \
// RUN:   | FileCheck %s --check-prefix=XW

module {
  llvm.mlir.global private constant @load_l1("{6442:\220,1\22}\00") {
    addr_space = 1 : i32, section = "llvm.metadata"}
  llvm.mlir.global private constant @load_l3("{6442:\221,1\22}\00") {
    addr_space = 1 : i32, section = "llvm.metadata"}
  llvm.mlir.global private constant @store_l1("{6443:\220,2\22}\00") {
    addr_space = 1 : i32, section = "llvm.metadata"}
  llvm.mlir.global private constant @file("\00") {
    addr_space = 1 : i32, section = "llvm.metadata"}

  func.func @direct(%pointer: !llvm.ptr<1>, %offset: i64, %value: i32)
      attributes {xw.simd_width = 16 : i32} {
    %load_l1 = llvm.mlir.addressof @load_l1 : !llvm.ptr<1>
    %load_l3 = llvm.mlir.addressof @load_l3 : !llvm.ptr<1>
    %store_l1 = llvm.mlir.addressof @store_l1 : !llvm.ptr<1>
    %file = llvm.mlir.addressof @file : !llvm.ptr<1>
    %zero = llvm.mlir.zero : !llvm.ptr<1>
    %line = llvm.mlir.constant(0 : i32) : i32
    %p0 = "llvm.intr.ptr.annotation"(%pointer, %load_l1, %file, %line, %zero)
        : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, !llvm.ptr<1>)
        -> !llvm.ptr<1>
    %p1 = "llvm.intr.ptr.annotation"(%p0, %load_l3, %file, %line, %zero)
        : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, !llvm.ptr<1>)
        -> !llvm.ptr<1>
    %p2 = "llvm.intr.ptr.annotation"(%p1, %store_l1, %file, %line, %zero)
        : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, !llvm.ptr<1>)
        -> !llvm.ptr<1>
    %derived = llvm.getelementptr %p2[%offset]
        : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %loaded = llvm.load %derived : !llvm.ptr<1> -> i32
    llvm.store %value, %derived : i32, !llvm.ptr<1>
    return
  }

  func.func @same_select(%condition: i1, %lhs: !llvm.ptr<1>,
                         %rhs: !llvm.ptr<1>)
      attributes {xw.simd_width = 16 : i32} {
    %load_l1 = llvm.mlir.addressof @load_l1 : !llvm.ptr<1>
    %file = llvm.mlir.addressof @file : !llvm.ptr<1>
    %zero = llvm.mlir.zero : !llvm.ptr<1>
    %line = llvm.mlir.constant(0 : i32) : i32
    %a = llvm.call_intrinsic "llvm.ptr.annotation.p1.p1"(
        %lhs, %load_l1, %file, %line, %zero)
        : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, !llvm.ptr<1>)
        -> !llvm.ptr<1>
    %b = "llvm.intr.ptr.annotation"(%rhs, %load_l1, %file, %line, %zero)
        : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, !llvm.ptr<1>)
        -> !llvm.ptr<1>
    %selected = llvm.select %condition, %a, %b : i1, !llvm.ptr<1>
    %loaded = llvm.load %selected : !llvm.ptr<1> -> i32
    return
  }

  func.func @same_phi(%condition: i1, %lhs: !llvm.ptr<1>,
                      %rhs: !llvm.ptr<1>)
      attributes {xw.simd_width = 16 : i32} {
    %load_l1 = llvm.mlir.addressof @load_l1 : !llvm.ptr<1>
    %file = llvm.mlir.addressof @file : !llvm.ptr<1>
    %zero = llvm.mlir.zero : !llvm.ptr<1>
    %line = llvm.mlir.constant(0 : i32) : i32
    %a = "llvm.intr.ptr.annotation"(%lhs, %load_l1, %file, %line, %zero)
        : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, !llvm.ptr<1>)
        -> !llvm.ptr<1>
    %b = "llvm.intr.ptr.annotation"(%rhs, %load_l1, %file, %line, %zero)
        : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, !llvm.ptr<1>)
        -> !llvm.ptr<1>
    cf.cond_br %condition, ^then, ^else
  ^then:
    cf.br ^merge(%a : !llvm.ptr<1>)
  ^else:
    cf.br ^merge(%b : !llvm.ptr<1>)
  ^merge(%merged: !llvm.ptr<1>):
    %loaded = llvm.load %merged : !llvm.ptr<1> -> i32
    return
  }
}

// CHECK-NOT: llvm.mlir.global
// CHECK-NOT: llvm.intr.ptr.annotation
// CHECK-LABEL: func.func @direct
// CHECK: llvm.load {{.*}} {xw.cache_control = {l1 = #xw.cache_policy<cached>, l3 = #xw.cache_policy<cached>}}
// CHECK: llvm.store {{.*}} {xw.cache_control = {l1 = #xw.cache_policy<write_back>}}
// CHECK-LABEL: func.func @same_select
// CHECK: llvm.select
// CHECK: llvm.load {{.*}} {xw.cache_control = {l1 = #xw.cache_policy<cached>}}
// CHECK-LABEL: func.func @same_phi
// CHECK: ^bb3(%[[MERGED:.*]]: !llvm.ptr<1>):
// CHECK: llvm.load %[[MERGED]] {xw.cache_control = {l1 = #xw.cache_policy<cached>}}

// XW-LABEL: func.func @direct
// XW: xw.load {{.*}} {xw.cache_control = {l1 = #xw.cache_policy<cached>, l3 = #xw.cache_policy<cached>}}
// XW: xw.store {{.*}} {xw.cache_control = {l1 = #xw.cache_policy<write_back>}}
