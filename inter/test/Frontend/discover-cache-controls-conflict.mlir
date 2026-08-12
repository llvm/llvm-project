// RUN: not inter-opt %s --inter-discover-cache-controls 2>&1 | FileCheck %s

module {
  llvm.mlir.global private constant @cached("{6442:\220,1\22}\00") {
    addr_space = 1 : i32, section = "llvm.metadata"}
  llvm.mlir.global private constant @uncached("{6442:\220,0\22}\00") {
    addr_space = 1 : i32, section = "llvm.metadata"}
  llvm.mlir.global private constant @file("\00") {
    addr_space = 1 : i32, section = "llvm.metadata"}

  func.func @conflict(%condition: i1, %lhs: !llvm.ptr<1>,
                      %rhs: !llvm.ptr<1>) {
    %cached = llvm.mlir.addressof @cached : !llvm.ptr<1>
    %uncached = llvm.mlir.addressof @uncached : !llvm.ptr<1>
    %file = llvm.mlir.addressof @file : !llvm.ptr<1>
    %zero = llvm.mlir.zero : !llvm.ptr<1>
    %line = llvm.mlir.constant(0 : i32) : i32
    %a = "llvm.intr.ptr.annotation"(%lhs, %cached, %file, %line, %zero)
        : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, !llvm.ptr<1>)
        -> !llvm.ptr<1>
    %b = "llvm.intr.ptr.annotation"(%rhs, %uncached, %file, %line, %zero)
        : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, !llvm.ptr<1>)
        -> !llvm.ptr<1>
    %selected = llvm.select %condition, %a, %b : i1, !llvm.ptr<1>
    %loaded = llvm.load %selected : !llvm.ptr<1> -> i32
    return
  }

  func.func @mixed(%condition: i1, %lhs: !llvm.ptr<1>,
                   %rhs: !llvm.ptr<1>) {
    %cached = llvm.mlir.addressof @cached : !llvm.ptr<1>
    %file = llvm.mlir.addressof @file : !llvm.ptr<1>
    %zero = llvm.mlir.zero : !llvm.ptr<1>
    %line = llvm.mlir.constant(0 : i32) : i32
    %annotated = "llvm.intr.ptr.annotation"(
        %lhs, %cached, %file, %line, %zero)
        : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i32, !llvm.ptr<1>)
        -> !llvm.ptr<1>
    %selected = llvm.select %condition, %annotated, %rhs
        : i1, !llvm.ptr<1>
    %loaded = llvm.load %selected : !llvm.ptr<1> -> i32
    return
  }
}

// CHECK: error: 'llvm.load' op conflicting load cache controls
// CHECK: error: 'llvm.load' op conflicting load cache controls
