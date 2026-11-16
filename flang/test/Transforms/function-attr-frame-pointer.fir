// RUN: fir-opt --function-attr="frame-pointer=All" %s | FileCheck %s --check-prefix=ALL
// RUN: fir-opt --function-attr="frame-pointer=NonLeaf" %s | FileCheck %s --check-prefix=NONLEAF
// RUN: fir-opt --function-attr="frame-pointer=NonLeafNoReserve" %s | FileCheck %s --check-prefix=NONLEAFNORESERVE
// RUN: fir-opt --function-attr="frame-pointer=None" %s | FileCheck %s --check-prefix=NONE

func.func @_QPtest() {
  return
}

// ALL-LABEL: func.func @_QPtest()
// ALL-SAME: llvm.frame_pointer = #llvm.framePointerKind<all>

// NONLEAF-LABEL: func.func @_QPtest()
// NONLEAF-SAME: llvm.frame_pointer = #llvm.framePointerKind<"non-leaf">

// NONLEAFNORESERVE-LABEL: func.func @_QPtest()
// NONLEAFNORESERVE-SAME: llvm.frame_pointer = #llvm.framePointerKind<"non-leaf-no-reserve">

// NONE-LABEL: func.func @_QPtest()
// NONE-NOT: llvm.frame_pointer
