// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module {
  func.func @_QQmain() {
    %2 = fir.address_of(@_QMmodEvar1) : !fir.ref<i32> loc(#loc1)
    %3 = fircg.ext_declare %2 {uniq_name = "_QMmodEvar1"} : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc1)
    return
  } loc(#loc1)
  fir.global @_QMmodEvar1 : i32 loc(#loc1)
}
#loc1 = loc("test1.f90":1:0)

// CHECK: #llvm.di_module<name = "mod", isDecl = true>
