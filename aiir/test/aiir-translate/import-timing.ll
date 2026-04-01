; RUN: aiir-translate %s -import-llvm -aiir-timing 2>&1 | FileCheck %s

; CHECK: Execution time report
; CHECK: Total Execution Time:
; CHECK: Name
; CHECK-NEXT: Translate LLVMIR to AIIR

define void @foo() {
  ret void
}
