; RUN: not %clang_cc1 -emit-llvm -x ir %s -o - -fprofile-instrument-use=llvm -fprofile-instrument-use-path=%t.nonexistent.profdata 2>&1 | FileCheck %s

; CHECK: error: {{.*}}.nonexistent.profdata:
; CHECK-NOT: Assertion failed

define i32 @main() {
  ret i32 0
}
