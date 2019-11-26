// RUN: %clang_cc1 -emit-llvm  -triple=armv7-apple-darwin -std=c++11 %s -o - -O1 \
// RUN:    | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -triple=armv7-apple-darwin -std=c++11 %s -o - -O1 \
// RUN:    -discard-value-names | FileCheck %s --check-prefix=DISCARDVALUE

extern "C" void branch();

bool test(bool pred) {
  // DISCARDVALUE: br i1 %0, label %2, label %3
  // CHECK: br i1 %pred, label %if.then, label %if.end

  if (pred) {
    // DISCARDVALUE: 2:
    // DISCARDVALUE-NEXT: tail call void @branch()
    // DISCARDVALUE-NEXT: br label %3

    // CHECK: if.then:
    // CHECK-NEXT: tail call void @branch()
    // CHECK-NEXT: br label %if.end
    branch();
  }

  // DISCARDVALUE: 3:
  // DISCARDVALUE-NEXT: ret i1 %0

  // CHECK: if.end:
  // CHECK-NEXT: ret i1 %pred
  return pred;
}
