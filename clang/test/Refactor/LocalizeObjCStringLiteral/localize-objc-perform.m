@class NSString;

void perform() {
  NSString *string = @"hello";
}
// CHECK1: "NSLocalizedString(" [[@LINE-2]]:22 -> [[@LINE-2]]:22
// CHECK1-NEXT: ", @"")" [[@LINE-3]]:30 -> [[@LINE-3]]:30
// RUN: clang-refactor-test perform -action localize-objc-string-literal -at=%s:4:22 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action localize-objc-string-literal -selected=%s:4:23-4:30 %s | FileCheck --check-prefix=CHECK1 %s
