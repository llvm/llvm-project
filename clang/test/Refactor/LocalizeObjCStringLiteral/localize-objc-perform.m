@class NSString;

void perform() {
  NSString *string = @"hello";
}
// CHECK1: "NSLocalizedString(" [[@LINE-2]]:22 -> [[@LINE-2]]:22
// CHECK1-NEXT: ", @"")" [[@LINE-3]]:30 -> [[@LINE-3]]:30
// RUN: clang-refactor-test perform -action localize-objc-string-literal -at=%s:4:22 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action localize-objc-string-literal -selected=%s:4:23-4:30 %s | FileCheck --check-prefix=CHECK1 %s

#define MACRO(x, y) x

void performInMacroArgument() {
  // macro-arg: +2:9
  // macro-arg-range-begin: +1:9
  MACRO(@"hello", 1);           // CHECK2: "NSLocalizedString(" [[@LINE]]:9 -> [[@LINE]]:9
  // macro-arg-range-end: -1:17 // CHECK2: ", @"")" [[@LINE-1]]:17 -> [[@LINE-1]]:17
}
// RUN: clang-refactor-test perform -action localize-objc-string-literal -at=macro-arg %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test perform -action localize-objc-string-literal -selected=macro-arg-range %s | FileCheck --check-prefix=CHECK2 %s
