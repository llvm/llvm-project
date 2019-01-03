@class NSString;

void initiate() {
  NSString *string = @"hello";
  const char *cString = "world";
}

// RUN: clang-refactor-test list-actions -at=%s:4:25 %s | FileCheck --check-prefix=CHECK-ACTION %s
// CHECK-ACTION: Wrap in NSLocalizedString

// Ensure the the action can be initiated in the string literal:

// RUN: clang-refactor-test initiate -action localize-objc-string-literal -in=%s:4:22-30 %s | FileCheck --check-prefix=CHECK1 %s
// CHECK1: Initiated the 'localize-objc-string-literal' action at 4:22

// Ensure that the action can't be initiated in other places:

// RUN: not clang-refactor-test initiate -action localize-objc-string-literal -in=%s:1:1-10 -in=%s:3:1-18 -in=%s:4:1-21 -in=%s:5:1-32 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
// CHECK-NO: Failed to initiate the refactoring action

// Ensure that the action can be initiated using a selection, and only when that
// selection doesn't go out of the string.

// RUN: clang-refactor-test initiate -action localize-objc-string-literal -selected=%s:4:22-4:30 -selected=%s:4:25-4:30 -selected=%s:4:22-4:27 -selected=%s:4:25-4:27 -selected=%s:4:24-4:29 -selected=%s:4:23-4:30 %s | FileCheck --check-prefix=CHECK1 %s

// RUN: not clang-refactor-test initiate -action localize-objc-string-literal -selected=%s:4:20-4:30 -selected=%s:4:1-4:25 -selected=%s:4:25-5:5 -selected=%s:3:17-6:2 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
