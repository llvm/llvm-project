// RUN: %clang_cc1 -I %S/Inputs -triple x86_64-apple-macosx10.10.0 -fobjc-runtime=macosx-10.10.0 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=CHECK-WITHOUT-EMPTY-COLLECTIONS %s
// RUN: %clang_cc1 -I %S/Inputs -triple x86_64-apple-macosx10.11.0 -fobjc-runtime=macosx-10.11.0 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=CHECK-WITH-EMPTY-COLLECTIONS %s

// RUN: %clang_cc1 -I %S/Inputs -triple arm64-apple-ios8.0 -fobjc-runtime=ios-8.0 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=CHECK-WITHOUT-EMPTY-COLLECTIONS %s
// RUN: %clang_cc1 -I %S/Inputs -triple arm64-apple-ios9.0 -fobjc-runtime=ios-9.0 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=CHECK-WITH-EMPTY-COLLECTIONS %s

// RUN: %clang_cc1 -I %S/Inputs -triple armv7k-apple-watchos2.0 -fobjc-runtime=watchos-1.0 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=CHECK-WITHOUT-EMPTY-COLLECTIONS %s
// RUN: %clang_cc1 -I %S/Inputs -triple armv7k-apple-watchos2.0 -fobjc-runtime=watchos-2.0 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=CHECK-WITH-EMPTY-COLLECTIONS %s

// RUN: %clang_cc1 -I %S/Inputs -triple arm64-apple-tvos8.0 -fobjc-runtime=ios-8.0 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=CHECK-WITHOUT-EMPTY-COLLECTIONS %s
// RUN: %clang_cc1 -I %S/Inputs -triple arm64-apple-tvos9.0 -fobjc-runtime=ios-9.0 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck -check-prefix=CHECK-WITH-EMPTY-COLLECTIONS %s

#include "literal-support.h"

void test_empty_array(void) {
  // CHECK-WITHOUT-EMPTY-COLLECTIONS-LABEL: define{{.*}} void @test_empty_array
  // CHECK-WITHOUT-EMPTY-COLLECTIONS-NOT: ret void
  // CHECK-WITHOUT-EMPTY-COLLECTIONS: {{call.*objc_msgSend}}
  // CHECK-WITHOUT-EMPTY-COLLECTIONS-NOT: ret void
  // CHECK-WITHOUT-EMPTY-COLLECTIONS: {{call.*llvm.objc.retainAutoreleasedReturnValue}}
  // CHECK-WITHOUT-EMPTY-COLLECTIONS: ret void

  // CHECK-WITH-EMPTY-COLLECTIONS-LABEL: define{{.*}} void @test_empty_array
  // CHECK-WITH-EMPTY-COLLECTIONS-NOT: ret void
  // CHECK-WITH-EMPTY-COLLECTIONS: load {{.*}} @__NSArray0__
  // CHECK-WITH-EMPTY-COLLECTIONS-NOT: ret void
  // CHECK-WITH-EMPTY-COLLECTIONS: {{call.*llvm.objc.retain\(}}
  // CHECK-WITH-EMPTY-COLLECTIONS-NOT: ret void
  // CHECK-WITH-EMPTY-COLLECTIONS: call void @llvm.objc.storeStrong
  // CHECK-WITH-EMPTY-COLLECTIONS-NEXT: ret void
  NSArray *arr = @[];
}

void test_empty_dictionary(void) {
  // CHECK-WITHOUT-EMPTY-COLLECTIONS-LABEL: define{{.*}} void @test_empty_dictionary
  // CHECK-WITHOUT-EMPTY-COLLECTIONS-NOT: ret void
  // CHECK-WITHOUT-EMPTY-COLLECTIONS: {{call.*objc_msgSend}}
  // CHECK-WITHOUT-EMPTY-COLLECTIONS-NOT: ret void
  // CHECK-WITHOUT-EMPTY-COLLECTIONS: {{call.*llvm.objc.retainAutoreleasedReturnValue}}
  // CHECK-WITHOUT-EMPTY-COLLECTIONS: ret void

  // CHECK-WITH-EMPTY-COLLECTIONS-LABEL: define{{.*}} void @test_empty_dictionary
  // CHECK-WITH-EMPTY-COLLECTIONS-NOT: ret void
  // CHECK-WITH-EMPTY-COLLECTIONS: load {{.*}} @__NSDictionary0__{{.*}}!invariant.load
  // CHECK-WITH-EMPTY-COLLECTIONS-NOT: ret void
  // CHECK-WITH-EMPTY-COLLECTIONS: {{call.*llvm.objc.retain\(}}
  // CHECK-WITH-EMPTY-COLLECTIONS-NOT: ret void
  // CHECK-WITH-EMPTY-COLLECTIONS: call void @llvm.objc.storeStrong
  // CHECK-WITH-EMPTY-COLLECTIONS-NEXT: ret void
  NSDictionary *dict = @{};
}
