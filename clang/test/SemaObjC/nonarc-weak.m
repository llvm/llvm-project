// RUN: %clang_cc1 -triple x86_64-apple-macosx10.8.0 -fobjc-runtime=macosx-10.8.0 -fsyntax-only -Wunused-function %s > %t.nonarc 2>&1
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.8.0 -fobjc-runtime=macosx-10.8.0 -fsyntax-only -Wunused-function -fobjc-arc %s > %t.arc 2>&1
// RUN: FileCheck -input-file=%t.nonarc %s
// RUN: FileCheck -input-file=%t.arc -check-prefix=ARC %s

static void bar() {} // Intentionally unused.

void foo(id self) {
  __weak id weakSelf = self;
}

// CHECK: 9:13: warning: __weak attribute cannot be specified on an automatic variable when ARC is not enabled
// CHECK: 6:13: warning: unused function 'bar'
// CHECK: 2 warnings generated
// ARC: 6:13: warning: unused function 'bar'
// ARC: 1 warning generated
