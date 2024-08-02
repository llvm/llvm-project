// RUN: %clang_analyze_cc1 -verify=common %s \
// RUN:   -analyzer-checker=deadcode.DeadStores,debug.ExprInspection \
// RUN:   -analyzer-note-analysis-entry-points

// RUN: %clang_analyze_cc1 -verify=common,textout %s \
// RUN:   -analyzer-checker=deadcode.DeadStores,debug.ExprInspection \
// RUN:   -analyzer-note-analysis-entry-points \
// RUN:   -analyzer-output=text

// Test the actual source locations/ranges of entry point notes.
// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=deadcode.DeadStores,debug.ExprInspection \
// RUN:   -analyzer-note-analysis-entry-points \
// RUN:   -analyzer-output=text 2>&1 \
// RUN: | FileCheck --strict-whitespace %s


void clang_analyzer_warnIfReached();

void other() {
  // common-warning@+1 {{REACHABLE}} textout-note@+1 {{REACHABLE}}
  clang_analyzer_warnIfReached();
}

struct SomeOtherStruct {
  // CHECK: note: [debug] analyzing from SomeOtherStruct::f()
  // CHECK-NEXT: |   void f() {
  // CHECK-NEXT: |        ^
  // textout-note@+1 {{[debug] analyzing from SomeOtherStruct::f()}}
  void f() {
    other(); // textout-note {{Calling 'other'}}
  }
};

// CHECK: note: [debug] analyzing from operator""_w(const char *)
// CHECK-NEXT: | unsigned operator ""_w(const char*) {
// CHECK-NEXT: |          ^
// textout-note@+1 {{[debug] analyzing from operator""_w(const char *)}}
unsigned operator ""_w(const char*) {
  // common-warning@+1 {{REACHABLE}} textout-note@+1 {{REACHABLE}}
  clang_analyzer_warnIfReached();
  return 404;
}

// textout-note@+1 {{[debug] analyzing from checkASTCodeBodyHasAnalysisEntryPoints()}}
void checkASTCodeBodyHasAnalysisEntryPoints() {
  int z = 1;
  z = 2;
  // common-warning@-1 {{Value stored to 'z' is never read}}
  // textout-note@-2    {{Value stored to 'z' is never read}}
}

void notInvokedLambdaScope() {
  // CHECK: note: [debug] analyzing from notInvokedLambdaScope()::(anonymous class)::operator()()
  // CHECK-NEXT: |   auto notInvokedLambda = []() {
  // CHECK-NEXT: |                           ^
  // textout-note@+1 {{[debug] analyzing from notInvokedLambdaScope()::(anonymous class)::operator()()}}
  auto notInvokedLambda = []() {
    // common-warning@+1 {{REACHABLE}} textout-note@+1 {{REACHABLE}}
    clang_analyzer_warnIfReached();
  };
  (void)notInvokedLambda; // Not invoking the lambda.
}

// CHECK: note: [debug] analyzing from invokedLambdaScope()
// CHECK-NEXT: | void invokedLambdaScope() {
// CHECK-NEXT: |      ^
// textout-note@+1 {{[debug] analyzing from invokedLambdaScope()}}
void invokedLambdaScope() {
  auto invokedLambda = []() {
    // common-warning@+1 {{REACHABLE}} textout-note@+1 {{REACHABLE}}
    clang_analyzer_warnIfReached();
  };
  invokedLambda(); // textout-note {{Calling 'operator()'}}
}