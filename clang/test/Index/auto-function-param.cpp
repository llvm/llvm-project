// Test case for auto function parameter reported as CXType_Auto
// This test verifies that auto parameters in function declarations
// are properly reported as CXType_Auto in the libclang C API
// See issue #172072

// RUN: c-index-test -test-type %s | FileCheck %s

// Function with auto parameter
int bar(auto p) {
  return p;
}

// CHECK: FunctionDecl=bar:{{.*}} CXType_FunctionProto
// CHECK: ParmDecl=p:{{.*}} CXType_Auto
// C++20 constrained auto parameter
template<typename T>
concept Integral = __is_integral(T);

int baz(Integral auto q) {
  return q;
}

// CHECK: FunctionDecl=baz:{{.*}} CXType_FunctionProto
// CHECK: ParmDecl=q:{{.*}} CXType_Auto