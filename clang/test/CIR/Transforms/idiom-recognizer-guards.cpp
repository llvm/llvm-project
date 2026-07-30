// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=cir.std.

namespace std {
// Variadic, only viable for the all-pointer call in test_variadic.
char *find(char *first, ...);
// Result type differs from the iterator type.
int find(char *first, char *last, const char &value);
// Searched value type differs from the element type.
char *find(char *first, char *last, const int &value);
// Wrong argument count. The C++17 std::find overload taking an ExecutionPolicy
// is also declined here, since it too differs from the recognized three
// argument shape.
char *find(char *first, char *last, const char &value, int n);
}

char *test_variadic(char *first, char *last, char *value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z13test_variadic
// CHECK: cir.call @_ZSt4findPcz

int test_result_type(char *first, char *last, const char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z16test_result_type
// CHECK: cir.call @_ZSt4findPcS_RKc

char *test_pattern_type(char *first, char *last, const int &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z17test_pattern_type
// CHECK: cir.call @_ZSt4findPcS_RKi

char *test_wrong_arg_count(char *first, char *last, const char &value) {
  return std::find(first, last, value, 1);
}
// CHECK-LABEL: @_Z20test_wrong_arg_count
// CHECK: cir.call @_ZSt4findPcS_RKci

// std membership is fixed when the tag is set, so a find outside std is never
// tagged and never raised, whatever the call shape.

// A nested namespace that is not inline is not std.
namespace std {
namespace another_ns {
template <class Iter, class T>
Iter find(Iter, Iter, const T &);
}
}

char *test_nested_namespace(char *first, char *last, const char &value) {
  return std::another_ns::find(first, last, value);
}
// CHECK-LABEL: @_Z21test_nested_namespace
// CHECK: cir.call

// An anonymous namespace function is not std::find.
namespace {
template <class Iter, class T>
Iter find(Iter, Iter, const T &);
}

char *test_anonymous_namespace(char *first, char *last, const char &value) {
  return find(first, last, value);
}
// CHECK-LABEL: @_Z24test_anonymous_namespace
// CHECK: cir.call
