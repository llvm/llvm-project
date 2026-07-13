// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=cir.std.

// std membership is fixed when the tag is set, so a find outside std is never
// tagged and never raised.

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
