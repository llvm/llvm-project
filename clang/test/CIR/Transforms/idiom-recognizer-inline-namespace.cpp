// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o /dev/null 2>&1 | FileCheck %s

// The versioning namespace of libc++ is inline, so std::find resolves through
// it and the call is still tagged and raised.

namespace std {
inline namespace __1 {
template <class Iter, class T>
Iter find(Iter, Iter, const T &);
}
}

char *test_inline_namespace(char *first, char *last, const char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z21test_inline_namespace
// CHECK: cir.std.find
