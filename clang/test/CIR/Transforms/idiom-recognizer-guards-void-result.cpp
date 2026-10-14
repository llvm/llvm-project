// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=cir.std.

// std::find returns the iterator, so a result is required. A void result
// fails the recognizer's result check and the call stays.

namespace std {
void find(char *first, char *last, const char &value);
}

void test_void_result(char *first, char *last, const char &value) {
  std::find(first, last, value);
}
// CHECK-LABEL: @_Z16test_void_result
// CHECK: cir.call @_ZSt4findPcS_RKc
