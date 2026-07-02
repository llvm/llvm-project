// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=cir.std.
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -DVOID_RESULT -fclangir -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=VOID --implicit-check-not=cir.std.

// Each call satisfies every recognizer check except the one guard it pins,
// and stays the call to the overload its comment names.

#ifdef VOID_RESULT

namespace std {
// std::find returns the iterator, so a result is required.
void find(char *first, char *last, const char &value);
}

void test_void_result(char *first, char *last, const char &value) {
  std::find(first, last, value);
}
// VOID-LABEL: @_Z16test_void_result
// VOID: cir.call @_ZSt4findPcS_RKc

#else

namespace std {
// Variadic, only viable for the all-pointer call in test_variadic.
char *find(char *first, ...);
// Result type differs from the iterator type.
int find(char *first, char *last, const char &value);
// Searched value type differs from the element type.
char *find(char *first, char *last, const int &value);
// Wrong arity.
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

char *test_arity(char *first, char *last, const char &value) {
  return std::find(first, last, value, 1);
}
// CHECK-LABEL: @_Z10test_arity
// CHECK: cir.call @_ZSt4findPcS_RKci

#endif
