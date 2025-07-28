// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.StoreToImmutable %s

// Global const variable
const int global_const = 42;

void test_global_const() {
  *(int *)&global_const = 100; // warn: Writing to immutable memory
}

// String literal
// NOTE: This only works in C++, not in C, as the analyzer treats string literals as non-const char arrays in C mode.
void test_string_literal() {
  char *str = (char *)"hello";
  str[0] = 'H'; // warn: Writing to immutable memory
}

// Const parameter
void test_const_param(const int param) {
  *(int *)&param = 100; // warn: Writing to immutable memory
}

// Const struct member
struct TestStruct {
  const int x;
  int y;
};

void test_const_member() {
  TestStruct s = {1, 2};
  *(int *)&s.x = 10; // warn: Writing to immutable memory
}