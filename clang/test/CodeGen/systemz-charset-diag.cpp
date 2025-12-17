// RUN: %clang_cc1 -triple s390x-none-zos -fexec-charset IBM-1047 %s -std=c++17 -emit-llvm -o - -verify

static_assert(false, "Error string"); // expected-error {{static assertion failed: Error string}}

[[deprecated("message")]] void test_deprecated() {return;} // expected-note {{'test_deprecated' has been explicitly marked deprecated here}}

int main() {
  test_deprecated(); // expected-warning {{'test_deprecated' is deprecated: message}}
}
