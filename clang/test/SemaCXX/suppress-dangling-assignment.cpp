
// RUN: %clang_cc1 -std=c++20 -verify -Wno-dangling-assignment %s
// expected-no-diagnostics

namespace std {
// std::basic_string has a hard-coded gsl::owner attr.
struct basic_string {
  const char* c_str();
};
}  // namespace std

void test(const char* a) {
  // verify that the dangling-assignment diagnostic are suppressed. 
  a = std::basic_string().c_str();
}
