// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -final-syntax-check -use-nullptr %t.cpp -- --std=c++11 -I %S
// RUN: FileCheck -input-file=%t.cpp %s

namespace std {

typedef decltype(nullptr) nullptr_t;

} // namespace std

// Just to make sure make_null() could have side effects.
void external();

std::nullptr_t make_null() {
  external();
  return nullptr;
}

void func() {
  void *CallTest = make_null();
  // CHECK: void *CallTest = make_null();

  int var = 1;
  void *CommaTest = (var+=2, make_null());
  // CHECK: void *CommaTest = (var+=2, make_null());

  int *CastTest = static_cast<int*>(make_null());
  // CHECK: int *CastTest = static_cast<int*>(make_null());
}
