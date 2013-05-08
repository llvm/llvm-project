// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -std=c++11
// RUN: FileCheck -input-file=%t.cpp %s
// XFAIL: *

// None of these tests can pass right now because TypeLoc information where CV
// qualifiers are concerned is not reliable/available.

class MyType {
};

int main (int argc, char **argv) {
  const MyType *d = new MyType();
  // CHECK: const auto *d = new MyType();

  volatile MyType *d2 = new MyType();
  // CHECK: volatile auto *d2 = new MyType();

  const MyType * volatile e = new MyType();
  // CHECK: const auto * volatile d = new MyType();

  volatile MyType * const f = new MyType();
  // CHECK: volatile auto * const d2 = new MyType();

  const MyType *d5 = new const MyType();
  // CHECK: auto d5 = new const MyType();

  volatile MyType *d6 = new volatile MyType();
  // CHECK: auto d6 = new volatile MyType();

  const MyType * const d7 = new const MyType();
  // CHECK: const auto d7 = new const MyType();

  volatile MyType * volatile d8 = new volatile MyType();
  // CHECK: volatile auto d8 = new volatile MyType();
}
