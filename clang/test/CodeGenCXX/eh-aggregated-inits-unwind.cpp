// Check that destructors of memcpy-able struct members are called properly
// during stack unwinding after an exception.
//
// Check that destructor's argument (address of member to be destroyed) is
// obtained by taking offset from struct, not by bitcasting pointers.
//
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -O0 -fno-elide-constructors -emit-llvm %s -o - | FileCheck %s

struct ImplicitCopy {
  int id;
  ImplicitCopy() { id = 10; }
  ~ImplicitCopy() { id = 20; }
};

struct ThrowCopy {
  int id;
  ThrowCopy() { id = 15; }
  ThrowCopy(const ThrowCopy &x) {
    id = 25;
    throw 1;
  }
  ~ThrowCopy() { id = 35; }
};

struct Container {
  int id;
  ImplicitCopy o1;
  ThrowCopy o2;

  Container() { id = 1000; }
  ~Container() { id = 2000; }
};

int main() {
  try {
    Container c1;
    // CHECK-LABEL: main
    // CHECK: %{{.+}} = getelementptr inbounds nuw %struct.Container, ptr %{{.+}}, i32 0, i32 1
    Container c2(c1);

    return 2;
  } catch (...) {
    return 1;
  }
  return 0;
}
