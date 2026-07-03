// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck --allow-empty %s < %t.out

// CHECK-NOT: TypeSanitizer

int foo() { return 0; }

struct Bar {
  struct S2 {
    int (*fnA)();
    int (*fnB)();
  };

  static int x() { return 0; }

  static const S2 &get() {
    static constexpr S2 Info = {&foo, &Bar::x};
    return Info;
  }
};

int main() {
  auto Info = Bar::get();
  return Info.fnB();
}
