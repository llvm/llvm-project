// RUN: %clang_hwasan -fsanitize=cfi -fno-sanitize-trap=cfi -flto -fvisibility=hidden -fuse-ld=lld %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// REQUIRES: android

// Smoke test for CFI + HWASAN.

struct A {
  virtual void f();
};

void A::f() {}

int main() {
  // CHECK: control flow integrity check for type {{.*}} failed during cast to unrelated type
  A *a = reinterpret_cast<A *>(reinterpret_cast<void *>(&main));
  (void)a;
}
