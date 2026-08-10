// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=thread -relaxed-aliasing -O1 | FileCheck %s

// Make sure we do not crash when relaxed-aliasing is on.
// CHECK-NOT: !tbaa
struct iterator { void *node; };

struct pair {
  iterator first;
  pair(const iterator &a) : first(a) {}
};

void equal_range() {
  (void)pair(iterator());
}
