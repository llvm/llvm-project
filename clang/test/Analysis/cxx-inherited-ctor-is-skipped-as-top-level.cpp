// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-display-progress %s 2>&1 | FileCheck %s

// Test that inheriting constructors are not analyzed as top-level functions.

// CHECK: ANALYZE (Path,  Inline_Regular): {{.*}} c()
// CHECK: ANALYZE (Path,  Inline_Regular): {{.*}} a::a(int)
// CHECK-NOT: ANALYZE (Path,  Inline_Regular): {{.*}} b::a(int)

class a {
public:
  a(int) {}
};
struct b : a {
  using a::a; // Ihnerited ctor.
};
void c() {
  int d;
  (b(d));
  (a(d));
}
