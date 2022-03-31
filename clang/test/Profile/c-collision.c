// Test that a slight change in the code leads to a different hash.
// RUN: %clang_cc1 -UEXTRA -triple x86_64-unknown-linux-gnu -main-file-name c-collision.c %s -o - -emit-llvm -fprofile-instrument=clang | FileCheck %s --check-prefix=CHECK-NOEXTRA
// RUN: %clang_cc1 -DEXTRA -triple x86_64-unknown-linux-gnu -main-file-name c-collision.c %s -o - -emit-llvm -fprofile-instrument=clang | FileCheck %s --check-prefix=CHECK-EXTRA

// CHECK-NOEXTRA: @__profd_foo = private global { {{.*}} } { i64 6699318081062747564, i64 7156072912471487002,
// CHECK-EXTRA:   @__profd_foo = private global { {{.*}} } { i64 6699318081062747564, i64 -4383447408116050035,

extern int bar;
void foo(void) {
  if (bar) {
  }
  if (bar) {
  }
  if (bar) {
    if (bar) {
#ifdef EXTRA
      if (bar) {
      }
#endif
    }
  }
}
