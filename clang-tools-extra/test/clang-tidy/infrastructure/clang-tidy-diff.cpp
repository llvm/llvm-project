// UNSUPPORTED: system-windows
// RUN: sed 's/placeholder_for_f/f/' %s > %t.cpp
// RUN: clang-tidy -checks=-*,modernize-use-override %t.cpp -- -std=c++11 | FileCheck -check-prefix=CHECK-SANITY %s
// RUN: not diff -U0 %s %t.cpp | %clang_tidy_diff -checks=-*,modernize-use-override -- -std=c++11 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-JMAX
// RUN: not diff -U0 %s %t.cpp | %clang_tidy_diff -checks=-*,modernize-use-override -quiet -- -std=c++11 2>&1 | FileCheck -check-prefix=CHECK-QUIET %s
// RUN: mkdir -p %t.dir/compilation-database-test/
// RUN: echo '[{"directory": "%t.dir", "command": "clang++ -o test.o -std=c++11 %t.cpp", "file": "%t.cpp"}]' > %t.dir/compilation-database-test/compile_commands.json
// RUN: not diff -U0 %s %t.cpp | %clang_tidy_diff -checks=-*,modernize-use-override -path %t.dir/compilation-database-test 2>&1 | FileCheck -check-prefix=CHECK %s

// RUN: not diff -U0 %s %t.cpp | %clang_tidy_diff -checks=-*,modernize-use-override -j 1 -- -std=c++11 2>&1 | FileCheck %s --check-prefix=CHECK-J1
// CHECK-J1: Running clang-tidy in 1 threads...
struct A {
  virtual void f() {}
  virtual void g() {}
};
// CHECK-JMAX: Running clang-tidy in {{[1-9][0-9]*}} threads...
// CHECK-NOT: warning:
// CHECK-QUIET-NOT: warning:
struct B : public A {
  void placeholder_for_f() {}
// CHECK-SANITY: [[@LINE-1]]:8: warning: annotate this
// CHECK: [[@LINE-2]]:8: warning: annotate this
// CHECK-QUIET: [[@LINE-3]]:8: warning: annotate this
  void g() {}
// CHECK-SANITY: [[@LINE-1]]:8: warning: annotate this
// CHECK-NOT: warning:
// CHECK-QUIET-NOT: warning:
};
// CHECK-SANITY-NOT: Suppressed
// CHECK-QUIET-NOT: Suppressed
