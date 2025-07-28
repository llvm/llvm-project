// Test profiling functionality with single file
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "[{\"directory\":\".\",\"command\":\"clang++ -c %/t/test.cpp\",\"file\":\"%/t/test.cpp\"}]" | sed -e 's/\\/\\\\/g' > %t/compile_commands.json
// RUN: echo "Checks: '-*,readability-function-size'" > %t/.clang-tidy
// RUN: cp "%s" "%t/test.cpp"
// RUN: cd "%t"
// RUN: %run_clang_tidy -enable-check-profile "test.cpp" 2>&1 | FileCheck %s --check-prefix=CHECK-SINGLE

// CHECK-SINGLE: Running clang-tidy in {{[1-9][0-9]*}} threads for 1 files out of 1 in compilation database
// CHECK-SINGLE: ===-------------------------------------------------------------------------===
// CHECK-SINGLE-NEXT:                          clang-tidy checks profiling
// CHECK-SINGLE-NEXT: ===-------------------------------------------------------------------------===
// CHECK-SINGLE-NEXT: Total Execution Time: {{.*}} seconds ({{.*}} wall clock)
// CHECK-SINGLE: {{.*}}  --- Name ---
// CHECK-SINGLE-NEXT: {{.*}}  readability-function-size
// CHECK-SINGLE-NEXT: {{.*}}  Total

// Test profiling functionality with multiple files and multiple checks
// RUN: rm -rf %t-multi
// RUN: mkdir %t-multi
// RUN: echo "[{\"directory\":\".\",\"command\":\"clang++ -c %/t-multi/test1.cpp\",\"file\":\"%/t-multi/test1.cpp\"},{\"directory\":\".\",\"command\":\"clang++ -c %/t-multi/test2.cpp\",\"file\":\"%/t-multi/test2.cpp\"}]" | sed -e 's/\\/\\\\/g' > %t-multi/compile_commands.json
// RUN: echo "Checks: '-*,readability-function-size,misc-unused-using-decls,llvm-qualified-auto'" > %t-multi/.clang-tidy
// RUN: cp "%s" "%t-multi/test1.cpp"
// RUN: cp "%s" "%t-multi/test2.cpp"
// RUN: cd "%t-multi"
// RUN: %run_clang_tidy -enable-check-profile -j 2 "test1.cpp" "test2.cpp" 2>&1 | FileCheck %s --check-prefix=CHECK-MULTIPLE

// CHECK-MULTIPLE: Running clang-tidy in 2 threads for 2 files out of 2 in compilation database
// CHECK-MULTIPLE: ===-------------------------------------------------------------------------===
// CHECK-MULTIPLE-NEXT:                          clang-tidy checks profiling
// CHECK-MULTIPLE-NEXT: ===-------------------------------------------------------------------------===
// CHECK-MULTIPLE-NEXT: Total Execution Time: {{.*}} seconds ({{.*}} wall clock)
// CHECK-MULTIPLE: {{.*}}  --- Name ---
// CHECK-MULTIPLE: {{.*}}  readability-function-size
// CHECK-MULTIPLE: {{.*}}  misc-unused-using-decls
// CHECK-MULTIPLE: {{.*}}  llvm-qualified-auto
// CHECK-MULTIPLE: {{.*}}  Total

// Test profiling functionality with no files (empty database)
// RUN: rm -rf %t-empty
// RUN: mkdir %t-empty
// RUN: echo "[]" > %t-empty/compile_commands.json
// RUN: echo "Checks: '-*'" > %t-empty/.clang-tidy
// RUN: cd "%t-empty"
// RUN: %run_clang_tidy -enable-check-profile -allow-no-checks 2>&1 | FileCheck %s --check-prefix=CHECK-EMPTY

// CHECK-EMPTY: Running clang-tidy in {{[1-9][0-9]*}} threads for 0 files out of 0 in compilation database
// CHECK-EMPTY: No profiling data found.

class A {
  A() {}
  ~A() {}
};
