// RUN: %strip_comments > %t.stripped.c
// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name builtinmacro.c %t.stripped.c | FileCheck %s

// Test the coverage mapping generation for built-in macroes.

// CHECK: filename
const char *filename (const char *name) { // CHECK-NEXT: File 0, [[@LINE]]:41 -> [[@LINE+3]]:2 = #0
  static const char this_file[] = __FILE__;
  return this_file;
}

int main() { // CHECK-NEXT: main
  filename(__FILE__ "test.c");
  return 0;
}
