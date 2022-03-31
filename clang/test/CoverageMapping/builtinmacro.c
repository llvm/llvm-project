// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name builtinmacro.c %s | FileCheck %s

// Test the coverage mapping generation for built-in macroes.

// CHECK: filename
const char *filename (const char *name) { // CHECK-NEXT: File 0, [[@LINE]]:41 -> [[@LINE+3]]:2 = #0
  static const char this_file[] = __FILE__;
  return this_file;
}

int main(void) { // CHECK-NEXT: main
  filename(__FILE__ "test.c");
  return 0;
}
