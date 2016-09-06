// Check the data structures emitted by coverage mapping
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name ir.c %s -o - -emit-llvm -fprofile-instrument=clang -fcoverage-mapping | FileCheck %s


void foo(void) { }

int main(void) {
  foo();
  return 0;
}

// CHECK: @__llvm_coverage_mapping = internal constant { { i32, i32, i32, i32 }, [2 x <{ i64, i32, i64 }>], [{{[0-9]+}} x i8] } { { i32, i32, i32, i32 } { i32 2, i32 {{[0-9]+}}, i32 {{[0-9]+}}, i32 {{[0-9]+}} }, [2 x <{ i64, i32, i64 }>] [<{{.*}}> <{{.*}}>, <{{.*}}> <{{.*}}>]
