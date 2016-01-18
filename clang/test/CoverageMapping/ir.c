// Check the data structures emitted by coverage mapping
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name ir.c %s -o - -emit-llvm -fprofile-instr-generate -fcoverage-mapping | FileCheck %s


void foo(void) { }

int main(void) {
  foo();
  return 0;
}

// CHECK: @__llvm_coverage_mapping = internal constant { i32, i32, i32, i32, [2 x <{ i8*, i32, i32, i64 }>], [{{[0-9]+}} x i8] } { i32 2, i32 {{[0-9]+}}, i32 {{[0-9]+}}, i32 0, [2 x <{ i8*, i32, i32, i64 }>] [<{ i8*, i32, i32, i64 }> <{ i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__llvm_profile_name_foo, i32 0, i32 0), i32 3, i32 9, i64 {{[0-9]+}} }>, <{ i8*, i32, i32, i64 }> <{ i8* getelementptr inbounds ([4 x i8], [4 x i8]* @__llvm_profile_name_main, i32 0, i32 0), i32 4, i32 9, i64 {{[0-9]+}} }>]
