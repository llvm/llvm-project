// Make sure instrementation data from available_externally functions doesn't
// get thrown out.
// RUN: %clang_cc1 -O2 -triple x86_64-apple-macosx10.9 -main-file-name c-linkage-available_externally.c %s -o - -emit-llvm -fprofile-instr-generate | FileCheck %s

// CHECK: @__llvm_profile_name_foo = linkonce_odr hidden constant [3 x i8] c"foo", section "__DATA,__llvm_prf_names", align 1

// CHECK: @__llvm_profile_counters_foo = linkonce_odr hidden global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
// CHECK: @__llvm_profile_data_foo = linkonce_odr hidden constant { i32, i32, i64, i8*, i64* } { i32 3, i32 1, i64 {{[0-9]+}}, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__llvm_profile_name_foo, i32 0, i32 0), i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__llvm_profile_counters_foo, i32 0, i32 0) }, section "__DATA,__llvm_prf_data", align 8
inline int foo(void) { return 1; }

int main(void) {
  return foo();
}
