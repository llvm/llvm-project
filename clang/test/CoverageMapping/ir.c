// Check the data structures emitted by coverage mapping
// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -triple x86_64-apple-macosx10.9 -main-file-name ir.c %s -o - -emit-llvm -fprofile-instrument=clang -fcoverage-mapping -mllvm -enable-name-compression=false -no-opaque-pointers | FileCheck %s -check-prefixes=COMMON,DARWIN
// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -triple x86_64-apple-macosx10.9 -main-file-name ir.c %s -o - -emit-llvm -fprofile-instrument=clang -fcoverage-mapping -mllvm -enable-name-compression=false -opaque-pointers | FileCheck %s -check-prefixes=COMMON,DARWIN
// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -triple x86_64--windows-msvc -main-file-name ir.c %s -o - -emit-llvm -fprofile-instrument=clang -fcoverage-mapping -mllvm -enable-name-compression=false | FileCheck %s -check-prefixes=COMMON,WINDOWS

static inline void unused(void) {}

void foo(void) {}

int main(void) {
  foo();
  return 0;
}

// Check the function records. Two of the record names should come in the 'used'
// flavor, and one should not.

// DARWIN: [[FuncRecord1:@__covrec_[0-9A-F]+u]] = linkonce_odr hidden constant <{ i64, i32, i64, i64, [{{.*}} x i8] }> <{ {{.*}} }>, section "__LLVM_COV,__llvm_covfun", align 8
// DARWIN: [[FuncRecord2:@__covrec_[0-9A-F]+u]] = linkonce_odr hidden constant <{ i64, i32, i64, i64, [{{.*}} x i8] }> <{ {{.*}} }>, section "__LLVM_COV,__llvm_covfun", align 8
// DARWIN: [[FuncRecord3:@__covrec_[0-9A-F]+]] = linkonce_odr hidden constant <{ i64, i32, i64, i64, [{{.*}} x i8] }> <{ {{.*}} }>, section "__LLVM_COV,__llvm_covfun", align 8
// DARWIN: @__llvm_coverage_mapping = private constant { { i32, i32, i32, i32 }, [{{.*}} x i8] } { {{.*}} }, section "__LLVM_COV,__llvm_covmap", align 8

// WINDOWS: [[FuncRecord1:@__covrec_[0-9A-F]+u]] = linkonce_odr hidden constant <{ i64, i32, i64, i64, [{{.*}} x i8] }> <{ {{.*}} }>, section ".lcovfun$M", comdat, align 8
// WINDOWS: [[FuncRecord2:@__covrec_[0-9A-F]+u]] = linkonce_odr hidden constant <{ i64, i32, i64, i64, [{{.*}} x i8] }> <{ {{.*}} }>, section ".lcovfun$M", comdat, align 8
// WINDOWS: [[FuncRecord3:@__covrec_[0-9A-F]+]] = linkonce_odr hidden constant <{ i64, i32, i64, i64, [{{.*}} x i8] }> <{ {{.*}} }>, section ".lcovfun$M", comdat, align 8
// WINDOWS: @__llvm_coverage_mapping = private constant { { i32, i32, i32, i32 }, [{{.*}} x i8] } { {{.*}} }, section ".lcovmap$M", align 8

// COMMON: @llvm.used = appending global [{{.*}}] [
// COMMON-SAME: [[FuncRecord1]]
// COMMON-SAME: [[FuncRecord2]]
// COMMON-SAME: [[FuncRecord3]]
// COMMON-SAME: @__llvm_coverage_mapping
