// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -mllvm -enable-name-compression=false -no-unused-coverage -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -main-file-name unused_names.c -o - %s > %t
// RUN: FileCheck -input-file %t %s

// There should only be a prf_names entry for bar, as the other two functions are
// unused.
//
// CHECK-DAG: @__llvm_prf_nm = private constant [5 x i8] c"\03\00bar", section "{{.*__llvm_prf_names|\.lprfn\$M}}"

int bar(void) { return 0; }
inline int baz(void) { return 0; }
static int qux(void) { return 42; }
