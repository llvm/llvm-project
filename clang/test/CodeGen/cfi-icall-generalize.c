// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=UNGENERALIZED %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -fsanitize-cfi-icall-generalize-pointers -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=GENERALIZED %s

// Test that const char* is generalized to const void* and that const char** is
// generalized to void*

// CHECK: define i32** @f({{.*}} !type [[TYPE:![0-9]+]] !type [[TYPE_GENERALIZED:![0-9]+]]
int** f(const char *a, const char **b) {
  return (int**)0;
}

void g(int** (*fp)(const char *, const char **)) {
  // UNGENERALIZED: call i1 @llvm.type.test(i8* {{.*}}, metadata !"_ZTSFPPiPKcPS2_E")
  // GENERALIZED: call i1 @llvm.type.test(i8* {{.*}}, metadata !"_ZTSFPvPKvS_E.generalized")
  fp(0, 0);
}

// CHECK: [[TYPE]] = !{i64 0, !"_ZTSFPPiPKcPS2_E"}
// CHECK: [[TYPE_GENERALIZED]] = !{i64 0, !"_ZTSFPvPKvS_E.generalized"}
