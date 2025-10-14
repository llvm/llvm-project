// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=UNGENERALIZED %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -fsanitize-cfi-icall-generalize-pointers -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=GENERALIZED %s

// Test that const char* is generalized to const ptr and that char** is
// generalized to ptr

// CHECK: define{{.*}} ptr @f({{.*}} !type [[TYPE:![0-9]+]] !type [[TYPE_GENERALIZED:![0-9]+]]
int** f(const char *a, const char **b) {
  return (int**)0;
}

void g(int** (*fp)(const char *, const char **)) {
  // UNGENERALIZED: call i1 @llvm.type.test(ptr {{.*}}, metadata !"_ZTSFPPiPKcPS2_E")
  // GENERALIZED: call i1 @llvm.type.test(ptr {{.*}}, metadata !"_ZTSFPvPKvS_E.generalized")
  fp(0, 0);
}

union Union {
  char *c;
  long *n;
} __attribute__((transparent_union));

// CHECK: define{{.*}} void @uni({{.*}} !type [[TYPE2:![0-9]+]] !type [[TYPE2_GENERALIZED:![0-9]+]]
void uni(void (*fn)(union Union), union Union arg1) {
  // UNGENERALIZED: call i1 @llvm.type.test(ptr {{.*}}, metadata !"_ZTSFvPcE")
  // GENERALIZED: call i1 @llvm.type.test(ptr {{.*}}, metadata !"_ZTSFvPvE.generalized")
    fn(arg1);
}

// CHECK: [[TYPE]] = !{i64 0, !"_ZTSFPPiPKcPS2_E"}
// CHECK: [[TYPE_GENERALIZED]] = !{i64 0, !"_ZTSFPvPKvS_E.generalized"}

// CHECK: [[TYPE2]] = !{i64 0, !"_ZTSFvPFv5UnionEPcE"}
// CHECK: [[TYPE2_GENERALIZED]] = !{i64 0, !"_ZTSFvPvS_E.generalized"}

