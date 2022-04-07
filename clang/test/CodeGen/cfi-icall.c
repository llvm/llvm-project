// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=ITANIUM %s
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-pc-windows-msvc -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=MS %s

// Tests that we assign appropriate identifiers to unprototyped functions.

// CHECK: define {{(dso_local )?}}void @f({{.*}} !type [[TVOID:![0-9]+]] !type [[TVOID_GENERALIZED:![0-9]+]]
void f() {
}

void xf();

// CHECK: define {{(dso_local )?}}void @g({{.*}} !type [[TINT:![0-9]+]] !type [[TINT_GENERALIZED:![0-9]+]]
void g(int b) {
  void (*fp)() = b ? f : xf;
  // ITANIUM: call i1 @llvm.type.test(i8* {{.*}}, metadata !"_ZTSFvE")
  fp();
}

// CHECK: declare !type [[TVOID]] !type [[TVOID_GENERALIZED]] {{(dso_local )?}}void @xf({{.*}}

// ITANIUM-DAG: [[TVOID]] = !{i64 0, !"_ZTSFvE"}
// ITANIUM-DAG: [[TVOID_GENERALIZED]] = !{i64 0, !"_ZTSFvE.generalized"}
// ITANIUM-DAG: [[TINT]] = !{i64 0, !"_ZTSFviE"}
// ITANIUM-DAG: [[TINT_GENERALIZED]] = !{i64 0, !"_ZTSFviE.generalized"}
// MS-DAG: [[TVOID]] = !{i64 0, !"?6AX@Z"}
// MS-DAG: [[TVOID_GENERALIZED]] = !{i64 0, !"?6AX@Z.generalized"}
// MS-DAG: [[TINT]] = !{i64 0, !"?6AXH@Z"}
// MS-DAG: [[TINT_GENERALIZED]] = !{i64 0, !"?6AXH@Z.generalized"}
