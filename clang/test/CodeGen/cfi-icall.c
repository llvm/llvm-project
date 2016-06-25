// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=ITANIUM %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=MS %s

// Tests that we assign appropriate identifiers to unprototyped functions.

// CHECK: define void @f({{.*}} !type [[TVOID:![0-9]+]]
void f() {
}

void xf();

// CHECK: define void @g({{.*}} !type [[TINT:![0-9]+]]
void g(int b) {
  void (*fp)() = b ? f : xf;
  // ITANIUM: call i1 @llvm.type.test(i8* {{.*}}, metadata !"_ZTSFvE")
  fp();
}

// CHECK: declare !type [[TVOID:![0-9]+]] void @xf({{.*}}

// ITANIUM-DAG: [[TVOID]] = !{i64 0, !"_ZTSFvE"}
// ITANIUM-DAG: [[TINT]] = !{i64 0, !"_ZTSFviE"}
// MS-DAG: [[TVOID]] = !{i64 0, !"?6AX@Z"}
// MS-DAG: [[TINT]] = !{i64 0, !"?6AXH@Z"}
