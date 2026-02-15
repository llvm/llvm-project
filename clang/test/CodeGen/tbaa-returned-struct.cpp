// RUN: %clang_cc1 -triple x86_64-linux-unknown -emit-llvm -o - -O1 -disable-llvm-passes %s | FileCheck %s --check-prefixes=CHECK

// Checking that we generate TBAA metadata for returned aggregates.
// Currently, TBAA metadata is only emitted when structs are returned directly and the returned LLVM struct exactly matches the LLVM struct representation of the type.
// We should update this test when TBAA metadata is added for more cases. Cases which aren't covered include:
//  - Direct return as scalar (e.g. { int x; int y; } returned as i64)
//  - Indirect return via sret pointer

struct S1 {
    // Currently, only structs small enough to be returned directly, but large enough not to be returned as a scalar, will get TBAA metadata.
    long x;
    double y;
};

S1 returns_s1() {
    return S1 {1, 2};
}

void receives_s1() {
    S1 x = returns_s1();
// CHECK: define dso_local void @_Z11receives_s1v()
// CHECK: %call = call { i64, double } @_Z10returns_s1v()
// CHECK-NEXT: %0 = getelementptr inbounds nuw { i64, double }, ptr %x, i32 0, i32 0
// CHECK-NEXT: %1 = extractvalue { i64, double } %call, 0
// CHECK-NEXT: store i64 %1, ptr %0, align 8, !tbaa ![[TBAA_LONG_IN_S1:[0-9]+]]
// CHECK-NEXT: %2 = getelementptr inbounds nuw { i64, double }, ptr %x, i32 0, i32 1
// CHECK-NEXT: %3 = extractvalue { i64, double } %call, 1
// CHECK-NEXT: store double %3, ptr %2, align 8, !tbaa ![[TBAA_DOUBLE_IN_S1:[0-9]+]]
}

// Validate TBAA MD
// CHECK-DAG: ![[TBAA_CHAR:[0-9]+]] = !{!"omnipotent char",
// CHECK-DAG: ![[TBAA_LONG:[0-9]+]] = !{!"long", ![[TBAA_CHAR]], i64 0}
// CHECK-DAG: ![[TBAA_DOUBLE:[0-9]+]] = !{!"double", ![[TBAA_CHAR]], i64 0}
// CHECK-DAG: ![[TBAA_S1:[0-9]+]] = !{!"_ZTS2S1", ![[TBAA_LONG]], i64 0, ![[TBAA_DOUBLE]], i64 8}
// CHECK-DAG: ![[TBAA_LONG_IN_S1]] = !{![[TBAA_S1]], ![[TBAA_LONG]], i64 0}
// CHECK-DAG: ![[TBAA_DOUBLE_IN_S1]] = !{![[TBAA_S1]], ![[TBAA_DOUBLE]], i64 8}
