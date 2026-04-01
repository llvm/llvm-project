
// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -debug-info-kind=limited -emit-llvm -disable-llvm-passes -o - | FileCheck %s

void fun() {
    // CHECK: %0 = alloca i8, i64 4{{.*}}, !dbg [[B1:!.*]]
    void *a = __builtin_alloca(4);

    // CHECK: %1 = alloca i8, i64 4{{.*}}, !dbg [[B2:!.*]]
    // Ensure calling same built-in twice only produces one `DISubprogram` entry
    void *b = __builtin_alloca(4);

    // CHECK: call void @llvm.memset{{.*}}, !dbg [[B3:!.*]]
    __builtin_memset(a, 0, 4);
}

// CHECK: [[B1]] = !DILocation(line: 0, scope: [[S1:!.*]], inlinedAt: [[I1:!.*]])
// CHECK: [[S1]] = distinct !DISubprogram(name: "__builtin_alloca"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I1]] = !DILocation(line: 6,

// Second call should reuse same `DISubprogram` scope
// CHECK: [[B2]] = !DILocation(line: 0, scope: [[S1:!.*]], inlinedAt: [[I2:!.*]])
// CHECK: [[I2]] = !DILocation(line: 10,

// CHECK: [[B3]] = !DILocation(line: 0, scope: [[S3:!.*]], inlinedAt: [[I3:!.*]])
// CHECK: [[S3]] = distinct !DISubprogram(name: "__builtin_memset"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I3]] = !DILocation(line: 13,
