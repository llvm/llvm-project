
// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -debug-info-kind=limited -emit-llvm -disable-llvm-passes -o - | FileCheck %s

void fun() {
    // Most call-like built-ins are wrapped in an artificial inlined function in
    // debug info, making them visible to e.g. profiling tools.
    // CHECK: %0 = alloca i8, i64 4{{.*}}, !dbg [[B1:!.*]]
    void *a = __builtin_alloca(4);

    // Ensure calling same built-in only produces one `DISubprogram` entry.
    // CHECK: %1 = alloca i8, i64 4{{.*}}, !dbg [[B2:!.*]]
    void *b = __builtin_alloca(4);

    // CHECK: call void @llvm.memset{{.*}}, !dbg [[B3:!.*]]
    __builtin_memset(a, 0, 4);

    // Ensure certain built-ins like optimisation hints are excluded.
    // CHECK: call void @llvm.assume{{.*}}, !dbg [[B4:!.*]]
    __builtin_assume(a != 0);

    // Ensure target-specific built-ins are excluded.
    // CHECK: call i64 @llvm.x86.rdtsc(), !dbg [[B5:!.*]]
    __builtin_ia32_rdtsc();

    // Ensure library functions emitted as normal calls are excluded.
    // CHECK: call ptr @malloc{{.*}}, !dbg [[B6:!.*]]
    void *c = __builtin_malloc(4);
}

// CHECK: [[B1]] = !DILocation(line: 0, scope: [[S1:!.*]], inlinedAt: [[I1:!.*]])
// CHECK: [[S1]] = distinct !DISubprogram(name: "__builtin_alloca"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I1]] = !DILocation(line: 8,

// Second call should reuse same `DISubprogram` scope.
// CHECK: [[B2]] = !DILocation(line: 0, scope: [[S1:!.*]], inlinedAt: [[I2:!.*]])
// CHECK: [[I2]] = !DILocation(line: 12,

// CHECK: [[B3]] = !DILocation(line: 0, scope: [[S3:!.*]], inlinedAt: [[I3:!.*]])
// CHECK: [[S3]] = distinct !DISubprogram(name: "__builtin_memset"{{.*}}, flags: DIFlagArtificial
// CHECK: [[I3]] = !DILocation(line: 15,

// Excluded built-ins should use location without inlined function wrapper.

// CHECK: [[B4]] = !DILocation(line: 19,
// CHECK-NOT: distinct !DISubprogram(name: "__builtin_assume"{{.*}}, flags: DIFlagArtificial

// CHECK: [[B5]] = !DILocation(line: 23,
// CHECK-NOT: distinct !DISubprogram(name: "__builtin_ia32_rdtsc"{{.*}}, flags: DIFlagArtificial

// CHECK: [[B6]] = !DILocation(line: 27,
// CHECK-NOT: distinct !DISubprogram(name: "__builtin_malloc"{{.*}}, flags: DIFlagArtificial
