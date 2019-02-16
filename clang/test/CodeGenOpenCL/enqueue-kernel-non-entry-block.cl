// RUN: %clang_cc1 -cl-std=CL2.0 -O0 -emit-llvm -o - -triple amdgcn < %s | FileCheck %s --check-prefixes=COMMON,AMDGPU
// RUN: %clang_cc1 -cl-std=CL2.0 -O0 -emit-llvm -o - -triple "spir-unknown-unknown" < %s | FileCheck %s --check-prefixes=COMMON,SPIR32
// RUN: %clang_cc1 -cl-std=CL2.0 -O0 -emit-llvm -o - -triple "spir64-unknown-unknown" < %s | FileCheck %s --check-prefixes=COMMON,SPIR64
// RUN: %clang_cc1 -cl-std=CL2.0 -O0 -debug-info-kind=limited -emit-llvm -o - -triple amdgcn < %s | FileCheck %s --check-prefixes=CHECK-DEBUG

// Check that the enqueue_kernel array temporary is in the entry block to avoid
// a dynamic alloca

typedef struct {int a;} ndrange_t;

kernel void test(int i) {
// COMMON-LABEL: define {{.*}} void @test
// COMMON-LABEL: entry:
// AMDGPU: %block_sizes = alloca [1 x i64]
// SPIR32: %block_sizes = alloca [1 x i32]
// SPIR64: %block_sizes = alloca [1 x i64]
// COMMON-LABEL: if.then:
// COMMON-NOT: alloca
// CHECK-DEBUG: getelementptr {{.*}} %block_sizes, {{.*}} !dbg ![[TEMPLOCATION:[0-9]+]]
// COMMON-LABEL: if.end
  queue_t default_queue;
  unsigned flags = 0;
  ndrange_t ndrange;
  if (i)
    enqueue_kernel(default_queue, flags, ndrange, ^(local void *a) { }, 32);
}

// Check that the temporary is scoped to the `if`

// CHECK-DEBUG: ![[TESTFILE:[0-9]+]] = !DIFile(filename: "<stdin>"
// CHECK-DEBUG: ![[TESTSCOPE:[0-9]+]] = distinct !DISubprogram(name: "test", {{.*}} file: ![[TESTFILE]]
// CHECK-DEBUG: ![[IFSCOPE:[0-9]+]] = distinct !DILexicalBlock(scope: ![[TESTSCOPE]], file: ![[TESTFILE]], line: 24)
// CHECK-DEBUG: ![[TEMPLOCATION]] = !DILocation(line: 25, scope: ![[IFSCOPE]])
