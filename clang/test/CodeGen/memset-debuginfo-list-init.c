// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

void use(void *);

// CHECK-LABEL: define {{.*}}void @test_memset_zero_init()
void test_memset_zero_init(void) {
  // CHECK: call void @llvm.memset.{{.*}} !dbg ![[LOC_MEMSET:[0-9]+]]
  char array[24] = { 0 };
  use(array);
}

// CHECK-LABEL: define {{.*}}void @test_memcpy_init()
void test_memcpy_init(void) {
  // CHECK: call void @llvm.memcpy.{{.*}} !dbg ![[LOC_MEMCPY:[0-9]+]]
  char array[24] = { 1, 2, 3 };
  use(array);
}

struct Big {
  int arr[20];
};

// CHECK-LABEL: define {{.*}}void @test_agg_memset()
void test_agg_memset(void) {
  // CHECK: call void @llvm.memset.{{.*}} !dbg ![[LOC_AGG:[0-9]+]]
  struct Big b = { 0 };
  use(&b);
}

// CHECK-DAG: ![[LOC_MEMSET]] = !DILocation(line: 0, scope: ![[SP_MEMSET:[0-9]+]], inlinedAt: ![[INLINED_MEMSET:[0-9]+]])
// CHECK-DAG: ![[SP_MEMSET]] = distinct !DISubprogram(name: "memset", scope: ![[FILE:[0-9]+]], file: ![[FILE]], type: ![[SUBROUTINE_TYPE:[0-9]+]], flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: ![[CU:[0-9]+]])
// CHECK-DAG: ![[INLINED_MEMSET]] = !DILocation(line: [[#]], column: [[#]], scope: ![[SP_FUNC_ZERO:[0-9]+]])

// CHECK-DAG: ![[LOC_MEMCPY]] = !DILocation(line: 0, scope: ![[SP_MEMCPY:[0-9]+]], inlinedAt: ![[INLINED_MEMCPY:[0-9]+]])
// CHECK-DAG: ![[SP_MEMCPY]] = distinct !DISubprogram(name: "memcpy", scope: ![[FILE]], file: ![[FILE]], type: ![[SUBROUTINE_TYPE]], flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: ![[CU]])
// CHECK-DAG: ![[INLINED_MEMCPY]] = !DILocation(line: [[#]], column: [[#]], scope: ![[SP_FUNC_CPY:[0-9]+]])

// CHECK-DAG: ![[LOC_AGG]] = !DILocation(line: 0, scope: ![[SP_MEMSET]], inlinedAt: ![[INLINED_AGG:[0-9]+]])

