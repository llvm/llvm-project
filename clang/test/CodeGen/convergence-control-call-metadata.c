// RUN: %clang_cc1 -triple spirv1.6-unknown-vulkan1.3-compute -fsanitize=alloc-token -emit-llvm -disable-llvm-passes %s -o - | FileCheck --check-prefix=ALLOC-TOKEN %s
// RUN: %clang_cc1 -triple spirv1.6-unknown-vulkan1.3-compute -fconvergent-functions -fexperimental-call-graph-section -emit-llvm -disable-llvm-passes %s -o - | FileCheck --check-prefix=CALL-GRAPH %s

// Check that metadata attached to a convergent call is preserved when it
// gets a convergence control token.

typedef __typeof(sizeof(int)) size_t;

void *my_malloc(size_t size) __attribute__((malloc, alloc_size(1), convergent));

// ALLOC-TOKEN-LABEL: @test_malloc(
// ALLOC-TOKEN: [[TOKEN:%.*]] = call token @llvm.experimental.convergence.entry()
// ALLOC-TOKEN: call spir_func noalias ptr @my_malloc(i32 noundef 4){{.*}} [ "convergencectrl"(token [[TOKEN]]) ], !alloc_token [[META_INT:![0-9]+]]
int *test_malloc(void) {
  return (int *)my_malloc(sizeof(int));
}

void (*fp)(int);

// CALL-GRAPH-LABEL: @test_indirect(
// CALL-GRAPH: [[TOKEN:%.*]] = call token @llvm.experimental.convergence.entry()
// CALL-GRAPH: call spir_func void %{{.*}}(i32 noundef %{{.*}}){{.*}} [ "convergencectrl"(token [[TOKEN]]) ], !callee_type [[CALLEE_TYPE:![0-9]+]]
void test_indirect(int x) {
  fp(x);
}

// ALLOC-TOKEN: [[META_INT]] = !{!"int", i1 false}
// CALL-GRAPH-DAG: [[CALLEE_TYPE]] = !{[[TYPE_ID:![0-9]+]]}
// CALL-GRAPH-DAG: [[TYPE_ID]] = !{!"_ZTSFviE"}
