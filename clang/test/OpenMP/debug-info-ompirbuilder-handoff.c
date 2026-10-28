// Check that the debug location clang has established survives the hand-off to
// the OpenMPIRBuilder, so that the IR the builder emits on clang's behalf still
// carries a !dbg attachment.

// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -triple x86_64-unknown-unknown \
// RUN:   -fopenmp-targets=x86_64-unknown-linux-gnu -debug-info-kind=limited \
// RUN:   -emit-llvm %s -o - | FileCheck %s

int cond;
void use(int);

// CGOpenMPRuntime::getThreadID() defers to the OpenMPIRBuilder when it is
// enabled, so the thread-num call is emitted by the builder and must inherit
// the location clang was holding.

// CHECK-LABEL: define {{.*}}@single_region
// CHECK:       entry:
// CHECK-NEXT:    call i32 @__kmpc_global_thread_num({{.*}}), !dbg ![[LOC:[0-9]+]]
// CHECK-NEXT:    call i32 @__kmpc_single({{.*}}), !dbg ![[LOC]]
void single_region(void) {
#pragma omp single
  use(1);
}

// CGOpenMPRuntime::emitTargetDataCalls() passes the 'if' condition down to
// OpenMPIRBuilder::createTargetData(), which emits the branch on it. The mapper
// calls of the region are not useful here because restoreIP() reinstalls a
// location from the instruction at the insertion point, so they keep their !dbg
// either way; this branch is emitted before that happens and is the only
// observable witness on that path.

// CHECK-LABEL: define {{.*}}@target_data_if
// CHECK:         %[[TOBOOL:.+]] = icmp ne i32 %{{.+}}, 0, !dbg ![[LOC2:[0-9]+]]
// CHECK-NEXT:    br i1 %[[TOBOOL]], label %{{.+}}, label %{{.+}}, !dbg ![[LOC2]]
void target_data_if(int *p) {
#pragma omp target data map(tofrom : p[0 : 4]) if (cond)
  use(2);
}
