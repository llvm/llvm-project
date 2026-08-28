// Check that the debug location clang has established survives the hand-off to
// the OpenMPIRBuilder, so that the IR the builder emits on clang's behalf still
// carries a !dbg attachment.

// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -triple x86_64-unknown-unknown \
// RUN:   -debug-info-kind=limited -emit-llvm %s -o - | FileCheck %s --check-prefix=GTID

// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-unknown \
// RUN:   -fopenmp-targets=x86_64-unknown-linux-gnu -debug-info-kind=limited \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=TDATA

int cond;
void use(int);

// CGOpenMPRuntime::getThreadID() defers to the OpenMPIRBuilder when it is
// enabled, so the thread-num call is emitted by the builder.

// GTID-LABEL: define {{.*}}@single_region
// GTID:       entry:
// GTID-NEXT:    call i32 @__kmpc_global_thread_num({{.*}}), !dbg
void single_region(void) {
#pragma omp single
  use(1);
}

// CGOpenMPRuntime::emitTargetDataCalls() passes the 'if' condition down to
// OpenMPIRBuilder::createTargetData(), which emits the branch on it.

// TDATA-LABEL: define {{.*}}@target_data_if
// TDATA:        %[[TOBOOL:.+]] = icmp ne i32 %{{.+}}, 0, !dbg
// TDATA-NEXT:   br i1 %[[TOBOOL]], label %{{.+}}, label %{{.+}}, !dbg
void target_data_if(int *p) {
#pragma omp target data map(tofrom : p[0 : 4]) if (cond)
  use(2);
}
