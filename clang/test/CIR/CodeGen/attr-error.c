// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

__attribute__((error("don't call"))) void err(void);
__attribute__((warning("don't call me maybe"))) void warn(void);

void bar(void) {
  err();
  warn();
}

// CIR-LABEL: cir.func {{.*}}@bar
// LLVM-LABEL: define{{.*}}@bar

// CIR: cir.call @err() {cir.srcloc = {{[0-9]+}} : i64}
// LLVM: call void @err(), !srcloc [[SRC_LOC_ERR:![0-9]+]]

// CIR: cir.call @warn() {cir.srcloc = {{[0-9]+}} : i64}
// LLVM: call void @warn(), !srcloc [[SRC_LOC_WARN:![0-9]+]]

// CIR: cir.func private @err() attributes {{.*}}"dontcall-error" = "don't call"{{.*}}
// LLVM: declare{{.*}} void @err() [[ATTR_ERROR:#[0-9]+]]

// CIR: cir.func private @warn() attributes {{.*}}"dontcall-warn" = "don't call me maybe"{{.*}}
// LLVM: declare{{.*}} void @warn() [[ATTR_WARN:#[0-9]+]]

// LLVM-DAG: attributes [[ATTR_ERROR]] ={{.*}}"dontcall-error"="don't call"
// LLVM-DAG: attributes [[ATTR_WARN]] ={{.*}}"dontcall-warn"="don't call me maybe"
// LLVM-DAG: [[SRC_LOC_ERR]] = !{i64 {{[0-9]+}}}
// LLVM-DAG: [[SRC_LOC_WARN]] = !{i64 {{[0-9]+}}}
