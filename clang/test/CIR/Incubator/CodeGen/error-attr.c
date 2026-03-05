// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-og.ll
// RUN: FileCheck --input-file=%t-og.ll --check-prefix=OGCG %s

// Test __attribute__((error("msg")))
__attribute__((error("This function should not be called")))
void error_function(void) {}

// CIR: #cir<extra({{.*}}dontcall = #cir.dontcall<"This function should not be called", true>
// LLVM: define{{.*}}@error_function{{.*}}#[[#ATTR_ERROR:]]
// OGCG: define{{.*}}@error_function{{.*}}#[[#OGCG_ATTR_ERROR:]]

// Test __attribute__((warning("msg")))
__attribute__((warning("This function is deprecated")))
void warning_function(void) {}

// CIR: #cir<extra({{.*}}dontcall = #cir.dontcall<"This function is deprecated", false>
// LLVM: define{{.*}}@warning_function{{.*}}#[[#ATTR_WARNING:]]
// OGCG: define{{.*}}@warning_function{{.*}}#[[#OGCG_ATTR_WARNING:]]

// LLVM-DAG: attributes #[[#ATTR_ERROR]] = {{.*}}"dontcall-error"="This function should not be called"
// LLVM-DAG: attributes #[[#ATTR_WARNING]] = {{.*}}"dontcall-warn"="This function is deprecated"
// OGCG-DAG: attributes #[[#OGCG_ATTR_ERROR]] = {{.*}}"dontcall-error"="This function should not be called"
// OGCG-DAG: attributes #[[#OGCG_ATTR_WARNING]] = {{.*}}"dontcall-warn"="This function is deprecated"
