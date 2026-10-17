// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

int normal_var = 10;
// CIR: cir.global external @normal_var {{.*}}
// LLVM: @normal_var = global {{.*}}
// OGCG: @normal_var = global {{.*}}

__attribute__((visibility("hidden")))
int hidden_var = 10;
// CIR: cir.global hidden external @hidden_var {{.*}}
// LLVM: @hidden_var = hidden global {{.*}}
// OGCG: @hidden_var = hidden global {{.*}}

static int normal_static_var = 10;
// CIR: cir.global "private" internal dso_local @normal_static_var {{.*}}
// LLVM: @normal_static_var = internal global {{.*}}
// OGCG: @normal_static_var = internal global {{.*}}

void normal_func() {
    normal_var = 0;
    normal_static_var = 0;
}
// CIR: cir.func no_inline no_proto dso_local @normal_func() {{.*}} {
// LLVM: define dso_local void @normal_func() {{.*}}
// OGCG: define dso_local void @normal_func() {{.*}}

__attribute__((visibility("hidden")))
void hidden_func() {
    hidden_var = 0;
}
// CIR: cir.func no_inline no_proto hidden dso_local @hidden_func() {{.*}} {
// LLVM: define hidden void @hidden_func() {{.*}}
// OGCG: define hidden void @hidden_func() {{.*}}
