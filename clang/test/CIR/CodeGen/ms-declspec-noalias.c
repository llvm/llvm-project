// CIR does not yet implement the Windows C++ ABI, so exercise __declspec(noalias)
// on a Linux triple with -fms-extensions.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fms-extensions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fms-extensions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fms-extensions -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

__declspec(noalias) void noalias_callee(int *x);
void noalias_caller(int *x) { noalias_callee(x); }

// CIR: cir.func {{.*}} @noalias_caller
// CIR:   cir.call @noalias_callee(%{{.*}}) nothrow side_effect(argmem)
// CIR: cir.func {{.*}}@noalias_callee(!cir.ptr<!s32i> {{.*}}) side_effect(argmem)

// LLVM: call void @noalias_callee({{.*}}) [[NA:#[0-9]+]]
// OGCG: call void @noalias_callee({{.*}}) [[NA:#[0-9]+]]

__declspec(noalias) void noalias_def(int *x) {}

// CIR: cir.func {{.*}} @noalias_def(%{{.*}}) side_effect(argmem)

// LLVM: define dso_local void @noalias_def({{.*}}) [[NA_DEF:#[0-9]+]]
// OGCG: define dso_local void @noalias_def({{.*}}) [[NA_DEF:#[0-9]+]]

// LLVM-DAG: attributes [[NA]] = { nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
// OGCG-DAG: attributes [[NA]] = { nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
// LLVM-DAG: attributes [[NA_DEF]] = { {{.*}}nounwind{{.*}}memory(argmem: readwrite, inaccessiblemem: readwrite)
// OGCG-DAG: attributes [[NA_DEF]] = { {{.*}}nounwind{{.*}}memory(argmem: readwrite, inaccessiblemem: readwrite)
