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

// Def attribute groups are emitted before the call-site group. Reject
// willreturn anywhere on those lines; NoAliasAttr must not add it.
// LLVM: attributes [[NA_DEF]] = {
// LLVM-NOT: willreturn
// LLVM-SAME: nounwind
// LLVM-NOT: willreturn
// LLVM-SAME: memory(argmem: readwrite, inaccessiblemem: readwrite)
// LLVM-NOT: willreturn
// LLVM-SAME: }
// LLVM: attributes [[NA]] = {
// LLVM-NOT: willreturn
// LLVM-SAME: nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) }

// OGCG: attributes [[NA_DEF]] = {
// OGCG-NOT: willreturn
// OGCG-SAME: nounwind
// OGCG-NOT: willreturn
// OGCG-SAME: memory(argmem: readwrite, inaccessiblemem: readwrite)
// OGCG-NOT: willreturn
// OGCG-SAME: }
// OGCG: attributes [[NA]] = {
// OGCG-NOT: willreturn
// OGCG-SAME: nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) }
