// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG
extern "C" {

// FIXME: We should figure out how to better print this on functions in the
// future.
// CIR: cir.func{{.*}}@pure_func() -> !s32i side_effect(pure) {
// LLVM: Function Attrs: {{.*}}nounwind{{.*}}willreturn{{.*}}memory(read)
// LLVM: define{{.*}} @pure_func() #{{.*}} {
// OGCG: Function Attrs: {{.*}}nounwind{{.*}}willreturn{{.*}}memory(read)
// OGCG: define{{.*}} @pure_func() #{{.*}} {
__attribute__((pure))
int pure_func() { return 2;}

// CIR: cir.func{{.*}}@const_func() -> !s32i side_effect(const) {
// LLVM: Function Attrs: {{.*}}nounwind{{.*}}willreturn{{.*}}memory(none)
// LLVM: define{{.*}} @const_func() #{{.*}} {
// OGCG: Function Attrs: {{.*}}nounwind{{.*}}willreturn{{.*}}memory(none)
// OGCG: define{{.*}} @const_func() #{{.*}} {
__attribute__((const))
int const_func() { return 1;}

void use() {
  // CIR: cir.call @pure_func() side_effect(pure) : () -> !s32i
  // LLVM: call i32 @pure_func() #[[PURE_ATTR:.*]]
  // OGCG: call i32 @pure_func() #[[PURE_ATTR:.*]]
  pure_func();
  // CIR: cir.call @const_func() side_effect(const) : () -> !s32i
  // LLVM: call i32 @const_func() #[[CONST_ATTR:.*]]
  // OGCG: call i32 @const_func() #[[CONST_ATTR:.*]]
  const_func();
}

// LLVM: attributes #[[PURE_ATTR]] = {{{.*}}nounwind{{.*}}willreturn{{.*}}memory(read) }
// OGCG: attributes #[[PURE_ATTR]] = {{{.*}}nounwind{{.*}}willreturn{{.*}}memory(read) }
// LLVM: attributes #[[CONST_ATTR]] = {{{.*}}nounwind{{.*}}willreturn{{.*}}memory(none) }
// OGCG: attributes #[[CONST_ATTR]] = {{{.*}}nounwind{{.*}}willreturn{{.*}}memory(none) }
}

