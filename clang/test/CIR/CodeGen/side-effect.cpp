// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fdeclspec -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fdeclspec -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fdeclspec -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG
extern "C" {

// FIXME: We should figure out how to better print this on functions in the
// future.
// CIR: cir.func{{.*}}@pure_func() -> !s32i attributes {{{.*}}nothrow, nounwind, willreturn, memory_effects = #cir.memory_effects<other = read, arg_mem = read, inaccessible_mem = read, errno_mem = read, target_mem0 = read, target_mem1 = read>} {
// LLVM: Function Attrs: {{.*}}nounwind{{.*}}willreturn{{.*}}memory(read)
// LLVM: define{{.*}} @pure_func() #{{.*}} {
// OGCG: Function Attrs: {{.*}}nounwind{{.*}}willreturn{{.*}}memory(read)
// OGCG: define{{.*}} @pure_func() #{{.*}} {
__attribute__((pure))
int pure_func() { return 2;}

// CIR: cir.func{{.*}}@const_func() -> !s32i attributes {{{.*}}nothrow, nounwind, willreturn, memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} {
// LLVM: Function Attrs: {{.*}}nounwind{{.*}}willreturn{{.*}}memory(none)
// LLVM: define{{.*}} @const_func() #{{.*}} {
// OGCG: Function Attrs: {{.*}}nounwind{{.*}}willreturn{{.*}}memory(none)
// OGCG: define{{.*}} @const_func() #{{.*}} {
__attribute__((const))
int const_func() { return 1;}

// CIR: cir.func{{.*}}@noalias_func(%{{.+}}: !cir.ptr<!s32i> {llvm.noundef}{{.*}}) -> !s32i attributes {{{.*}}nothrow, nounwind, memory_effects = #cir.memory_effects<other = none, arg_mem = readwrite, inaccessible_mem = readwrite, errno_mem = none, target_mem0 = none, target_mem1 = none>} {
// LLVM: Function Attrs: {{.*}}nounwind{{.*}}memory(argmem: readwrite, inaccessiblemem: readwrite)
// LLVM: define{{.*}} i32 @noalias_func(ptr noundef %{{.+}}) #{{.*}} {
// OGCG: Function Attrs: {{.*}}nounwind{{.*}}memory(argmem: readwrite, inaccessiblemem: readwrite)
// OGCG: define{{.*}} i32 @noalias_func(ptr noundef %{{.+}}) #{{.*}} {
__declspec(noalias)
int noalias_func(int *p) { return *p; }

void use() {
  // CIR: cir.call @pure_func() nounwind willreturn {memory_effects = #cir.memory_effects<other = read, arg_mem = read, inaccessible_mem = read, errno_mem = read, target_mem0 = read, target_mem1 = read>} : () -> !s32i
  // LLVM: call i32 @pure_func() #[[PURE_ATTR:.*]]
  // OGCG: call i32 @pure_func() #[[PURE_ATTR:.*]]
  pure_func();
  // CIR: cir.call @const_func() nounwind willreturn {memory_effects = #cir.memory_effects<other = none, arg_mem = none, inaccessible_mem = none, errno_mem = none, target_mem0 = none, target_mem1 = none>} : () -> !s32i
  // LLVM: call i32 @const_func() #[[CONST_ATTR:.*]]
  // OGCG: call i32 @const_func() #[[CONST_ATTR:.*]]
  const_func();
  // CIR: cir.call @noalias_func(%{{.+}}) nounwind {memory_effects = #cir.memory_effects<other = none, arg_mem = readwrite, inaccessible_mem = readwrite, errno_mem = none, target_mem0 = none, target_mem1 = none>} : (!cir.ptr<!s32i> {llvm.noundef}) -> !s32i
  // LLVM: call i32 @noalias_func(ptr noundef %{{.+}}) #[[NOALIAS_ATTR:.*]]
  // OGCG: call i32 @noalias_func(ptr noundef %{{.+}}) #[[NOALIAS_ATTR:.*]]
  int x = 0;
  noalias_func(&x);
}

// LLVM: attributes #[[PURE_ATTR]] = {{{.*}}nounwind{{.*}}willreturn{{.*}}memory(read) }
// OGCG: attributes #[[PURE_ATTR]] = {{{.*}}nounwind{{.*}}willreturn{{.*}}memory(read) }
// LLVM: attributes #[[CONST_ATTR]] = {{{.*}}nounwind{{.*}}willreturn{{.*}}memory(none) }
// OGCG: attributes #[[CONST_ATTR]] = {{{.*}}nounwind{{.*}}willreturn{{.*}}memory(none) }
// LLVM: attributes #[[NOALIAS_ATTR]] = { nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) }
// OGCG: attributes #[[NOALIAS_ATTR]] = { nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) }
}

