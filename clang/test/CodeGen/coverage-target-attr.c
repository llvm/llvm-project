// RUN: %clang -S -emit-llvm --coverage --target=aarch64-linux-android30 -fsanitize=hwaddress %s -o %t
// RUN: FileCheck %s < %t

// CHECK: define internal void @__llvm_gcov_writeout() unnamed_addr [[ATTR:#[0-9]+]]
// CHECK: define internal void @__llvm_gcov_reset() unnamed_addr [[ATTR]]
// CHECK: define internal void @__llvm_gcov_init() unnamed_addr [[ATTR]]
// CHECK: define internal void @hwasan.module_ctor() [[ATTR2:#[0-9]+]]
// CHECK: attributes [[ATTR]] = {{.*}} "target-cpu"="generic" "target-features"="+fix-cortex-a53-835769,+fp-armv8,+neon,+outline-atomics,+tagged-globals,+v8a"
// CHECK: attributes [[ATTR2]] = {{.*}} "target-cpu"="generic" "target-features"="+fix-cortex-a53-835769,+fp-armv8,+neon,+outline-atomics,+tagged-globals,+v8a"

__attribute__((weak)) int foo = 0;

__attribute__((weak)) void bar() {}

int main() {
  if (foo) bar();
}
