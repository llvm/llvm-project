// RUN: %clang -target x86_64-linux-gnu -fsplit-stack -S %s -emit-llvm -o - | FileCheck -check-prefix=CHECK-SEGSTK %s
// RUN: %clang -target x86_64-linux-gnu -S %s -emit-llvm -o - | FileCheck -check-prefix=CHECK-NOSEGSTK %s

int foo(void) {
  return 0;
}

__attribute__((no_split_stack))
int nosplit(void) {
  return 0;
}

int main(void) {
  return foo();
}

// CHECK-SEGSTK: define dso_local i32 @foo() [[SS:#[0-9]+]] {
// CHECK-SEGSTK: define dso_local i32 @nosplit() [[NSS:#[0-9]+]] {
// CHECK-SEGSTK: define dso_local i32 @main() [[SS]] {
// CHECK-SEGSTK-NOT: [[NSS]] = { {{.*}} "split-stack" {{.*}} }
// CHECK-SEGSTK: [[SS]] = { {{.*}} "split-stack" {{.*}} }
// CHECK-SEGSTK-NOT: [[NSS]] = { {{.*}} "split-stack" {{.*}} }

// CHECK-NOSEGSTK: define dso_local i32 @foo() #0 {
// CHECK-NOSEGSTK: define dso_local i32 @main() #0 {
// CHECK-NOSEGSTK-NOT: #0 = { {{.*}} "split-stack" {{.*}} }
