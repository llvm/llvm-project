// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o - -mstack-alignment=64 %s | FileCheck %s

// CHECK-LABEL: define void @other()
// CHECK: [[OTHER:#[0-9]+]]
// CHECK: {
void other(void) {}

// CHECK-LABEL: define i32 @main(
// CHECK: [[MAIN:#[0-9]+]]
// CHECK: {
int main(int argc, char **argv) {
  other();
  return 0;
}

// CHECK: attributes [[OTHER]] = { noinline nounwind optnone
// CHECK-NOT: "stackrealign"
// CHECK: }
// CHECK: attributes [[MAIN]] = { noinline nounwind optnone {{.*}}"stackrealign"{{.*}} }
