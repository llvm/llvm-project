// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-icall -fsanitize-cfi-cross-dso -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,NOCANON %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-icall -fsanitize-cfi-cross-dso -fsanitize-cfi-canonical-jump-tables -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,CANON %s

void ext(void);

// CHECK: define{{.*}} void @f({{.*}} [[ATTR1:#[0-9]+]]
void f() {
  ext();
}

// NOCANON: declare !type {{.*}} @ext()
// CANON: declare void @ext()

// CHECK: define{{.*}} void @g({{.*}} [[ATTR2:#[0-9]+]]
__attribute__((cfi_canonical_jump_table)) void g() {}

// CHECK: [[ATTR1]] = {
// CHECK-NOT: "cfi-canonical-jump-table"
// CHECK: }

// CHECK: [[ATTR2]] = { {{.*}} "cfi-canonical-jump-table" {{.*}} }

// NOCANON: !{i32 4, !"CFI Canonical Jump Tables", i32 0}
// CANON: !{i32 4, !"CFI Canonical Jump Tables", i32 1}
