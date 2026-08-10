// RUN: %clang_cc1 -triple x86_64-linux-unknown -emit-llvm -o - %s -fsanitize=safe-stack | FileCheck -check-prefix=SP %s

__attribute__((no_sanitize("safe-stack")))
int foo(int *a) {  return *a; }

// SP-NOT: attributes #{{.*}} = { {{.*}}safestack{{.*}} }
