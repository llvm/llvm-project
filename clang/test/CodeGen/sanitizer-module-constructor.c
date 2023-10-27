// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=address -O3 -emit-llvm -fdebug-pass-manager -o - %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=thread -O3 -emit-llvm -fdebug-pass-manager -o - %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=memory -O3 -emit-llvm -fdebug-pass-manager -o - %s 2>&1 | FileCheck %s

// This is regression test for PR42877

typedef struct a *b;
struct a {
  int c;
};
int d;
b e;
static void f(b g) {
  for (d = g->c;;)
    ;
}
void h(void) { f(e); }

// CHECK: Running pass: {{.*}}SanitizerPass
// CHECK-NOT: Running pass: LoopSimplifyPass on {{.*}}san.module_ctor
// CHECK: Running analysis: TargetLibraryAnalysis on {{.*}}san.module_ctor
