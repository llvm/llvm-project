// Only output llvm.assume(llvm.type.test()) if cfi-vcall is disabled and whole-program-vtables is enabled
// RUN: %clang_cc1 -flto -fvisibility=hidden -fsanitize=cfi-vcall -fwhole-program-vtables -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=CFI %s
// RUN: %clang_cc1 -flto -fvisibility=hidden -fwhole-program-vtables -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=NOCFI %s

struct S1 {
  virtual void f();
};

// CHECK: define{{.*}}s1f
// CFI: llvm.type.test(ptr {{.*}}, metadata !"{{.*}}S1{{.*}}")
// CFI: llvm.type.test(ptr {{.*}}, metadata !"all-vtables")
// CFI-NOT: llvm.assume
// NOCFI: llvm.type.test(ptr {{.*}}, metadata !"{{.*}}S1{{.*}}")
// NOCFI: llvm.assume
void s1f(S1 *s1) {
  s1->f();
}
