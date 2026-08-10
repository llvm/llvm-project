// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-linux -O2 \
// RUN:   -finline-functions -emit-llvm -disable-llvm-passes -o - \
// RUN: | FileCheck -allow-deprecated-dag-overlap %s \
// RUN:   --check-prefix=CHECK --check-prefix=SUITABLE
// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-linux -O2 \
// RUN:   -finline-hint-functions -emit-llvm -disable-llvm-passes -o - \
// RUN: | FileCheck -allow-deprecated-dag-overlap %s \
// RUN:   --check-prefix=CHECK --check-prefix=HINTED
// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-linux -O2 \
// RUN:   -fno-inline -emit-llvm -disable-llvm-passes -o - \
// RUN: | FileCheck -allow-deprecated-dag-overlap %s \
// RUN:   --check-prefix=CHECK --check-prefix=NOINLINE

struct A {
  inline void int_run(int);

  template <class T>
  inline void template_run(T);
};

// CHECK: @_ZN1A7int_runEi({{.*}}) [[ATTR:#[0-9]+]]
void A::int_run(int) {}
// CHECK: @_ZN1A12template_runIiEEvT_({{.*}}) [[ATTR]]
template <typename T>
void A::template_run(T) {}

void bar() {
  A().int_run(1);
  A().template_run(1);
}

// SUITABLE: attributes [[ATTR]] = { {{.*}}inlinehint{{.*}} }
//   HINTED: attributes [[ATTR]] = { {{.*}}inlinehint{{.*}} }
// NOINLINE: attributes [[ATTR]] = { {{.*}}noinline{{.*}} }
