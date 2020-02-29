// RUN: %clang_cc1 -x c++ -std=c++11 %s -triple x86_64-unknown-linux-gnu -main-file-name def-assignop.cpp -o - -emit-llvm -fprofile-instrument=clang | FileCheck --check-prefix=PGOGEN %s
// RUN: %clang_cc1 -x c++ -std=c++11 %s -triple x86_64-unknown-linux-gnu -main-file-name def-assignop.cpp -o - -emit-llvm -fprofile-instrument=clang -fcoverage-mapping | FileCheck --check-prefix=COVMAP %s

struct B {
  B& operator=(const B &b);
  B& operator=(const B &&b);
};

struct A {
  A &operator=(const A &) = default;
  // PGOGEN: define {{.*}}@_ZN1AaSERKS_(
  // PGOGEN: %pgocount = load {{.*}} @__profc__ZN1AaSERKS_
  // PGOGEN: {{.*}}add{{.*}}%pgocount, 1
  // PGOGEN: store{{.*}}@__profc__ZN1AaSERKS_
  A &operator=(A &&) = default;
  // PGOGEN: define {{.*}}@_ZN1AaSEOS_
  // PGOGEN: %pgocount = load {{.*}} @__profc__ZN1AaSEOS_
  // PGOGEN: {{.*}}add{{.*}}%pgocount, 1
  // PGOGEN: store{{.*}}@__profc__ZN1AaSEOS_

  // Check that coverage mapping includes 6 function records including the
  // defaulted copy and move operators: A::operator=
  // COVMAP: @__llvm_coverage_mapping = {{.*}} { { i32, i32, i32, i32 }, [3 x <{{.*}}>],
  B b;
};

A a1, a2;
void foo() {
  a1 = a2;
  a2 = static_cast<A &&>(a1);
}
