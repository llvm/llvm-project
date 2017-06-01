// RUN: %clang_cc1 -x c++ -std=c++11 %s -triple x86_64-unknown-linux-gnu  -main-file-name def-ctors.cpp -o - -emit-llvm -fprofile-instrument=clang |  FileCheck --check-prefix=PGOGEN %s

// RUN: %clang_cc1 -x c++ -std=c++11 %s -triple x86_64-unknown-linux-gnu -main-file-name def-ctors.cpp -o - -emit-llvm -fprofile-instrument=clang -fcoverage-mapping | FileCheck --check-prefix=COVMAP %s

struct Base {
  int B;
  Base() : B(2) {}
  Base(const struct Base &b2) {}
};

struct Derived : public Base {
  Derived(const Derived &) = default;
  // PGOGEN-DAG: define {{.*}}@_ZN7DerivedC2ERKS_
  // PGOGEN-DAG: %pgocount = load {{.*}} @__profc__ZN7DerivedC2ERKS_
  // PGOGEN-DAG: {{.*}}add{{.*}}%pgocount, 1
  // PGOGEN-DAG: store{{.*}}@__profc__ZN7DerivedC2ERKS_
  Derived() = default;
  // PGOGEN-DAG: define {{.*}}@_ZN7DerivedC2Ev
  // PGOGEN-DAG: %pgocount = load {{.*}} @__profc__ZN7DerivedC2Ev
  // PGOGEN-DAG: {{.*}}add{{.*}}%pgocount, 1
  // PGOGEN-DAG: store{{.*}}@__profc__ZN7DerivedC2Ev

  // Check that coverage mapping has 6 function records including
  // the defaulted Derived::Derived(const Derived), and Derived::Derived()
  // methds.
  // COVMAP: @__llvm_coverage_mapping = {{.*}} { { i32, i32, i32, i32 }, [5 x
  // <{{.*}}>],
};

Derived dd;
int g;
int main() {
  Derived dd2(dd);
  g = dd2.B;
  return 0;
}
