// RUN: %clang_cc1 -x c++ -std=c++11 %s -triple x86_64-unknown-linux-gnu -main-file-name def-dtors.cpp -o - -emit-llvm -fprofile-instrument=clang  | FileCheck --check-prefix=PGOGEN %s

// RUN: %clang_cc1 -x c++ -std=c++11 %s -triple x86_64-unknown-linux-gnu -main-file-name def-dtors.cpp -o - -emit-llvm -fprofile-instrument=clang -fcoverage-mapping | FileCheck --check-prefix=COVMAP %s

struct Base {
  int B;
  Base(int B_) : B(B_) {}
  ~Base() {}
};

struct Derived : public Base {
  Derived(int K) : Base(K), I(K), J(K) {}
  ~Derived() = default;
  // PGOGEN-LABEL: define {{.*}}@_ZN7DerivedD2Ev
  // PGOGEN: %pgocount = load {{.*}} @__profc__ZN7DerivedD2Ev
  // PGOGEN: {{.*}}add{{.*}}%pgocount, 1
  // PGOGEN: store{{.*}}@__profc__ZN7DerivedD2Ev

  // Check that coverage mapping has 6 function records including
  // the default destructor in the derived class.
  // COVMAP: @__llvm_coverage_mapping = {{.*}} { { i32, i32, i32, i32 }, [6 x
  // <{{.*}}>],

  int I;
  int J;
  int getI() { return I; }
};

Derived dd(100);
int g;
int main() {
  Derived dd2(dd.getI());
  g = dd2.getI();
  return 0;
}
