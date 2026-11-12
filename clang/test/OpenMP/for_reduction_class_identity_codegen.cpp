// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

struct Complex {
  double Re;
  double Im;
  Complex(double Re = 0.0, double Im = 0.0) : Re(Re), Im(Im) {}
  Complex &operator+=(const Complex &RHS) {
    Re += RHS.Re;
    Im += RHS.Im;
    return *this;
  }
  Complex &operator*=(const Complex &RHS) {
    double NewRe = Re * RHS.Re - Im * RHS.Im;
    Im = Re * RHS.Im + Im * RHS.Re;
    Re = NewRe;
    return *this;
  }
};

void add_reduction(Complex *Data) {
  Complex Sum;
#pragma omp parallel for reduction(+ : Sum)
  for (int I = 0; I < 4; ++I)
    Sum += Data[I];
}

void mul_reduction(Complex *Data) {
  Complex Product;
#pragma omp parallel for reduction(* : Product)
  for (int I = 0; I < 4; ++I)
    Product *= Data[I];
}

// CHECK-LABEL: define {{.*}}void @_Z13add_reductionP7Complex
// CHECK: call {{.*}} @_ZN7ComplexC1Edd(ptr {{[^,]*}}%{{[^,]*}}, double {{.*}}0.000000e+00, double {{.*}}0.000000e+00)
// CHECK-LABEL: define internal {{.*}}void @_Z13add_reductionP7Complex.omp_outlined
// CHECK: call {{.*}} @_ZN7ComplexC1Edd(ptr {{[^,]*}}%{{[^,]*}}, double {{.*}}0.000000e+00, double {{.*}}0.000000e+00)

// CHECK-LABEL: define {{.*}}void @_Z13mul_reductionP7Complex
// CHECK: call {{.*}} @_ZN7ComplexC1Edd(ptr {{[^,]*}}%{{[^,]*}}, double {{.*}}0.000000e+00, double {{.*}}0.000000e+00)
// CHECK-LABEL: define internal {{.*}}void @_Z13mul_reductionP7Complex.omp_outlined
// CHECK: call {{.*}} @_ZN7ComplexC1Edd(ptr {{[^,]*}}%{{[^,]*}}, double {{.*}}1.000000e+00, double {{.*}}0.000000e+00)
